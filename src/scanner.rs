use std::sync::Mutex;

use opencv::core::{Point2f, Size, Vec3b, Vector};
use opencv::prelude::*;
use opencv::{geometry, imgcodecs, imgproc};
use ort::session::Session;
use ort::value::Tensor;

use crate::error::AppError;

/// Result of a document scan: the perspective-corrected, encoded image bytes
/// and whether the model actually found a document (vs. falling back to the
/// full frame).
pub struct ScanOutput {
    pub jpeg_bytes: Vec<u8>,
    pub corners_detected: bool,
    pub width: i32,
    pub height: i32,
    /// Corners used for the crop, in the *original* image's pixel space,
    /// ordered [top-left, top-right, bottom-right, bottom-left]. Present
    /// even when detection failed (falls back to the full-image rect) so
    /// callers always have a starting quad to show/adjust.
    pub corners: [(f32, f32); 4],
    pub orig_width: i32,
    pub orig_height: i32,
}

/// Wraps the loaded corner-detection network. `dnn::Net` is a handle to
/// mutable OpenCV state (`forward()` needs `&mut self`), so concurrent
/// requests share one instance behind a mutex rather than each loading
/// their own copy — reloading from disk is pure overhead once the weights
/// are already in memory.
pub struct DocDetector(Mutex<Session>);

/// DocAligner's `lcnet100` heatmap-regression model (Apache-2.0,
/// github.com/DocsaidLab/DocAligner): a PP-LCNet-1.0 backbone + BiFPN neck
/// predicting one heatmap per corner rather than regressing point
/// coordinates directly. DocAligner's own README documents why: a
/// point-regression head trained at a fixed low resolution (256x256)
/// suffers "amplification error" — a ~5-10px shift once the predicted
/// point is scaled back up to the original photo's resolution, since a
/// whole neighborhood of source pixels collapses to one predicted pixel.
/// A heatmap only has to say "the corner is roughly here", so upscaling
/// its centroid to the original resolution doesn't carry that fixed
/// pixel-error term. In practice, an A/B test against the original
/// point-regression `lcnet050` and the much heavier `fastvit_sa24` variant
/// found identical field-extraction accuracy across all three on a
/// 105-image batch — so `lcnet100` was kept for being the same weight class
/// as the original model with no measured downside, not for a proven
/// accuracy win. Runs at the same fixed 256x256 input as the point model it
/// replaces.
const MODEL_PATH: &str = "models/docaligner_lcnet100.onnx";
const MODEL_INPUT_SIZE: i32 = 256;
/// Below this, a corner's heatmap is treated as empty (no document edge
/// found there) — matches DocAligner's own reference postprocessing
/// (`heatmap_threshold` in `heatmap_reg/infer.py`).
const HEATMAP_THRESHOLD: f64 = 0.3;

impl DocDetector {
    pub fn load() -> Result<Self, AppError> {
        let net = Session::builder()?.commit_from_file(MODEL_PATH)?;
        Ok(Self(Mutex::new(net)))
    }

    /// Runs the model on `image` (full resolution — the fixed 256x256 input
    /// means there's no separate "detection copy" to manage) and returns
    /// its 4 corners in `image`'s own pixel space, or `None` if any of the
    /// 4 corners' heatmaps came back empty (no document found there).
    fn detect(&self, image: &Mat) -> Result<Option<[Point2f; 4]>, AppError> {
        let size = image.size().map_err(AppError::Processing)?;
        let (orig_w, orig_h) = (size.width, size.height);

        // Same preprocessing as the point model this replaces: scale to
        // [0,1], no mean/std subtraction (DocAligner's heatmap preprocess
        // is a plain `/255`), BGR->RGB since imdecode gives BGR and the
        // model was trained on RGB.
        let mut resized_bgr = Mat::default();
        imgproc::resize(
            image,
            &mut resized_bgr,
            Size::new(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        let mut resized_rgb = Mat::default();
        imgproc::cvt_color_def(&resized_bgr, &mut resized_rgb, imgproc::COLOR_BGR2RGB)?;
        let data = scale_nchw(&resized_rgb)?;
        let input = Tensor::from_array((
            [1usize, 3, MODEL_INPUT_SIZE as usize, MODEL_INPUT_SIZE as usize],
            data,
        ))?;

        let debug = std::env::var("SCAN_DEBUG_TIMING").is_ok();
        let t_fwd = std::time::Instant::now();
        let mut net = self.0.lock().expect("ort Session mutex poisoned");
        let outputs = net.run(ort::inputs!["img" => input])?;
        // heatmap is [1, 4, H, W] — one channel per corner, in
        // [top-left, top-right, bottom-right, bottom-left] order (verified
        // against a real scan: each channel's peak lands where that corner
        // visually is). Already sigmoid-activated by the exported graph.
        let (shape, heatmap) = outputs["heatmap"].try_extract_tensor::<f32>()?;
        let (map_h, map_w) = (shape[2] as i32, shape[3] as i32);
        let heatmap = heatmap.to_vec();
        drop(outputs);
        drop(net);
        if debug {
            eprintln!("[scan_timing]   detect/forward: {:?}", t_fwd.elapsed());
        }

        let t_post = std::time::Instant::now();
        let mut corners = [Point2f::new(0.0, 0.0); 4];
        for (channel, corner) in corners.iter_mut().enumerate() {
            let Some(point) =
                largest_heatmap_blob_centroid(&heatmap, channel, map_h, map_w, orig_w, orig_h)?
            else {
                return Ok(None);
            };
            *corner = point;
        }
        if debug {
            eprintln!(
                "[scan_timing]   detect/heatmap_postprocess (x4): {:?}",
                t_post.elapsed()
            );
        }

        let corners = expand_quad(order_corners(&corners), orig_w as f32, orig_h as f32);
        Ok(Some(corners))
    }
}

/// Extracts channel `channel` of a `[1, 4, H, W]` heatmap tensor, upscales
/// it to the original image's resolution, thresholds it, and returns the
/// centroid of its largest connected blob — DocAligner's own
/// heatmap-to-corner postprocessing (`heatmap_reg/infer.py`: resize,
/// threshold, binarize, take the largest-area polygon's centroid).
/// Upscaling *before* thresholding (rather than finding a peak at the
/// heatmap's native low resolution and scaling that single point up) is
/// what avoids reintroducing the same fixed pixel-error the heatmap
/// architecture exists to avoid.
fn largest_heatmap_blob_centroid(
    heatmap: &[f32],
    channel: usize,
    map_h: i32,
    map_w: i32,
    orig_w: i32,
    orig_h: i32,
) -> Result<Option<Point2f>, AppError> {
    let plane_len = (map_h * map_w) as usize;
    // heatmap is a contiguous [1, 4, H, W] tensor fresh off the network, so
    // it can be read as one flat row-major slice and sliced per channel
    // rather than indexed element-by-element.
    let start = channel * plane_len;
    let plane = &heatmap[start..start + plane_len];
    let plane_mat = Mat::new_rows_cols_with_data(map_h, map_w, plane)?.try_clone()?;

    let mut upscaled = Mat::default();
    imgproc::resize(
        &plane_mat,
        &mut upscaled,
        Size::new(orig_w, orig_h),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let mut binary = Mat::default();
    imgproc::threshold(
        &upscaled,
        &mut binary,
        HEATMAP_THRESHOLD,
        255.0,
        imgproc::THRESH_BINARY,
    )?;
    let mut binary_u8 = Mat::default();
    binary.convert_to(&mut binary_u8, opencv::core::CV_8U, 1.0, 0.0)?;

    let mut contours = Vector::<Vector<opencv::core::Point>>::new();
    imgproc::find_contours_def(
        &binary_u8,
        &mut contours,
        imgproc::RETR_LIST,
        imgproc::CHAIN_APPROX_SIMPLE,
    )?;

    let mut best: Option<(f64, opencv::core::Moments)> = None;
    for contour in &contours {
        let area = geometry::contour_area_def(&contour)?;
        if area <= 0.0 {
            continue;
        }
        if best.as_ref().is_none_or(|(best_area, _)| area > *best_area) {
            let moments = geometry::moments(&contour, false)?;
            best = Some((area, moments));
        }
    }

    let Some((_, moments)) = best else {
        return Ok(None);
    };
    if moments.m00 == 0.0 {
        return Ok(None);
    }
    Ok(Some(Point2f::new(
        (moments.m10 / moments.m00) as f32,
        (moments.m01 / moments.m00) as f32,
    )))
}

/// The model's predicted edge tends to land a hair inside the document's
/// true boundary rather than exactly on it (regression models trained on
/// human-labeled corners systematically do this — annotators tend to click
/// slightly inside a busy edge/text region), which was visibly clipping a
/// sliver of text on some scans. Pushes each corner outward from the quad's
/// own center by [`CORNER_EXPAND_FRACTION`], clamped back to the image
/// bounds, to compensate.
const CORNER_EXPAND_FRACTION: f32 = 0.025;
const MAX_CROP_DIMENSION: f64 = 16_384.0;
const MIN_CROP_EDGE: f64 = 2.0;

fn expand_quad(corners: [Point2f; 4], width: f32, height: f32) -> [Point2f; 4] {
    let cx = corners.iter().map(|p| p.x).sum::<f32>() / 4.0;
    let cy = corners.iter().map(|p| p.y).sum::<f32>() / 4.0;
    corners.map(|p| {
        let x = cx + (p.x - cx) * (1.0 + CORNER_EXPAND_FRACTION);
        let y = cy + (p.y - cy) * (1.0 + CORNER_EXPAND_FRACTION);
        Point2f::new(x.clamp(0.0, width - 1.0), y.clamp(0.0, height - 1.0))
    })
}

/// Packs a resized RGB `u8` `Mat` into an NCHW `f32` tensor, scaled to
/// `[0,1]` with no mean/std subtraction — DocAligner's heatmap preprocess.
fn scale_nchw(rgb: &Mat) -> Result<Vec<f32>, AppError> {
    let rows = rgb.rows();
    let cols = rgb.cols();
    let pixels = rgb.data_typed::<Vec3b>()?;
    let plane = (rows * cols) as usize;
    let mut out = vec![0f32; 3 * plane];
    for (index, pixel) in pixels.iter().enumerate() {
        for channel in 0..3 {
            out[channel * plane + index] = f32::from(pixel[channel]) / 255.0;
        }
    }
    Ok(out)
}

pub fn scan_document(detector: &DocDetector, bytes: &[u8]) -> Result<ScanOutput, AppError> {
    let debug = std::env::var("SCAN_DEBUG_TIMING").is_ok();
    let t0 = std::time::Instant::now();
    let buf = Vector::<u8>::from_slice(bytes);
    let original = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR).map_err(AppError::Decode)?;
    if original.empty() {
        return Err(AppError::EmptyImage);
    }
    let orig_size = original.size().map_err(AppError::Processing)?;
    if debug {
        eprintln!(
            "[scan_timing] decode: {:?} ({}x{})",
            t0.elapsed(),
            orig_size.width,
            orig_size.height
        );
    }

    let t1 = std::time::Instant::now();
    let (corners_full, corners_detected) = match detector.detect(&original)? {
        Some(corners) => (corners, true),
        None => (full_frame_corners(&original)?, false),
    };
    if debug {
        eprintln!("[scan_timing] detect: {:?}", t1.elapsed());
    }

    let t2 = std::time::Instant::now();
    let warped = warp_document(&original, &corners_full)?;
    let out_size = warped.size().map_err(AppError::Processing)?;
    if debug {
        eprintln!("[scan_timing] warp: {:?}", t2.elapsed());
    }

    let t3 = std::time::Instant::now();
    let mut params = Vector::<i32>::new();
    params.push(imgcodecs::IMWRITE_JPEG_QUALITY);
    params.push(92);
    let mut out_buf = Vector::<u8>::new();
    imgcodecs::imencode(".jpg", &warped, &mut out_buf, &params)?;
    if debug {
        eprintln!("[scan_timing] encode: {:?}", t3.elapsed());
        eprintln!("[scan_timing] total: {:?}", t0.elapsed());
    }

    Ok(ScanOutput {
        jpeg_bytes: out_buf.to_vec(),
        corners_detected,
        width: out_size.width,
        height: out_size.height,
        corners: corners_full.map(|p| (p.x, p.y)),
        orig_width: orig_size.width,
        orig_height: orig_size.height,
    })
}

/// Re-warps an already-decoded original image against caller-supplied
/// corners (e.g. after the user manually adjusted the auto-detected crop).
/// Corners are in the original image's pixel space, any order, and need not
/// be axis-aligned; they are used as-is for the perspective transform.
pub fn crop_with_corners(bytes: &[u8], corners: [(f32, f32); 4]) -> Result<ScanOutput, AppError> {
    let buf = Vector::<u8>::from_slice(bytes);
    let original = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR).map_err(AppError::Decode)?;
    if original.empty() {
        return Err(AppError::EmptyImage);
    }
    let orig_size = original.size().map_err(AppError::Processing)?;

    let corners_full = corners.map(|(x, y)| Point2f::new(x, y));
    validate_crop(&corners_full, orig_size.width, orig_size.height)?;
    let warped = warp_document(&original, &corners_full)?;
    let out_size = warped.size().map_err(AppError::Processing)?;

    let mut params = Vector::<i32>::new();
    params.push(imgcodecs::IMWRITE_JPEG_QUALITY);
    params.push(92);
    let mut out_buf = Vector::<u8>::new();
    imgcodecs::imencode(".jpg", &warped, &mut out_buf, &params)?;

    Ok(ScanOutput {
        jpeg_bytes: out_buf.to_vec(),
        corners_detected: true,
        width: out_size.width,
        height: out_size.height,
        corners: corners_full.map(|p| (p.x, p.y)),
        orig_width: orig_size.width,
        orig_height: orig_size.height,
    })
}

fn validate_crop(
    corners: &[Point2f; 4],
    image_width: i32,
    image_height: i32,
) -> Result<(), AppError> {
    let max_x = (image_width - 1) as f32;
    let max_y = (image_height - 1) as f32;
    if corners.iter().any(|point| {
        !point.x.is_finite()
            || !point.y.is_finite()
            || point.x < 0.0
            || point.y < 0.0
            || point.x > max_x
            || point.y > max_y
    }) {
        return Err(AppError::InvalidCrop(
            "crop corners must be finite and inside the original image",
        ));
    }

    let [tl, tr, br, bl] = *corners;
    let width = dist(br, bl).max(dist(tr, tl));
    let height = dist(tr, br).max(dist(tl, bl));
    if width < MIN_CROP_EDGE || height < MIN_CROP_EDGE {
        return Err(AppError::InvalidCrop("crop is too small"));
    }
    if width > MAX_CROP_DIMENSION || height > MAX_CROP_DIMENSION {
        return Err(AppError::InvalidCrop("crop dimensions are too large"));
    }

    let twice_area = corners
        .iter()
        .zip(corners.iter().cycle().skip(1))
        .take(corners.len())
        .map(|(a, b)| a.x * b.y - b.x * a.y)
        .sum::<f32>()
        .abs();
    if twice_area < 4.0 {
        return Err(AppError::InvalidCrop("crop corners are degenerate"));
    }

    Ok(())
}

fn full_frame_corners(image: &Mat) -> Result<[Point2f; 4], AppError> {
    let size = image.size().map_err(AppError::Processing)?;
    let (w, h) = (size.width as f32, size.height as f32);
    Ok([
        Point2f::new(0.0, 0.0),
        Point2f::new(w - 1.0, 0.0),
        Point2f::new(w - 1.0, h - 1.0),
        Point2f::new(0.0, h - 1.0),
    ])
}

/// Orders 4 arbitrary points as [top-left, top-right, bottom-right, bottom-left]
/// using the sum/difference trick: top-left has the smallest x+y, bottom-right
/// the largest; top-right has the smallest x-y, bottom-left the largest.
fn order_corners(points: &[Point2f; 4]) -> [Point2f; 4] {
    let top_left = *points
        .iter()
        .min_by(|a, b| (a.x + a.y).total_cmp(&(b.x + b.y)))
        .expect("exactly 4 points");
    let bottom_right = *points
        .iter()
        .max_by(|a, b| (a.x + a.y).total_cmp(&(b.x + b.y)))
        .expect("exactly 4 points");
    let top_right = *points
        .iter()
        .max_by(|a, b| (a.x - a.y).total_cmp(&(b.x - b.y)))
        .expect("exactly 4 points");
    let bottom_left = *points
        .iter()
        .min_by(|a, b| (a.x - a.y).total_cmp(&(b.x - b.y)))
        .expect("exactly 4 points");

    [top_left, top_right, bottom_right, bottom_left]
}

fn dist(a: Point2f, b: Point2f) -> f64 {
    (((a.x - b.x).powi(2) + (a.y - b.y).powi(2)) as f64).sqrt()
}

/// Applies a perspective transform so the quadrilateral defined by `corners`
/// (top-left, top-right, bottom-right, bottom-left) fills a fresh
/// axis-aligned output image, i.e. a top-down "scan" of the document.
fn warp_document(image: &Mat, corners: &[Point2f]) -> Result<Mat, AppError> {
    let [tl, tr, br, bl] = [corners[0], corners[1], corners[2], corners[3]];

    // Corner coordinates address pixel centers and are inclusive. A full-width
    // edge from x=0 through x=2159 therefore contains 2160 pixels even though
    // the Euclidean distance between the endpoint centers is 2159.
    let width = dist(br, bl).max(dist(tr, tl)).round() as i32 + 1;
    let height = dist(tr, br).max(dist(tl, bl)).round() as i32 + 1;
    let width = width.max(1);
    let height = height.max(1);

    let src = [tl, tr, br, bl];
    let dst = [
        Point2f::new(0.0, 0.0),
        Point2f::new((width - 1) as f32, 0.0),
        Point2f::new((width - 1) as f32, (height - 1) as f32),
        Point2f::new(0.0, (height - 1) as f32),
    ];

    let transform = geometry::get_perspective_transform_slice_def(src, dst)?;

    let mut warped = Mat::default();
    imgproc::warp_perspective_def(image, &mut warped, &transform, Size::new(width, height))?;

    Ok(warped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_valid_crop() {
        let corners = [
            Point2f::new(10.0, 10.0),
            Point2f::new(90.0, 10.0),
            Point2f::new(90.0, 90.0),
            Point2f::new(10.0, 90.0),
        ];

        validate_crop(&corners, 100, 100).expect("valid crop");
    }

    #[test]
    fn rejects_non_finite_and_out_of_bounds_crops() {
        let non_finite = [
            Point2f::new(f32::NAN, 10.0),
            Point2f::new(90.0, 10.0),
            Point2f::new(90.0, 90.0),
            Point2f::new(10.0, 90.0),
        ];
        let out_of_bounds = [
            Point2f::new(10.0, 10.0),
            Point2f::new(100.0, 10.0),
            Point2f::new(90.0, 90.0),
            Point2f::new(10.0, 90.0),
        ];

        assert!(validate_crop(&non_finite, 100, 100).is_err());
        assert!(validate_crop(&out_of_bounds, 100, 100).is_err());
    }

    #[test]
    fn rejects_degenerate_crops() {
        let corners = [
            Point2f::new(10.0, 10.0),
            Point2f::new(20.0, 20.0),
            Point2f::new(30.0, 30.0),
            Point2f::new(40.0, 40.0),
        ];

        assert!(validate_crop(&corners, 100, 100).is_err());
    }

    #[test]
    fn full_frame_warp_preserves_original_pixel_dimensions() {
        let image =
            Mat::new_rows_cols_with_default(
                80,
                120,
                opencv::core::CV_8UC3,
                opencv::core::Scalar::all(0.0),
            )
                .expect("test image");
        let corners = full_frame_corners(&image).expect("full-frame corners");

        let warped = warp_document(&image, &corners).expect("full-frame warp");

        assert_eq!(warped.cols(), 120);
        assert_eq!(warped.rows(), 80);
    }
}
