use std::sync::Mutex;

use opencv::core::{Point2f, Size, Vec3b, Vector};
use opencv::prelude::*;
use opencv::{geometry, imgcodecs, imgproc};
use ort::session::Session;
use ort::value::Tensor;

use crate::error::AppError;

/// Perspective-corrected JPEG of the card, plus whether corners were actually detected.
pub struct ScanOutput {
    pub jpeg_bytes: Vec<u8>,
    pub corners_detected: bool,
    pub width: i32,
    pub height: i32,
    /// Crop corners in original-image pixels, TL/TR/BR/BL. Full-image rect if detection failed.
    pub corners: [(f32, f32); 4],
    pub orig_width: i32,
    pub orig_height: i32,
}

/// Card-corner model; one shared session behind a mutex (ONNX `run` needs `&mut`).
pub struct DocDetector(Mutex<Session>);

/// DocAligner `lcnet100` (Apache-2.0): one heatmap per corner instead of point regression, which
/// avoids the ~5-10px error of upscaling a 256x256 point prediction. Same accuracy as the
/// alternatives in an A/B on 105 images; kept for its size.
const MODEL_PATH: &str = "models/docaligner_lcnet100.onnx";
const MODEL_INPUT_SIZE: i32 = 256;
/// Heatmap values below this count as empty (DocAligner reference value).
const HEATMAP_THRESHOLD: f64 = 0.3;

impl DocDetector {
    pub fn load() -> Result<Self, AppError> {
        let net = crate::ort_config::session_builder()?.commit_from_file(MODEL_PATH)?;
        Ok(Self(Mutex::new(net)))
    }

    /// The 4 corners in `image` pixels, or `None` if any corner heatmap is empty.
    fn detect(&self, image: &Mat) -> Result<Option<[Point2f; 4]>, AppError> {
        let size = image.size().map_err(AppError::Processing)?;
        let (orig_w, orig_h) = (size.width, size.height);

        // DocAligner preprocessing: RGB, scaled to [0,1], no mean/std.
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
        // [1, 4, H, W], channels TL/TR/BR/BL, already sigmoid-activated.
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

/// Centroid of the largest blob in one corner heatmap, upscaled to image size *before*
/// thresholding (as DocAligner does) so no low-res pixel error is reintroduced.
fn largest_heatmap_blob_centroid(
    heatmap: &[f32],
    channel: usize,
    map_h: i32,
    map_w: i32,
    orig_w: i32,
    orig_h: i32,
) -> Result<Option<Point2f>, AppError> {
    let plane_len = (map_h * map_w) as usize;
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

/// Predicted edges land slightly inside the card; push corners out by this fraction of the quad.
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

/// RGB u8 -> NCHW f32 in `[0, 1]`.
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
        // All 4 heatmaps can peak while two land on the same pixel (card filling the frame),
        // collapsing the quad — validate before trusting it.
        Some(corners) if !quad_is_degenerate(&corners) => (corners, true),
        _ => (full_frame_corners(&original)?, false),
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

/// Re-warps the original against user-adjusted corners (original-image pixels, any order).
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

    if quad_is_degenerate(corners) {
        return Err(AppError::InvalidCrop("crop corners are degenerate"));
    }

    Ok(())
}

/// Collapsed, self-intersecting or too-thin quads. Needs both checks: two corners on the same
/// pixel make a triangle whose shoelace area looks fine.
const MIN_QUAD_TWICE_AREA: f32 = 4.0;
const MIN_QUAD_CORNER_SEPARATION: f32 = 2.0;

fn quad_is_degenerate(corners: &[Point2f; 4]) -> bool {
    let twice_area = corners
        .iter()
        .zip(corners.iter().cycle().skip(1))
        .take(corners.len())
        .map(|(a, b)| a.x * b.y - b.x * a.y)
        .sum::<f32>()
        .abs();
    if twice_area < MIN_QUAD_TWICE_AREA {
        return true;
    }

    for i in 0..corners.len() {
        for j in (i + 1)..corners.len() {
            if dist(corners[i], corners[j]) < f64::from(MIN_QUAD_CORNER_SEPARATION) {
                return true;
            }
        }
    }
    false
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

/// Orders as TL/TR/BR/BL: TL = min x+y, BR = max x+y, TR = max x-y, BL = min x-y.
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

/// Warps the TL/TR/BR/BL quad into an axis-aligned top-down image.
fn warp_document(image: &Mat, corners: &[Point2f]) -> Result<Mat, AppError> {
    let [tl, tr, br, bl] = [corners[0], corners[1], corners[2], corners[3]];

    // Corners are inclusive pixel centres: x=0..2159 is 2160 px wide, hence +1.
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
