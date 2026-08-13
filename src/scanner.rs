use std::sync::Mutex;

use opencv::core::{Point2f, Scalar, Size, Vector};
use opencv::prelude::*;
use opencv::{dnn, geometry, imgcodecs, imgproc};

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
pub struct DocDetector(Mutex<dnn::Net>);

/// DocAligner's `lcnet050` point-regression model (Apache-2.0,
/// github.com/DocsaidLab/DocAligner): a PP-LCNet-0.5 backbone trained to
/// regress the 4 corners of a document directly, in one forward pass, with
/// no edges/contours/color-segmentation heuristics involved. Runs at a
/// fixed 256x256 input, so it costs the same few milliseconds regardless of
/// the source photo's resolution — verified at ~5ms/image on CPU, an
/// improvement of nearly 3 orders of magnitude over the classical CV
/// pipeline it replaces (which additionally could not find the document at
/// all on some real, cluttered photos; this model does).
const MODEL_PATH: &str = "models/docaligner_lcnet050.onnx";
const MODEL_INPUT_SIZE: i32 = 256;
/// Below this the model itself is saying "I don't think there's a document
/// here" — matches the threshold DocAligner's own reference code uses.
const HAS_OBJ_THRESHOLD: f32 = 0.5;

impl DocDetector {
    pub fn load() -> Result<Self, AppError> {
        let net = dnn::read_net_from_onnx_def(MODEL_PATH).map_err(AppError::Processing)?;
        Ok(Self(Mutex::new(net)))
    }

    /// Runs the model on `image` (full resolution — the fixed 256x256 input
    /// means there's no separate "detection copy" to manage) and returns
    /// its 4 corners in `image`'s own pixel space, or `None` if the model's
    /// own confidence is below [`HAS_OBJ_THRESHOLD`].
    fn detect(&self, image: &Mat) -> Result<Option<[Point2f; 4]>, AppError> {
        let size = image.size().map_err(AppError::Processing)?;
        let (orig_w, orig_h) = (size.width as f32, size.height as f32);

        let blob = dnn::blob_from_image(
            image,
            1.0 / 255.0,
            Size::new(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            Scalar::all(0.0),
            true, // swapRB: model was trained on RGB, imdecode gives BGR
            false,
            opencv::core::CV_32F,
        )?;

        let mut net = self.0.lock().expect("dnn::Net mutex poisoned");
        net.set_input(&blob, "img", 1.0, Scalar::all(0.0))?;

        let mut outputs = Vector::<Mat>::new();
        let names = Vector::<String>::from_iter(["points", "has_obj"]);
        net.forward(&mut outputs, &names)?;
        drop(net);

        let has_obj: f32 = *outputs.get(1)?.at(0)?;
        if has_obj < HAS_OBJ_THRESHOLD {
            return Ok(None);
        }

        // `points` is [1, 8]: 4 (x, y) pairs, each a fraction (0..1) of the
        // *original* image's own width/height — the model's own postprocess
        // convention, not relative to the 256x256 input it actually saw.
        let points = outputs.get(0)?;
        let mut corners = [Point2f::new(0.0, 0.0); 4];
        for i in 0..4 {
            let x: f32 = *points.at(2 * i)?;
            let y: f32 = *points.at(2 * i + 1)?;
            corners[i as usize] = Point2f::new(x * orig_w, y * orig_h);
        }
        let corners = expand_quad(order_corners(&corners), orig_w, orig_h);
        Ok(Some(corners))
    }
}

/// The model's predicted edge tends to land a hair inside the document's
/// true boundary rather than exactly on it (regression models trained on
/// human-labeled corners systematically do this — annotators tend to click
/// slightly inside a busy edge/text region), which was visibly clipping a
/// sliver of text on some scans. Pushes each corner outward from the quad's
/// own center by [`CORNER_EXPAND_FRACTION`], clamped back to the image
/// bounds, to compensate.
const CORNER_EXPAND_FRACTION: f32 = 0.025;

fn expand_quad(corners: [Point2f; 4], width: f32, height: f32) -> [Point2f; 4] {
    let cx = corners.iter().map(|p| p.x).sum::<f32>() / 4.0;
    let cy = corners.iter().map(|p| p.y).sum::<f32>() / 4.0;
    corners.map(|p| {
        let x = cx + (p.x - cx) * (1.0 + CORNER_EXPAND_FRACTION);
        let y = cy + (p.y - cy) * (1.0 + CORNER_EXPAND_FRACTION);
        Point2f::new(x.clamp(0.0, width - 1.0), y.clamp(0.0, height - 1.0))
    })
}

pub fn scan_document(detector: &DocDetector, bytes: &[u8]) -> Result<ScanOutput, AppError> {
    let buf = Vector::<u8>::from_slice(bytes);
    let original = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR).map_err(AppError::Decode)?;
    if original.empty() {
        return Err(AppError::EmptyImage);
    }
    let orig_size = original.size().map_err(AppError::Processing)?;

    let (corners_full, corners_detected) = match detector.detect(&original)? {
        Some(corners) => (corners, true),
        None => (full_frame_corners(&original)?, false),
    };

    let warped = warp_document(&original, &corners_full)?;
    let out_size = warped.size().map_err(AppError::Processing)?;

    let mut params = Vector::<i32>::new();
    params.push(imgcodecs::IMWRITE_JPEG_QUALITY);
    params.push(92);
    let mut out_buf = Vector::<u8>::new();
    imgcodecs::imencode(".jpg", &warped, &mut out_buf, &params)?;

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

    let width = dist(br, bl).max(dist(tr, tl)).round() as i32;
    let height = dist(tr, br).max(dist(tl, bl)).round() as i32;
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
