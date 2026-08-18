use std::sync::Mutex;

use opencv::core::{
    CV_8U, Mat, MatTraitConst, Point, Point2f, Rect, RotatedRect, Scalar, Size, Vec3b, Vector,
};
use opencv::prelude::*;
use opencv::{geometry, imgcodecs, imgproc};
use ort::session::Session;
use ort::value::Tensor;

use crate::error::AppError;
use crate::fields::{self, CitizenshipDocument, OcrLine};
use crate::preprocess;

// Detection is script-agnostic (unlike recognition, which has no larger
// Devanagari model available — see fields.rs/local_ocr.rs history), so it's
// the one component here worth trying alternatives for. Two were tried:
//
// - PP-OCRv5_server_det (same `x` -> `fetch_name_0` tensor names, drop-in):
//   just a bigger version of the same v5 architecture. Made real-scan
//   results *worse* — its unclip/merge behavior pulled noticeably more
//   distant text into single boxes on this service's low-resolution real
//   test image, which didn't just fail to find more content, it actively
//   cross-contaminated fields (a box spanning "आमाको नाम थर: फुलमाया
//   अधिकारी : ना.प्र.नं: ..." got the mother's name attributed to
//   `full_name` via a "नाम थर" substring match). Reverted. (See
//   `value_after_keyword`'s stop-keyword hardening in fields.rs, added
//   because of this finding — a box that spans further than expected
//   should no longer misattribute the extra content regardless of which
//   detector produced it.)
// - PP-OCRv6_medium_det (currently active): a genuinely different
//   architecture (LCNetV4 + RepLKFPN vs v5's backbone/neck), not just a
//   bigger v5 — reports +4.6 detection Hmean over PP-OCRv5_server in
//   PaddleOCR's own benchmarks. Confirmed as a real improvement here too,
//   not just a bigger-model gamble: the back page's citizenship number
//   went from a one-digit misread (65-01-77-02873) to byte-exact
//   (65-01-77-02872) on the same real scan, with every other previously-
//   correct field holding steady. Kept.
const DET_MODEL_PATH: &str = "models/PP-OCRv6_medium_det.onnx";
const TEXTLINE_ORI_MODEL_PATH: &str = "models/textline_ori.onnx";
/// PP-OCRv5's Devanagari recognizer + dict, shared verbatim with the
/// NID/passport services — it extends the full Latin base charset too, so
/// one model reads both scripts printed on this bilingual certificate.
const REC_MODEL_PATH: &str = "models/devanagari_rec.onnx";
const REC_DICT: &str = include_str!("../models/devanagari_rec_dict.txt");

const NET_INPUT_NAME: &str = "x";
const NET_OUTPUT_NAME: &str = "fetch_name_0";

// Same PP-OCRv5_mobile_det hyperparameters the other two services use — the
// detection model is script-agnostic.
const DET_LIMIT_SIDE_LEN: i32 = 64;
const DET_MAX_SIDE_LIMIT: i32 = 4000;
const DET_THRESH: f64 = 0.3;
const DET_BOX_THRESH: f32 = 0.6;
const DET_UNCLIP_RATIO: f32 = 1.5;
const DET_MIN_BOX_SIDE: f32 = 3.0;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

const TEXTLINE_ORI_WIDTH: i32 = 160;
const TEXTLINE_ORI_HEIGHT: i32 = 80;

const REC_HEIGHT: i32 = 48;
const REC_MAX_WIDTH: i32 = 2000;
const DEFAULT_ORIENTATION_BATCH_SIZE: usize = 8;
const DEFAULT_RECOGNITION_BATCH_SIZE: usize = 1;
const MAX_BATCH_SIZE: usize = 64;
const MAX_RECOGNITION_WORKERS: usize = 8;

pub struct LocalOcrEngine {
    det: Mutex<Session>,
    textline_ori: Mutex<Session>,
    rec: Vec<Mutex<Session>>,
    rec_vocab: Vec<&'static str>,
    min_confidence: f32,
    orientation_batch_size: usize,
    recognition_batch_size: usize,
    /// Off switch for the watermark/lighting cleanup pass in `preprocess.rs`
    /// — kept configurable in case a particular scan reads better without
    /// it (e.g. an already-clean, non-watermarked photocopy).
    preprocess_enabled: bool,
    /// Where the OCR-input debug image gets written (see `extract_blocking`)
    /// — not the crop/original, which callers don't need kept around at all.
    output_dir: std::path::PathBuf,
}

impl LocalOcrEngine {
    pub fn load(output_dir: std::path::PathBuf) -> Result<Self, AppError> {
        let det = Session::builder()?.commit_from_file(DET_MODEL_PATH)?;
        let textline_ori = Session::builder()?.commit_from_file(TEXTLINE_ORI_MODEL_PATH)?;
        let mut rec_vocab = Vec::with_capacity(570);
        rec_vocab.push(""); // CTC blank
        rec_vocab.extend(REC_DICT.lines());
        rec_vocab.push(" ");

        let min_confidence = std::env::var("CITIZENSHIP_OCR_MIN_CONFIDENCE")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(0.45);
        let orientation_batch_size =
            env_usize("OCR_ORIENTATION_BATCH_SIZE", DEFAULT_ORIENTATION_BATCH_SIZE)
                .clamp(1, MAX_BATCH_SIZE);
        let recognition_batch_size =
            env_usize("OCR_RECOGNITION_BATCH_SIZE", DEFAULT_RECOGNITION_BATCH_SIZE)
                .clamp(1, MAX_BATCH_SIZE);
        let recognition_workers =
            env_usize("OCR_RECOGNITION_WORKERS", default_recognition_workers())
                .clamp(1, MAX_RECOGNITION_WORKERS);
        let mut rec = Vec::with_capacity(recognition_workers);
        for _ in 0..recognition_workers {
            rec.push(Mutex::new(
                Session::builder()?.commit_from_file(REC_MODEL_PATH)?,
            ));
        }
        tracing::info!(recognition_workers, "OCR recognition workers configured");

        let preprocess_enabled = std::env::var("CITIZENSHIP_OCR_PREPROCESS")
            .ok()
            .and_then(|value| value.parse::<bool>().ok())
            .unwrap_or(true);

        Ok(Self {
            det: Mutex::new(det),
            textline_ori: Mutex::new(textline_ori),
            rec,
            rec_vocab,
            min_confidence,
            orientation_batch_size,
            recognition_batch_size,
            preprocess_enabled,
            output_dir,
        })
    }

    /// Dispatches the actually-synchronous, CPU-bound work onto the
    /// blocking pool so it can't stall the async reactor — same reasoning
    /// as `scanner::scan_document`. `side` (expected `"front"` or `"back"`,
    /// anything else falls back to a generic name) only controls the debug
    /// preprocessed-image filename below; extraction logic is identical
    /// either way.
    pub async fn extract(&self, image: &[u8], side: Option<&str>) -> Result<CitizenshipDocument, AppError> {
        tokio::task::block_in_place(|| self.extract_blocking(image, side))
    }

    fn extract_blocking(&self, image: &[u8], side: Option<&str>) -> Result<CitizenshipDocument, AppError> {
        let debug = std::env::var("OCR_DEBUG_TIMING").is_ok();
        let t0 = std::time::Instant::now();
        let buf = Vector::<u8>::from_slice(image);
        let bgr = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR).map_err(AppError::Decode)?;
        if bgr.empty() {
            return Err(AppError::EmptyImage);
        }

        // Citizenship certificates are printed on watermarked security
        // paper and are usually photographed under uneven light, not
        // flatbed-scanned — clean that up before text detection runs, since
        // the detector otherwise picks up the watermark's edges as noise
        // and the recognizer loses contrast against it.
        let cleaned = if self.preprocess_enabled {
            let t_pre = std::time::Instant::now();
            let cleaned = preprocess::denoise_for_ocr(&bgr)?;
            if debug {
                eprintln!("[ocr_timing] preprocess: {:?}", t_pre.elapsed());
            }
            cleaned
        } else {
            bgr
        };

        // The crop/original are deleted from disk right after this request
        // (see routes::extract) — nothing about a scanned citizenship
        // certificate needs to linger. This debug image is the one
        // exception: it's what the OCR models actually saw, which is the
        // single most useful thing to look at when a field comes back
        // wrong, so it's kept — but at a fixed filename per side, always
        // overwritten by the next scan of that side, never accumulating.
        let side_name = match side {
            Some("front") => "front",
            Some("back") => "back",
            _ => "last",
        };
        let debug_path = self.output_dir.join(format!("_debug_preprocessed_{side_name}.jpg"));
        let mut params = Vector::<i32>::new();
        params.push(imgcodecs::IMWRITE_JPEG_QUALITY);
        params.push(90);
        let mut debug_bytes = Vector::<u8>::new();
        if imgcodecs::imencode(".jpg", &cleaned, &mut debug_bytes, &params).is_ok() {
            let _ = std::fs::write(&debug_path, debug_bytes.as_slice());
        }

        let t1 = std::time::Instant::now();
        let lines = self.run_detection_pipeline(&self.det, &cleaned, debug, "primary")?;
        if debug {
            eprintln!("[ocr_timing] detection pipeline total: {:?}", t1.elapsed());
        }

        // Same fixed-filename-per-side convention as the preprocessed-image
        // dump above: draws every surviving box (green, numbered) on top of
        // what the detector saw, plus each box's recognition confidence —
        // the direct way to tell "detector found the right region but
        // recognizer misread it" apart from "detector never found this text
        // at all", which reading recognized strings alone can't distinguish.
        if let Ok(annotated) = draw_debug_bboxes(&cleaned, &lines) {
            let bbox_debug_path = self.output_dir.join(format!("_debug_bboxes_{side_name}.jpg"));
            let mut bbox_bytes = Vector::<u8>::new();
            if imgcodecs::imencode(".jpg", &annotated, &mut bbox_bytes, &params).is_ok() {
                let _ = std::fs::write(&bbox_debug_path, bbox_bytes.as_slice());
            }
        }

        if std::env::var("OCR_DEBUG_LINES").is_ok() {
            for (i, line) in lines.iter().enumerate() {
                tracing::info!("line {i}: {:?} conf={:.3}", line.text, line.confidence);
            }
        }

        let document = fields::extract(&lines);
        if debug {
            eprintln!("[ocr_timing] total: {:?}", t0.elapsed());
        }
        Ok(document)
    }

    /// The full detect -> crop -> orient -> recognize pass, filtered to the
    /// lines that clear `min_confidence`.
    fn run_detection_pipeline(
        &self,
        det: &Mutex<Session>,
        cleaned: &Mat,
        debug: bool,
        label: &str,
    ) -> Result<Vec<OcrLine>, AppError> {
        let t1 = std::time::Instant::now();
        let quads = detect_text_boxes(det, cleaned)?;
        if debug {
            eprintln!("[ocr_timing] {label} detect_text_boxes: {:?} ({} boxes)", t1.elapsed(), quads.len());
        }

        let t2 = std::time::Instant::now();
        let mut crops = Vec::with_capacity(quads.len());
        for quad in &quads {
            crops.push(crop_quad(cleaned, quad)?);
        }
        let crops = classify_and_fix_rotation_batch(&self.textline_ori, crops, self.orientation_batch_size)?;
        if debug {
            eprintln!("[ocr_timing] {label} crop + batched orientation: {:?}", t2.elapsed());
        }

        let t3 = std::time::Instant::now();
        let recognized = recognize_lines_parallel(&self.rec, &self.rec_vocab, &crops, self.recognition_batch_size)?;
        if debug {
            eprintln!("[ocr_timing] {label} batched recognition: {:?}", t3.elapsed());
        }

        let mut lines = Vec::with_capacity(quads.len());
        for (quad, (text, confidence)) in quads.into_iter().zip(recognized) {
            let text = text.trim();
            if text.is_empty() || confidence < self.min_confidence {
                continue;
            }
            lines.push(OcrLine {
                text: text.to_owned(),
                confidence,
                polygon: quad.map(|p| [p.x.round() as i32, p.y.round() as i32]),
            });
        }
        Ok(lines)
    }
}

/// Draws every detected text box (green outline) plus its index and
/// recognition confidence over a copy of `image` — the box's own
/// recognized text isn't rendered on top of it, since OpenCV's built-in
/// font has no Devanagari glyphs and would just draw garbage; cross-
/// reference the index against `OCR_DEBUG_LINES=1`'s log output instead.
fn draw_debug_bboxes(image: &Mat, lines: &[OcrLine]) -> Result<Mat, AppError> {
    let mut annotated = image.try_clone()?;
    let box_color = Scalar::new(0.0, 220.0, 0.0, 0.0);
    let label_color = Scalar::new(0.0, 0.0, 255.0, 0.0);
    for (index, line) in lines.iter().enumerate() {
        let points: Vector<Point> =
            line.polygon.iter().map(|p| Point::new(p[0], p[1])).collect();
        let mut polygon = Vector::<Vector<Point>>::new();
        polygon.push(points);
        imgproc::polylines(&mut annotated, &polygon, true, box_color, 2, imgproc::LINE_8, 0)?;

        let label = format!("{index}:{:.2}", line.confidence);
        let origin = Point::new(line.polygon[0][0], (line.polygon[0][1] - 4).max(10));
        imgproc::put_text_def(
            &mut annotated,
            &label,
            origin,
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.4,
            label_color,
        )?;
    }
    Ok(annotated)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn default_recognition_workers() -> usize {
    let logical_cpus = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let request_concurrency = env_usize("OCR_MAX_CONCURRENCY", 2).clamp(1, 64);
    logical_cpus
        .div_ceil(request_concurrency)
        .clamp(1, MAX_RECOGNITION_WORKERS)
}

fn forward(
    net: &Mutex<Session>,
    shape: [usize; 4],
    data: Vec<f32>,
) -> Result<(Vec<i64>, Vec<f32>), AppError> {
    let input = Tensor::from_array((shape, data))?;
    let mut net = net.lock().expect("ort Session mutex poisoned");
    let outputs = net.run(ort::inputs![NET_INPUT_NAME => input])?;
    let (out_shape, out_data) = outputs[NET_OUTPUT_NAME].try_extract_tensor::<f32>()?;
    Ok((out_shape.to_vec(), out_data.to_vec()))
}

fn imagenet_nchw(rgb: &Mat) -> Result<Vec<f32>, AppError> {
    let rows = rgb.rows();
    let cols = rgb.cols();
    let pixels = rgb.data_typed::<Vec3b>()?;
    let plane = (rows * cols) as usize;
    let mut out = vec![0f32; 3 * plane];
    for (index, pixel) in pixels.iter().enumerate() {
        for channel in 0..3 {
            let value = f32::from(pixel[channel]) / 255.0;
            out[channel * plane + index] = (value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
        }
    }
    Ok(out)
}

fn det_resize_dims(height: i32, width: i32) -> (i32, i32) {
    let mut ratio = 1.0_f64;
    if f64::from(height.min(width)) < f64::from(DET_LIMIT_SIDE_LEN) {
        ratio = f64::from(DET_LIMIT_SIDE_LEN) / f64::from(height.min(width));
    }
    let mut resize_h = (f64::from(height) * ratio) as i32;
    let mut resize_w = (f64::from(width) * ratio) as i32;
    if f64::from(resize_h.max(resize_w)) > f64::from(DET_MAX_SIDE_LIMIT) {
        let cap = f64::from(DET_MAX_SIDE_LIMIT) / f64::from(resize_h.max(resize_w));
        resize_h = (f64::from(resize_h) * cap) as i32;
        resize_w = (f64::from(resize_w) * cap) as i32;
    }
    let round32 = |value: i32| ((f64::from(value) / 32.0).round() as i32 * 32).max(32);
    (round32(resize_h), round32(resize_w))
}

fn detect_text_boxes(det: &Mutex<Session>, image: &Mat) -> Result<Vec<[Point2f; 4]>, AppError> {
    let size = image.size()?;
    let (orig_w, orig_h) = (size.width, size.height);
    let (resize_h, resize_w) = det_resize_dims(orig_h, orig_w);

    let mut resized_bgr = Mat::default();
    imgproc::resize(
        image,
        &mut resized_bgr,
        Size::new(resize_w, resize_h),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    let mut resized_rgb = Mat::default();
    imgproc::cvt_color_def(&resized_bgr, &mut resized_rgb, imgproc::COLOR_BGR2RGB)?;

    let data = imagenet_nchw(&resized_rgb)?;
    let (_shape, prob_data) = forward(det, [1, 3, resize_h as usize, resize_w as usize], data)?;
    let prob_2d: Mat = Mat::new_rows_cols_with_data(resize_h, resize_w, &prob_data)?.try_clone()?;

    let mut binary = Mat::default();
    imgproc::threshold(
        &prob_2d,
        &mut binary,
        DET_THRESH,
        255.0,
        imgproc::THRESH_BINARY,
    )?;
    let mut binary_u8 = Mat::default();
    binary.convert_to(&mut binary_u8, CV_8U, 1.0, 0.0)?;

    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours_def(
        &binary_u8,
        &mut contours,
        imgproc::RETR_LIST,
        imgproc::CHAIN_APPROX_SIMPLE,
    )?;

    let ratio_w = resize_w as f32 / orig_w as f32;
    let ratio_h = resize_h as f32 / orig_h as f32;

    let mut boxes = Vec::new();
    for contour in &contours {
        if contour.len() < 3 {
            continue;
        }
        let rect = geometry::min_area_rect(&contour)?;
        if rect.size.width.min(rect.size.height) < DET_MIN_BOX_SIDE {
            continue;
        }

        let score = box_score(&prob_2d, rect, resize_w, resize_h)?;
        if score < DET_BOX_THRESH {
            continue;
        }

        let expanded = unclip(rect, DET_UNCLIP_RATIO);
        if expanded.size.width.min(expanded.size.height) < DET_MIN_BOX_SIDE {
            continue;
        }

        let mut points = [Point2f::default(); 4];
        expanded.points(&mut points)?;
        let quad = order_corners(points).map(|p| {
            Point2f::new(
                (p.x / ratio_w).clamp(0.0, orig_w as f32 - 1.0),
                (p.y / ratio_h).clamp(0.0, orig_h as f32 - 1.0),
            )
        });
        boxes.push(quad);
    }

    boxes.sort_unstable_by(|a, b| {
        let a_top = a.iter().map(|p| p.y).fold(f32::MAX, f32::min);
        let b_top = b.iter().map(|p| p.y).fold(f32::MAX, f32::min);
        let a_left = a.iter().map(|p| p.x).fold(f32::MAX, f32::min);
        let b_left = b.iter().map(|p| p.x).fold(f32::MAX, f32::min);
        a_top
            .total_cmp(&b_top)
            .then_with(|| a_left.total_cmp(&b_left))
    });

    Ok(boxes)
}

fn box_score(prob_2d: &Mat, rect: RotatedRect, map_w: i32, map_h: i32) -> Result<f32, AppError> {
    let bounding = rect.bounding_rect()?;
    let x0 = bounding.x.max(0);
    let y0 = bounding.y.max(0);
    let x1 = (bounding.x + bounding.width).min(map_w);
    let y1 = (bounding.y + bounding.height).min(map_h);
    if x1 <= x0 || y1 <= y0 {
        return Ok(0.0);
    }
    let roi = prob_2d.roi(Rect::new(x0, y0, x1 - x0, y1 - y0))?;

    let mut quad_points = [Point2f::default(); 4];
    rect.points(&mut quad_points)?;
    let local_points: Vector<Point> = quad_points
        .iter()
        .map(|p| {
            Point::new(
                (p.x - x0 as f32).round() as i32,
                (p.y - y0 as f32).round() as i32,
            )
        })
        .collect();
    let mut contour_group = Vector::<Vector<Point>>::new();
    contour_group.push(local_points);

    let mut mask = Mat::new_rows_cols_with_default(y1 - y0, x1 - x0, CV_8U, Scalar::all(0.0))?;
    imgproc::fill_poly_def(&mut mask, &contour_group, Scalar::all(1.0))?;

    let mean = opencv::core::mean(&roi, &mask)?;
    Ok(mean[0] as f32)
}

fn unclip(rect: RotatedRect, ratio: f32) -> RotatedRect {
    let (w, h) = (rect.size.width, rect.size.height);
    let perimeter = 2.0 * (w + h);
    let distance = if perimeter > 0.0 {
        w * h * ratio / perimeter
    } else {
        0.0
    };
    RotatedRect {
        center: rect.center,
        size: opencv::core::Size2f::new(w + 2.0 * distance, h + 2.0 * distance),
        angle: rect.angle,
    }
}

fn order_corners(points: [Point2f; 4]) -> [Point2f; 4] {
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
    (f64::from(a.x - b.x).powi(2) + f64::from(a.y - b.y).powi(2)).sqrt()
}

fn crop_quad(image: &Mat, corners: &[Point2f; 4]) -> Result<Mat, AppError> {
    let [tl, tr, br, bl] = *corners;
    let width = (dist(tl, tr).max(dist(bl, br)).round() as i32).max(1);
    let height = (dist(tl, bl).max(dist(tr, br)).round() as i32).max(1);

    let dst = [
        Point2f::new(0.0, 0.0),
        Point2f::new((width - 1) as f32, 0.0),
        Point2f::new((width - 1) as f32, (height - 1) as f32),
        Point2f::new(0.0, (height - 1) as f32),
    ];
    let transform = geometry::get_perspective_transform_slice_def([tl, tr, br, bl], dst)?;

    let mut warped = Mat::default();
    imgproc::warp_perspective_def(image, &mut warped, &transform, Size::new(width, height))?;
    Ok(warped)
}

fn classify_and_fix_rotation_batch(
    textline_ori: &Mutex<Session>,
    mut crops: Vec<Mat>,
    batch_size: usize,
) -> Result<Vec<Mat>, AppError> {
    for chunk in crops.chunks_mut(batch_size) {
        let sample_len = (3 * TEXTLINE_ORI_HEIGHT * TEXTLINE_ORI_WIDTH) as usize;
        let mut data = Vec::with_capacity(chunk.len() * sample_len);
        for crop in chunk.iter() {
            let mut resized = Mat::default();
            imgproc::resize(
                crop,
                &mut resized,
                Size::new(TEXTLINE_ORI_WIDTH, TEXTLINE_ORI_HEIGHT),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;
            let mut rgb = Mat::default();
            imgproc::cvt_color_def(&resized, &mut rgb, imgproc::COLOR_BGR2RGB)?;
            data.extend(imagenet_nchw(&rgb)?);
        }
        let (_shape, out) = forward(
            textline_ori,
            [
                chunk.len(),
                3,
                TEXTLINE_ORI_HEIGHT as usize,
                TEXTLINE_ORI_WIDTH as usize,
            ],
            data,
        )?;

        for (index, crop) in chunk.iter_mut().enumerate() {
            let upright = out[index * 2];
            let flipped = out[index * 2 + 1];
            if flipped > upright {
                let mut rotated = Mat::default();
                opencv::core::flip(crop, &mut rotated, -1)?;
                *crop = rotated;
            }
        }
    }
    Ok(crops)
}

fn recognize_lines_parallel(
    recognizers: &[Mutex<Session>],
    vocab: &[&str],
    crops: &[Mat],
    batch_size: usize,
) -> Result<Vec<(String, f32)>, AppError> {
    if recognizers.len() == 1 || crops.len() <= 1 {
        return recognize_lines_batch(&recognizers[0], vocab, crops, batch_size);
    }

    let worker_count = recognizers.len().min(crops.len());
    let chunk_size = crops.len().div_ceil(worker_count);
    let mut recognized = Vec::with_capacity(crops.len());
    std::thread::scope(|scope| -> Result<(), AppError> {
        let mut handles = Vec::with_capacity(worker_count);
        for (recognizer, chunk) in recognizers.iter().zip(crops.chunks(chunk_size)) {
            handles.push(
                scope.spawn(move || recognize_lines_batch(recognizer, vocab, chunk, batch_size)),
            );
        }
        for handle in handles {
            recognized.extend(handle.join().map_err(|_| AppError::OcrWorkerPanicked)??);
        }
        Ok(())
    })?;
    Ok(recognized)
}

fn recognize_lines_batch(
    rec: &Mutex<Session>,
    vocab: &[&str],
    crops: &[Mat],
    batch_size: usize,
) -> Result<Vec<(String, f32)>, AppError> {
    let mut widths = Vec::with_capacity(crops.len());
    for (index, crop) in crops.iter().enumerate() {
        let size = crop.size()?;
        let width = ((f64::from(size.width) * f64::from(REC_HEIGHT)
            / f64::from(size.height.max(1)))
        .round() as i32)
            .clamp(1, REC_MAX_WIDTH);
        widths.push((index, width));
    }
    widths.sort_unstable_by_key(|&(_, width)| width);
    let mut recognized = vec![(String::new(), 0.0); crops.len()];
    let mut start = 0;
    while start < widths.len() {
        let mut end = (start + batch_size).min(widths.len());
        while end > start + 1 && widths[end - 1].1 > widths[start].1.saturating_mul(2) {
            end -= 1;
        }
        let group = &widths[start..end];
        let padded_width = group.last().expect("non-empty recognition batch").1;
        let plane = (REC_HEIGHT * padded_width) as usize;
        let mut data = vec![0.0_f32; group.len() * 3 * plane];
        for (batch_index, &(crop_index, width)) in group.iter().enumerate() {
            let mut resized = Mat::default();
            imgproc::resize(
                &crops[crop_index],
                &mut resized,
                Size::new(width, REC_HEIGHT),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;
            let mut rgb = Mat::default();
            imgproc::cvt_color_def(&resized, &mut rgb, imgproc::COLOR_BGR2RGB)?;
            let pixels = rgb.data_typed::<Vec3b>()?;
            let batch_base = batch_index * 3 * plane;
            for (pixel_index, pixel) in pixels.iter().enumerate() {
                let y = pixel_index / width as usize;
                let x = pixel_index % width as usize;
                for channel in 0..3 {
                    data[batch_base + channel * plane + y * padded_width as usize + x] =
                        (f32::from(pixel[channel]) - 127.5) / 127.5;
                }
            }
        }
        let (shape, out) = forward(
            rec,
            [group.len(), 3, REC_HEIGHT as usize, padded_width as usize],
            data,
        )?;
        for (batch_index, &(crop_index, width)) in group.iter().enumerate() {
            recognized[crop_index] =
                decode_recognition(&out, &shape, batch_index, width, padded_width, vocab)?;
        }
        start = end;
    }
    Ok(recognized)
}

fn decode_recognition(
    out: &[f32],
    shape: &[i64],
    batch_index: usize,
    valid_width: i32,
    padded_width: i32,
    vocab: &[&str],
) -> Result<(String, f32), AppError> {
    let full_timesteps = shape[1] as usize;
    let vocab_len = shape[2] as usize;
    let timesteps = (((full_timesteps as i32) * valid_width + padded_width - 1) / padded_width)
        .clamp(1, full_timesteps as i32) as usize;
    let batch_offset = batch_index * full_timesteps * vocab_len;

    let mut text = String::new();
    let mut confidence_sum = 0.0_f32;
    let mut confidence_count = 0_usize;
    let mut previous_index = -1_i32;
    for t in 0..timesteps {
        let row = &out[batch_offset + t * vocab_len..batch_offset + (t + 1) * vocab_len];
        let mut best_index = 0_i32;
        let mut best_value = f32::MIN;
        for (v, &value) in row.iter().enumerate() {
            if value > best_value {
                best_value = value;
                best_index = v as i32;
            }
        }
        if best_index != 0
            && best_index != previous_index
            && let Some(character) = vocab.get(best_index as usize)
        {
            text.push_str(character);
            confidence_sum += best_value;
            confidence_count += 1;
        }
        previous_index = best_index;
    }

    let confidence = if confidence_count == 0 {
        0.0
    } else {
        confidence_sum / confidence_count as f32
    };
    Ok((text, confidence))
}
