use std::sync::Mutex;

use opencv::core::{
    CV_8U, Mat, MatTraitConst, Point, Point2f, Rect, RotatedRect, Scalar, Size, Vec3b, Vector,
};
use opencv::prelude::*;
use opencv::{dnn, geometry, imgcodecs, imgproc};

use crate::error::AppError;
use crate::mrz::{self, MrzDocument};

const DET_MODEL_PATH: &str = "models/ppocrv5_det.onnx";
const TEXTLINE_ORI_MODEL_PATH: &str = "models/textline_ori.onnx";
/// Reuses the NID service's exact recognition model + dict (copied
/// verbatim, not shared as a dependency). Its dict is PP-OCRv5's Devanagari
/// vocab, which extends the *full* Latin base charset — every character an
/// MRZ can contain (digits, A-Z, `<`) is already in it (verified:
/// devanagari_rec_dict.txt lines 16-25 = 0-9, 32-57 = A-Z, line 28 = `<`).
/// So no separate Latin/OCR-B model is needed; only the character-set
/// filtering in `normalize_mrz_chars` and the MRZ-specific line picking
/// below differ from the NID service's usage of this same network.
const REC_MODEL_PATH: &str = "models/devanagari_rec.onnx";
const REC_DICT_PATH: &str = "models/devanagari_rec_dict.txt";

const NET_INPUT_NAME: &str = "x";
const NET_OUTPUT_NAME: &str = "fetch_name_0";

// Same PP-OCRv5_mobile_det hyperparameters as the NID service's OCR.yaml —
// the detection model is script-agnostic, only the recognizer differs.
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

/// TD3 MRZ line length; a recognized line within this margin of 44 chars is
/// a plausible MRZ row before it's padded/truncated for `mrz::parse_td3`.
const MRZ_LINE_LEN: usize = 44;

pub struct MrzOcrEngine {
    det: Mutex<dnn::Net>,
    textline_ori: Mutex<dnn::Net>,
    rec: Mutex<dnn::Net>,
    rec_vocab: Vec<String>,
    min_confidence: f32,
}

impl MrzOcrEngine {
    pub fn load() -> Result<Self, AppError> {
        let det = dnn::read_net_from_onnx_def(DET_MODEL_PATH)?;
        let textline_ori = dnn::read_net_from_onnx_def(TEXTLINE_ORI_MODEL_PATH)?;
        let rec = dnn::read_net_from_onnx_def(REC_MODEL_PATH)?;

        let dict_text = std::fs::read_to_string(REC_DICT_PATH).map_err(AppError::Io)?;
        let mut rec_vocab = Vec::new();
        rec_vocab.push(String::new()); // CTC blank
        rec_vocab.extend(dict_text.lines().map(str::to_owned));
        rec_vocab.push(" ".to_owned());

        let min_confidence = std::env::var("MRZ_OCR_MIN_CONFIDENCE")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(0.45);

        Ok(Self {
            det: Mutex::new(det),
            textline_ori: Mutex::new(textline_ori),
            rec: Mutex::new(rec),
            rec_vocab,
            min_confidence,
        })
    }

    pub async fn extract(&self, image: &[u8]) -> Result<MrzDocument, AppError> {
        let buf = Vector::<u8>::from_slice(image);
        let bgr = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR).map_err(AppError::Decode)?;
        if bgr.empty() {
            return Err(AppError::EmptyImage);
        }

        let quads = detect_text_boxes(&self.det, &bgr)?;

        let mut lines: Vec<(String, f32, i32)> = Vec::with_capacity(quads.len());
        for quad in quads {
            let crop = crop_quad(&bgr, &quad)?;
            let crop = classify_and_fix_rotation(&self.textline_ori, crop)?;
            let (text, confidence) = recognize_line(&self.rec, &self.rec_vocab, &crop)?;
            let text = normalize_mrz_chars(text.trim());
            if text.is_empty() || confidence < self.min_confidence {
                continue;
            }
            let top = quad.iter().map(|p| p.y).fold(f32::MAX, f32::min).round() as i32;
            lines.push((text, confidence, top));
        }

        // MRZ rows are the two bottommost lines that are long, dense in
        // MRZ-valid characters, and close to the fixed 44-char TD3 width —
        // no label/keyword to anchor on, unlike the NID service's field
        // parser, since the MRZ has none.
        lines.sort_by_key(|(_, _, top)| *top);
        let mrz_candidates: Vec<&(String, f32, i32)> = lines
            .iter()
            .filter(|(text, _, _)| looks_like_mrz_line(text))
            .collect();

        let [line1, line2] = mrz_candidates
            .iter()
            .rev()
            .take(2)
            .rev()
            .map(|(text, ..)| pad_or_truncate_mrz(text))
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| AppError::MrzNotFound)?;

        mrz::parse_td3(&line1, &line2)
    }
}

/// Maps common OCR-B misreads to their MRZ alphabet equivalents and drops
/// anything outside `[A-Z0-9<]` — MRZ text is fixed-alphabet by spec, so any
/// other character recognized is noise.
fn normalize_mrz_chars(text: &str) -> String {
    text.chars()
        .map(|c| c.to_ascii_uppercase())
        .map(|c| match c {
            ' ' | '_' | '«' | '‹' => '<',
            _ => c,
        })
        .filter(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '<')
        .collect()
}

fn looks_like_mrz_line(text: &str) -> bool {
    let len = text.chars().count();
    len >= MRZ_LINE_LEN.saturating_sub(6) && len <= MRZ_LINE_LEN + 6
}

fn pad_or_truncate_mrz(text: &str) -> String {
    let mut chars: Vec<char> = text.chars().collect();
    chars.resize(MRZ_LINE_LEN, '<');
    chars.truncate(MRZ_LINE_LEN);
    chars.into_iter().collect()
}

fn forward(net: &Mutex<dnn::Net>, blob: &Mat) -> Result<Mat, AppError> {
    let mut net = net.lock().expect("dnn::Net mutex poisoned");
    net.set_input(blob, NET_INPUT_NAME, 1.0, Scalar::all(0.0))?;
    let mut outputs = Vector::<Mat>::new();
    let names = Vector::<String>::from_iter([NET_OUTPUT_NAME]);
    net.forward(&mut outputs, &names)?;
    Ok(outputs.get(0)?)
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

fn rec_nchw(rgb: &Mat) -> Result<Vec<f32>, AppError> {
    let rows = rgb.rows();
    let cols = rgb.cols();
    let pixels = rgb.data_typed::<Vec3b>()?;
    let plane = (rows * cols) as usize;
    let mut out = vec![0f32; 3 * plane];
    for (index, pixel) in pixels.iter().enumerate() {
        for channel in 0..3 {
            let value = f32::from(pixel[channel]);
            out[channel * plane + index] = (value - 127.5) / 127.5;
        }
    }
    Ok(out)
}

fn nchw_blob(sizes: &[i32], data: &[f32]) -> Result<Mat, AppError> {
    Ok(Mat::new_nd_with_data(sizes, data)?.try_clone()?)
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

fn detect_text_boxes(det: &Mutex<dnn::Net>, image: &Mat) -> Result<Vec<[Point2f; 4]>, AppError> {
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
    let blob = nchw_blob(&[1, 3, resize_h, resize_w], &data)?;
    let prob_map = forward(det, &blob)?;
    let prob_2d: Mat = prob_map.reshape(1, resize_h)?.try_clone()?;

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
        .map(|p| Point::new((p.x - x0 as f32).round() as i32, (p.y - y0 as f32).round() as i32))
        .collect();
    let mut contour_group = Vector::<Vector<Point>>::new();
    contour_group.push(local_points);

    let mut mask = Mat::new_rows_cols_with_default(
        y1 - y0,
        x1 - x0,
        CV_8U,
        Scalar::all(0.0),
    )?;
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

fn classify_and_fix_rotation(textline_ori: &Mutex<dnn::Net>, crop: Mat) -> Result<Mat, AppError> {
    let size = crop.size()?;
    if size.width < 1 || size.height < 1 {
        return Ok(crop);
    }

    let mut resized = Mat::default();
    imgproc::resize(
        &crop,
        &mut resized,
        Size::new(TEXTLINE_ORI_WIDTH, TEXTLINE_ORI_HEIGHT),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    let mut rgb = Mat::default();
    imgproc::cvt_color_def(&resized, &mut rgb, imgproc::COLOR_BGR2RGB)?;
    let data = imagenet_nchw(&rgb)?;
    let blob = nchw_blob(&[1, 3, TEXTLINE_ORI_HEIGHT, TEXTLINE_ORI_WIDTH], &data)?;
    let out = forward(textline_ori, &blob)?;

    let upright: f32 = *out.at_2d(0, 0)?;
    let flipped: f32 = *out.at_2d(0, 1)?;
    if flipped > upright {
        let mut rotated = Mat::default();
        opencv::core::flip(&crop, &mut rotated, -1)?;
        Ok(rotated)
    } else {
        Ok(crop)
    }
}

fn recognize_line(
    rec: &Mutex<dnn::Net>,
    vocab: &[String],
    crop: &Mat,
) -> Result<(String, f32), AppError> {
    let size = crop.size()?;
    if size.width < 1 || size.height < 1 {
        return Ok((String::new(), 0.0));
    }
    let new_w = ((f64::from(size.width) * f64::from(REC_HEIGHT) / f64::from(size.height)).round()
        as i32)
        .clamp(1, REC_MAX_WIDTH);

    let mut resized = Mat::default();
    imgproc::resize(
        crop,
        &mut resized,
        Size::new(new_w, REC_HEIGHT),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    let mut rgb = Mat::default();
    imgproc::cvt_color_def(&resized, &mut rgb, imgproc::COLOR_BGR2RGB)?;
    let data = rec_nchw(&rgb)?;
    let blob = nchw_blob(&[1, 3, REC_HEIGHT, new_w], &data)?;
    let out = forward(rec, &blob)?;

    let dims = out.mat_size();
    let timesteps = dims[1];
    let vocab_len = dims[2];

    let mut text = String::new();
    let mut confidences = Vec::new();
    let mut previous_index = -1_i32;
    for t in 0..timesteps {
        let mut best_index = 0_i32;
        let mut best_value = f32::MIN;
        for v in 0..vocab_len {
            let value: f32 = *out.at_3d(0, t, v)?;
            if value > best_value {
                best_value = value;
                best_index = v;
            }
        }
        if best_index != 0
            && best_index != previous_index
            && let Some(character) = vocab.get(best_index as usize)
        {
            text.push_str(character);
            confidences.push(best_value);
        }
        previous_index = best_index;
    }

    let confidence = if confidences.is_empty() {
        0.0
    } else {
        confidences.iter().sum::<f32>() / confidences.len() as f32
    };
    Ok((text, confidence))
}
