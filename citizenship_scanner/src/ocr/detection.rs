//! Text-line detection (PP-OCRv6 medium): probability map -> expanded line boxes
//! ([`detect_text_boxes`]); [`crop_quad`] cuts one out. Seconds per call — the costly stage.

use std::sync::Mutex;

use opencv::core::{CV_8U, Mat, Point, Point2f, Rect, RotatedRect, Scalar, Size, Vector};
use opencv::prelude::*;
use opencv::{geometry, imgproc};
use ort::session::Session;

use super::Quad;
use super::tensor::{forward, imagenet_nchw};
use crate::error::AppError;

// PP-OCRv6 medium. Don't swap in PP-OCRv5_server_det: tried, its boxes merged neighbouring
// fields and mis-assigned values.
pub(super) const DET_MODEL_PATH: &str = "models/PP-OCRv6_medium_det.onnx";

// Same detector hyperparameters as the NID/passport services.
const DET_LIMIT_SIDE_LEN: i32 = 64;

/// Max side fed to the detector (`OCR_DET_MAX_SIDE`, default 4000, production 1600). Memory
/// scales with its square; below 1600 changes results on the test scans.
fn det_max_side_limit() -> i32 {
    std::env::var("OCR_DET_MAX_SIDE")
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(4000)
        .clamp(320, 8000)
}

const DET_THRESH: f64 = 0.3;

const DET_BOX_THRESH: f32 = 0.6;

const DET_UNCLIP_RATIO: f32 = 1.5;

const DET_MIN_BOX_SIDE: f32 = 3.0;

fn det_resize_dims(height: i32, width: i32) -> (i32, i32) {
    let mut ratio = 1.0_f64;
    if f64::from(height.min(width)) < f64::from(DET_LIMIT_SIDE_LEN) {
        ratio = f64::from(DET_LIMIT_SIDE_LEN) / f64::from(height.min(width));
    }
    let mut resize_h = (f64::from(height) * ratio) as i32;
    let mut resize_w = (f64::from(width) * ratio) as i32;
    if f64::from(resize_h.max(resize_w)) > f64::from(det_max_side_limit()) {
        let cap = f64::from(det_max_side_limit()) / f64::from(resize_h.max(resize_w));
        resize_h = (f64::from(resize_h) * cap) as i32;
        resize_w = (f64::from(resize_w) * cap) as i32;
    }
    let round32 = |value: i32| ((f64::from(value) / 32.0).round() as i32 * 32).max(32);
    (round32(resize_h), round32(resize_w))
}

pub(super) fn detect_text_boxes(det: &Mutex<Session>, image: &Mat) -> Result<Vec<Quad>, AppError> {
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

pub(super) fn crop_quad(image: &Mat, corners: &Quad) -> Result<Mat, AppError> {
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
