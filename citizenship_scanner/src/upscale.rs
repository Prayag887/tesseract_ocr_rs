//! Optional pre-detection upscaler for crops that came out small (see
//! `MIN_PREPROCESS_SHORT_SIDE` in `local_ocr.rs`) — gated behind
//! `CITIZENSHIP_OCR_UPSCALE` (default off, see `models/README.md` for why).
//! Not something either author has verified as a net accuracy win yet;
//! this exists so it can be tested against real scans without guessing.
//!
//! The model (`super-resolution-10.onnx`, the classic ESPCN-style
//! sub-pixel CNN from the official ONNX Model Zoo) only accepts a fixed
//! 224x224 single-channel input and produces a 672x672 output (3x) — it
//! has no notion of "upscale this arbitrary-sized image." So the crop is
//! tiled into 224x224 blocks (reflect-padded up to a multiple of 224),
//! each tile is upscaled independently, and the results are stitched back
//! into one image. Only the luma (Y) channel goes through the model — it's
//! trained on Y only, per the model's own documentation — while color
//! (Cr/Cb) is upscaled with a plain bicubic resize, cheap and sufficient
//! since human/OCR legibility lives almost entirely in luma contrast, not
//! color detail.

use std::sync::Mutex;

use opencv::core::{self, CV_8U, Mat, MatTraitConst, Rect, Scalar, Vector};
use opencv::imgproc;
use opencv::prelude::*;
use ort::session::Session;
use ort::value::Tensor;

use crate::error::AppError;

pub const UPSCALE_MODEL_PATH: &str = "models/super-resolution-10.onnx";

const TILE: i32 = 224;
const SCALE: i32 = 3;
const INPUT_NAME: &str = "input";
const OUTPUT_NAME: &str = "output";

/// Upscales `bgr` 3x via the tiled Y-channel model, falling back to a plain
/// bicubic resize for Cr/Cb. Returns a new, larger `Mat` — the caller
/// decides what to do with it (this module has no opinion on preprocessing
/// or detection, it just upscales).
pub fn upscale_3x(session: &Mutex<Session>, bgr: &Mat) -> Result<Mat, AppError> {
    let mut ycrcb = Mat::default();
    imgproc::cvt_color_def(bgr, &mut ycrcb, imgproc::COLOR_BGR2YCrCb)?;

    let mut channels = Vector::<Mat>::new();
    core::split(&ycrcb, &mut channels)?;
    let y = channels.get(0)?;
    let cr = channels.get(1)?;
    let cb = channels.get(2)?;

    let size = y.size()?;
    let (width, height) = (size.width, size.height);
    let pad_bottom = (TILE - height % TILE) % TILE;
    let pad_right = (TILE - width % TILE) % TILE;

    let mut y_padded = Mat::default();
    core::copy_make_border_def(&y, &mut y_padded, 0, pad_bottom, 0, pad_right, core::BORDER_REFLECT)?;
    let padded_size = y_padded.size()?;
    let (padded_w, padded_h) = (padded_size.width, padded_size.height);

    let mut y_up = Mat::new_rows_cols_with_default(
        padded_h * SCALE,
        padded_w * SCALE,
        CV_8U,
        Scalar::all(0.0),
    )?;

    let mut ty = 0;
    while ty < padded_h {
        let mut tx = 0;
        while tx < padded_w {
            let tile_rect = Rect::new(tx, ty, TILE, TILE);
            // `roi` is a strided view into `y_padded`, not a contiguous
            // buffer — clone it so the pixel read below (which assumes a
            // packed row-major buffer) is actually reading this tile and
            // not garbage from the next row of the source image.
            let tile = Mat::roi(&y_padded, tile_rect)?.try_clone()?;
            let tile_up = upscale_tile(session, &tile)?;

            let out_rect = Rect::new(tx * SCALE, ty * SCALE, TILE * SCALE, TILE * SCALE);
            let mut out_view = Mat::roi_mut(&mut y_up, out_rect)?;
            tile_up.copy_to(&mut out_view)?;

            tx += TILE;
        }
        ty += TILE;
    }

    // Padding was only added on the bottom/right (see copy_make_border
    // above), so the real image's upscaled content starts at (0, 0).
    let y_up_cropped = Mat::roi(&y_up, Rect::new(0, 0, width * SCALE, height * SCALE))?;

    let target = core::Size::new(width * SCALE, height * SCALE);
    let mut cr_up = Mat::default();
    imgproc::resize(&cr, &mut cr_up, target, 0.0, 0.0, imgproc::INTER_CUBIC)?;
    let mut cb_up = Mat::default();
    imgproc::resize(&cb, &mut cb_up, target, 0.0, 0.0, imgproc::INTER_CUBIC)?;

    let mut merged = Vector::<Mat>::new();
    merged.push(y_up_cropped.try_clone()?);
    merged.push(cr_up);
    merged.push(cb_up);
    let mut ycrcb_up = Mat::default();
    core::merge(&merged, &mut ycrcb_up)?;

    let mut bgr_up = Mat::default();
    imgproc::cvt_color_def(&ycrcb_up, &mut bgr_up, imgproc::COLOR_YCrCb2BGR)?;
    Ok(bgr_up)
}

/// Runs one 224x224 luma tile through the model, returning a 672x672 tile.
fn upscale_tile(session: &Mutex<Session>, tile: &Mat) -> Result<Mat, AppError> {
    let pixels = tile.data_typed::<u8>()?;
    let data: Vec<f32> = pixels.iter().map(|&p| f32::from(p) / 255.0).collect();
    let input = Tensor::from_array(([1_usize, 1, TILE as usize, TILE as usize], data))?;

    let mut session = session.lock().expect("upscale ort Session mutex poisoned");
    let outputs = session.run(ort::inputs![INPUT_NAME => input])?;
    let (_shape, out_data) = outputs[OUTPUT_NAME].try_extract_tensor::<f32>()?;

    let out_side = (TILE * SCALE) as usize;
    let out_pixels: Vec<u8> = out_data.iter().map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8).collect();
    Mat::new_rows_cols_with_data(out_side as i32, out_side as i32, &out_pixels)?.try_clone().map_err(Into::into)
}
