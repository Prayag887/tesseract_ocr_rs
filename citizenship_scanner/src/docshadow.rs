//! Optional shadow-removal pass, run before upscaling (see `upscale.rs`) —
//! gated behind `CITIZENSHIP_OCR_DOCSHADOW`.
//!
//! The model (`docshadow_sd7k.onnx`) is an ONNX export of DocShadow
//! (Li et al., ICCV 2023), trained on SD7K — the largest published
//! document-shadow dataset. Downloaded from
//! `fabio-sim/DocShadow-ONNX-TensorRT`'s v1.0.0 release (MIT license).
//!
//! Inference runs at reduced resolution and the result is applied back to
//! the full-size crop as a per-pixel gain map, rather than feeding the
//! model the whole crop directly. Two reasons, one of them a hard failure:
//!
//! 1. Memory. The graph accepts a dynamic input size, so a full-resolution
//!    pass *runs* — it just allocates enormously doing it. A 2048x1405
//!    crop drove the container to ~6 GiB and got it SIGKILLed (exit 137)
//!    mid-request, surfacing as a 502 from the gateway; smaller crops
//!    survived, which is what made it look intermittent. The model's own
//!    reference runner resizes to 512x512 before inference, so full-res
//!    was never the intended usage.
//! 2. Detail. Resizing the model's *output* back up would soften every
//!    stroke on the page, which is the opposite of what the OCR stage
//!    needs. What shadow removal actually produces is a smooth,
//!    low-frequency brightness correction — so computing that correction
//!    small and multiplying it into the original pixels keeps text at
//!    native sharpness while still removing the shadow.

use opencv::core::{self, Mat, MatTraitConst, Size, Vec3b, Vec3f};
use opencv::imgproc;
use opencv::prelude::*;
use ort::session::Session;
use ort::value::Tensor;
use std::sync::Mutex;

use crate::error::AppError;

pub const DOCSHADOW_MODEL_PATH: &str = "models/docshadow_sd7k.onnx";

const INPUT_NAME: &str = "image";
const OUTPUT_NAME: &str = "result";

/// Longest side, in pixels, of the image actually fed to the model. Shadow
/// and illumination structure is smooth enough to survive this — and the
/// upstream reference implementation runs at 512, well below it.
const MAX_INFERENCE_SIDE: i32 = 768;

/// Floor for the divisor when deriving the gain map. Pixels that are
/// near-black in the input carry no usable illumination signal, and
/// dividing by them turns sensor noise into enormous gains; clamping here
/// leaves those pixels essentially untouched instead.
const MIN_GAIN_DIVISOR: f32 = 0.05;

/// Caps how far a single pixel may be brightened or darkened. Without it,
/// a small patch the model decides to blow out can push a stroke to pure
/// white and erase text the recognizer would otherwise have read.
const MAX_GAIN: f32 = 3.0;

/// Removes shadow/uneven illumination from `bgr`, returning an image of
/// the same size. See the module docs for why this runs the model at
/// reduced resolution and re-applies the result as a gain map.
pub fn remove_shadow(session: &Mutex<Session>, bgr: &Mat) -> Result<Mat, AppError> {
    let size = bgr.size()?;
    let longest = size.width.max(size.height);

    let small = if longest > MAX_INFERENCE_SIDE {
        let scale = f64::from(MAX_INFERENCE_SIDE) / f64::from(longest);
        let target = Size::new(
            ((f64::from(size.width) * scale).round() as i32).max(1),
            ((f64::from(size.height) * scale).round() as i32).max(1),
        );
        let mut resized = Mat::default();
        imgproc::resize(bgr, &mut resized, target, 0.0, 0.0, imgproc::INTER_AREA)?;
        resized
    } else {
        bgr.try_clone()?
    };

    let cleaned_small = infer(session, &small)?;

    // Already at inference size — the gain-map round trip would only add
    // rounding error, so hand back the model's own output.
    if small.size()? == size {
        return Ok(cleaned_small);
    }

    // gain = cleaned / original, per channel, computed small.
    let small_size = small.size()?;
    let source = small.data_typed::<Vec3b>()?;
    let cleaned = cleaned_small.data_typed::<Vec3b>()?;
    let mut gain_data = vec![Vec3f::default(); source.len()];
    for (index, gain) in gain_data.iter_mut().enumerate() {
        for channel in 0..3 {
            let before = (f32::from(source[index][channel]) / 255.0).max(MIN_GAIN_DIVISOR);
            let after = f32::from(cleaned[index][channel]) / 255.0;
            gain[channel] = (after / before).clamp(0.0, MAX_GAIN);
        }
    }
    let gain_small = Mat::new_rows_cols_with_data(small_size.height, small_size.width, &gain_data)?;

    // Bilinear is the right interpolation here precisely because the gain
    // map is smooth: no ringing to reintroduce, unlike on image content.
    let mut gain_full = Mat::default();
    imgproc::resize(&gain_small, &mut gain_full, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;

    let original = bgr.data_typed::<Vec3b>()?;
    let gains = gain_full.data_typed::<Vec3f>()?;
    let mut out = vec![Vec3b::default(); original.len()];
    for (index, pixel) in out.iter_mut().enumerate() {
        for channel in 0..3 {
            let value = f32::from(original[index][channel]) * gains[index][channel];
            pixel[channel] = value.clamp(0.0, 255.0) as u8;
        }
    }
    Ok(Mat::new_rows_cols_with_data(size.height, size.width, &out)?.try_clone()?)
}

/// One forward pass. Preprocessing/postprocessing match the model's own
/// reference implementation (`onnx_runner/docshadow.py` in the source
/// repo): RGB, scaled to [0, 1], NCHW, float32 in; same layout out, scaled
/// back to [0, 255] and clamped.
fn infer(session: &Mutex<Session>, bgr: &Mat) -> Result<Mat, AppError> {
    let mut rgb = Mat::default();
    imgproc::cvt_color_def(bgr, &mut rgb, imgproc::COLOR_BGR2RGB)?;

    let size = rgb.size()?;
    let (width, height) = (size.width as usize, size.height as usize);
    let pixels = rgb.data_typed::<Vec3b>()?;

    let plane = height * width;
    let mut data = vec![0f32; 3 * plane];
    for (index, pixel) in pixels.iter().enumerate() {
        for channel in 0..3 {
            data[channel * plane + index] = f32::from(pixel[channel]) / 255.0;
        }
    }

    let input = Tensor::from_array(([1_usize, 3, height, width], data))?;
    let mut session = session.lock().expect("docshadow ort Session mutex poisoned");
    let outputs = session.run(ort::inputs![INPUT_NAME => input])?;
    let (_shape, out_data) = outputs[OUTPUT_NAME].try_extract_tensor::<f32>()?;

    let mut out_pixels = vec![Vec3b::default(); plane];
    for (index, pixel) in out_pixels.iter_mut().enumerate() {
        for channel in 0..3 {
            let value = out_data[channel * plane + index] * 255.0;
            pixel[channel] = value.clamp(0.0, 255.0) as u8;
        }
    }
    let out_rgb = Mat::new_rows_cols_with_data(size.height, size.width, &out_pixels)?.try_clone()?;

    let mut out_bgr = Mat::default();
    imgproc::cvt_color_def(&out_rgb, &mut out_bgr, imgproc::COLOR_RGB2BGR)?;
    Ok(out_bgr)
}
