//! Watermark cleanup on the OCR copy of the crop (the returned crop stays untouched).
//! The card's watermark is tiled light-gray *text*, so blur/illumination-division can't separate
//! it (tried: amplified it). Ink darkness can: adaptive thresholding keeps dark strokes and drops
//! the lighter watermark.

use opencv::core::{Mat, Size};
use opencv::prelude::*;
use opencv::imgproc;

use crate::error::AppError;

/// Odd block size scaled to resolution, spanning several text lines.
fn threshold_block_size(width: i32, height: i32) -> i32 {
    let short_side = width.min(height).max(1);
    let size = (f64::from(short_side) / 20.0).round() as i32;
    let size = size.max(31);
    if size % 2 == 0 { size + 1 } else { size }
}

/// Threshold offset: how gray counts as background. Tuned on a real watermarked scan.
const ADAPTIVE_THRESHOLD_C: f64 = 12.0;

// Don't upscale small photos here: tried cubic before and nearest/linear after thresholding,
// both made results worse. Low resolution needs a better photo.

pub fn denoise_for_ocr(bgr: &Mat) -> Result<Mat, AppError> {
    let size = bgr.size()?;
    let (width, height) = (size.width, size.height);

    let mut gray = Mat::default();
    imgproc::cvt_color_def(bgr, &mut gray, imgproc::COLOR_BGR2GRAY)?;

    let block_size = threshold_block_size(width, height);
    let mut binary = Mat::default();
    imgproc::adaptive_threshold(
        &gray,
        &mut binary,
        255.0,
        imgproc::ADAPTIVE_THRESH_GAUSSIAN_C,
        imgproc::THRESH_BINARY,
        block_size,
        ADAPTIVE_THRESHOLD_C,
    )?;

    // Close small gaps thresholding punches into strokes (on inverted text: morphology treats
    // bright as foreground). Kernel is far smaller than inter-glyph gaps.
    let mut inverted = Mat::default();
    opencv::core::bitwise_not_def(&binary, &mut inverted)?;
    let kernel = imgproc::get_structuring_element_def(imgproc::MORPH_RECT, Size::new(2, 2))?;
    let mut closed_inverted = Mat::default();
    imgproc::morphology_ex_def(&inverted, &mut closed_inverted, imgproc::MORPH_CLOSE, &kernel)?;
    let mut closed = Mat::default();
    opencv::core::bitwise_not_def(&closed_inverted, &mut closed)?;

    // Median blur removes remaining speckle without softening edges.
    let mut denoised = Mat::default();
    imgproc::median_blur(&closed, &mut denoised, 3)?;

    // Models expect 3-channel input.
    let mut bgr_out = Mat::default();
    imgproc::cvt_color_def(&denoised, &mut bgr_out, imgproc::COLOR_GRAY2BGR)?;
    Ok(bgr_out)
}
