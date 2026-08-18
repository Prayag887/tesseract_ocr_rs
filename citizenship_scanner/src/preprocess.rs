//! OCR-only image cleanup run on the cropped document, right before text
//! detection. Never persisted to disk or returned to the caller (except the
//! fixed-name debug dump in `local_ocr.rs`) — `scanner.rs`'s crop stays
//! visually faithful for display/audit, only the copy fed to the OCR
//! models is cleaned.
//!
//! Nepali citizenship certificates are printed on watermarked security
//! paper: the government emblem's associated phrase ("नेपाली नागरिकताको
//! प्रमाणपत्र") is tiled across the *entire* page as light-gray repeating
//! text, not a smooth tint or stain. An earlier version of this module
//! tried the standard illumination-division trick (divide by a heavily
//! blurred copy of the image to cancel smooth background variation), which
//! is the wrong tool here: that trick works when the unwanted background
//! is low-frequency (paper shading, vignetting) and the wanted content is
//! high-frequency (text strokes) — but this watermark *is* text, at
//! similar stroke width to the real foreground text, so any blur radius
//! that suppresses it also blurs real content, and dividing by an
//! under-blurred estimate amplified the watermark instead of cancelling
//! it. Confirmed against a real scan's debug output: the watermark came
//! out nearly as dark as genuine field text, and the OCR recognizer read
//! chunks of it as if it were real content sitting in otherwise-blank
//! fields.
//!
//! What actually separates the two on this document is *ink darkness*, not
//! spatial frequency: the watermark prints as a lighter gray than the bold
//! black foreground text, at any resolution. Adaptive thresholding exploits
//! that directly — each pixel is compared against a locally-computed
//! threshold (not one global cutoff, which uneven photo lighting would
//! throw off) and only pixels dark enough relative to their neighborhood
//! survive as foreground; everything else, watermark included, goes white.

use opencv::core::{Mat, Size};
use opencv::prelude::*;
use opencv::imgproc;

use crate::error::AppError;

/// Odd, and scaled to the image's own resolution rather than a fixed pixel
/// count — a fixed block size tuned for a 1000px-wide crop would be far too
/// small (over-fragmented local thresholds) on a 4000px one. Large enough
/// to span several lines of text so the local threshold reflects genuine
/// page-level lighting, not one letter's own ink.
fn threshold_block_size(width: i32, height: i32) -> i32 {
    let short_side = width.min(height).max(1);
    let size = (f64::from(short_side) / 20.0).round() as i32;
    let size = size.max(31);
    if size % 2 == 0 { size + 1 } else { size }
}

/// Binarizes the page so the lighter-gray watermark text drops out and the
/// bold black foreground text survives — see the module docs for why this
/// replaced an illumination-division approach. `C` (subtracted from each
/// pixel's local mean before comparing) is the knob that controls how
/// aggressively "gray but not black enough" gets treated as background;
/// tuned against a real watermarked scan, not derived analytically.
const ADAPTIVE_THRESHOLD_C: f64 = 12.0;

// A phone photo of a full citizenship certificate at well under ~2200px on
// its short side (confirmed against a real 1270x938 upload) leaves some
// rows' font — noticeably smaller than the field values' font on this card
// — only a few pixels tall, and those rows came back missing entirely
// rather than just misread. Upscaling to compensate was tried two ways
// (`INTER_CUBIC` before thresholding, `INTER_NEAREST`/`INTER_LINEAR` on the
// already-binarized result afterward) and made real-scan results *worse*
// both times — cubic rings at edges and the threshold step reads that
// ringing as spurious contrast; nearest/linear on already-binary pixels
// gives the text detector a mass of small blocky/fuzzy shapes to chase
// instead of the same clean strokes it had at native resolution. Left out.
// A genuinely low-resolution source photo is a hard limit this service
// can't preprocess its way around — the fix is a higher-resolution photo.

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

    // Adaptive thresholding, run per-pixel, routinely punctures a real
    // stroke into two or more disconnected specks wherever that stroke
    // crosses a locally weaker-contrast patch (residual paper grain, a
    // faded dot-matrix-printed row) — confirmed against a real scan where
    // this fragmented several field rows into a scatter of single-glyph
    // boxes instead of one coherent text line, which the recognizer then
    // read as garbage. A morphological close (dilate then erode) on the
    // *text* — inverted first since OpenCV's morphology ops treat the
    // brighter pixels as foreground, and here the text is the darker
    // ones — bridges those small gaps back together without measurably
    // thickening or merging separate characters, since the closing
    // kernel is far smaller than the gap between two different glyphs.
    let mut inverted = Mat::default();
    opencv::core::bitwise_not_def(&binary, &mut inverted)?;
    let kernel = imgproc::get_structuring_element_def(imgproc::MORPH_RECT, Size::new(2, 2))?;
    let mut closed_inverted = Mat::default();
    imgproc::morphology_ex_def(&inverted, &mut closed_inverted, imgproc::MORPH_CLOSE, &kernel)?;
    let mut closed = Mat::default();
    opencv::core::bitwise_not_def(&closed_inverted, &mut closed)?;

    // Remaining salt-and-pepper speckle (paper grain, JPEG artifacts
    // crossing the local threshold in isolated pixels) that's too small
    // and scattered for the closing step above to have bridged into
    // anything: median blur removes it without softening a real stroke's
    // edges the way a Gaussian blur would.
    let mut denoised = Mat::default();
    imgproc::median_blur(&closed, &mut denoised, 3)?;

    // Both downstream models (text detector, recognizer) were trained on
    // 3-channel color crops and normalize per-channel — feed them a
    // 3-channel image even though it's now pure black/white, rather than
    // changing their input contract for this one service.
    let mut bgr_out = Mat::default();
    imgproc::cvt_color_def(&denoised, &mut bgr_out, imgproc::COLOR_GRAY2BGR)?;
    Ok(bgr_out)
}
