//! Fallback citizenship-number re-read: many crop variants, majority vote (`fields::vote_number`).

use opencv::core::{Mat, Rect};
use opencv::prelude::*;

use crate::error::AppError;
use crate::fields::{self, OcrLine};
use crate::ocr::OcrEngine;

/// Re-reads the number line (`fields::number_line`) from tighter crops and votes; padded boxes
/// misread these italic digits, and which trim works varies.
/// `Ok(None)`: no number line. `Ok(Some(None))`: reads disagreed.
pub(super) fn reread_citizenship_number(engine: &OcrEngine, image: &Mat, lines: &[OcrLine]) -> Result<Option<Option<String>>, AppError> {
    let Some(index) = fields::number_line(lines) else { return Ok(None) };
    let line = &lines[index];
    let size = image.size()?;
    let left = line.polygon.iter().map(|p| p[0]).min().unwrap_or(0).clamp(0, size.width - 1);
    let right = line.polygon.iter().map(|p| p[0]).max().unwrap_or(0).clamp(left + 1, size.width);
    let top = line.polygon.iter().map(|p| p[1]).min().unwrap_or(0).clamp(0, size.height - 1);
    let bottom = line.polygon.iter().map(|p| p[1]).max().unwrap_or(0).clamp(top + 1, size.height);
    let (width, height) = (right - left, bottom - top);

    // Digit start estimated from the first digit's char position, scaled down (label glyphs are narrower).
    let chars: Vec<char> = line.text.chars().collect();
    let first_digit = chars
        .iter()
        .position(|c| c.is_ascii_digit() || ('०'..='९').contains(c))
        .unwrap_or(0);
    let digit_fraction = first_digit as f32 / chars.len().max(1) as f32;

    // Horizontal jitter in line-height units keeps reads independent when the number starts the
    // line. 0.15 off the right is inside the box padding.
    let mut windows: Vec<(i32, i32)> = Vec::new();
    for start_scale in [0.5_f32, 0.7] {
        let x0 = left + (width as f32 * digit_fraction * start_scale) as i32;
        for (start_shift, end_trim) in [(0.0_f32, 0.0_f32), (0.25, 0.0), (0.0, 0.15)] {
            let window = (x0 + (height as f32 * start_shift) as i32, right - (height as f32 * end_trim) as i32);
            if window.1 - window.0 >= 8 && !windows.contains(&window) {
                windows.push(window);
            }
        }
    }
    let mut crops = Vec::new();
    for &(x0, x1) in &windows {
        for trim in [0.0_f32, 0.05, 0.1, 0.15, 0.2] {
            let y0 = top + (height as f32 * trim) as i32;
            let y1 = bottom - (height as f32 * trim) as i32;
            if y1 - y0 < 8 {
                continue;
            }
            crops.push(image.roi(Rect::new(x0, y0, x1 - x0, y1 - y0))?.try_clone()?);
        }
    }
    let mut reads = engine.recognize(&crops)?;
    reads.push((line.text.clone(), line.confidence));
    if std::env::var("OCR_DEBUG_LINES").is_ok() {
        tracing::info!(?reads, "citizenship number re-read candidates");
    }
    let number = fields::vote_number(&reads);
    tracing::info!(?number, reads = reads.len(), "citizenship number re-read");
    Ok(Some(number))
}
