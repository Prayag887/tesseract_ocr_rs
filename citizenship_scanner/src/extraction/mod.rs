//! Uploaded side -> [`CitizenshipDocument`]. All decisions about which OCR runs to do live here;
//! `ocr` only knows how to run one.
//!
//! [`Extractor::extract_document`]:
//! 1. [`Extractor::read_image`] on the crop: decode, cleanup (`preprocess.rs`), detect + read,
//!    [`fields::extract`].
//! 2. Fewer than [`fields::LOW_YIELD_FIELDS`] fields (working scans never get here): re-parse with
//!    old-card rules, retry opposite orientation, retry raw crop both ways, keep the best,
//!    re-vote the citizenship number (`number.rs`).
//! 3. Still low-yield and the crop isn't the whole photo: steps 1-2 on the original upload.
//!
//! `CITIZENSHIP_OCR_FALLBACK=false` disables steps 2-3.

mod number;

use opencv::core::{Mat, Vector};
use opencv::imgcodecs;
use opencv::prelude::*;

use crate::error::AppError;
use crate::fields::{self, CitizenshipDocument};
use crate::ocr::{OcrEngine, Orientation, draw_debug_bboxes};
use crate::preprocess;

/// Skip cleanup below this short side: its fixed-size kernels erase strokes on small crops.
/// Set between real scans (needed: 938px+, broken: 599-636px); move only on new scan evidence.
const MIN_PREPROCESS_SHORT_SIDE: i32 = 800;

pub struct Extractor {
    engine: OcrEngine,
    /// `CITIZENSHIP_OCR_PREPROCESS` — off switch for the watermark cleanup.
    preprocess_enabled: bool,
    /// `CITIZENSHIP_OCR_FALLBACK` — off switch for steps 2-3.
    fallback_enabled: bool,
}

impl Extractor {
    pub fn load(output_dir: std::path::PathBuf) -> Result<Self, AppError> {
        let engine = OcrEngine::load(output_dir)?;
        let preprocess_enabled = std::env::var("CITIZENSHIP_OCR_PREPROCESS")
            .ok()
            .and_then(|value| value.parse::<bool>().ok())
            .unwrap_or(true);
        let fallback_enabled = std::env::var("CITIZENSHIP_OCR_FALLBACK")
            .ok()
            .and_then(|value| value.parse::<bool>().ok())
            .unwrap_or(true);
        Ok(Self { engine, preprocess_enabled, fallback_enabled })
    }

    /// Full extraction for one side. Pass `original` only when the crop isn't the full frame.
    /// Synchronous and CPU-bound.
    pub fn extract_document(
        &self,
        crop: &[u8],
        original: Option<&[u8]>,
        side: Option<&str>,
    ) -> Result<CitizenshipDocument, AppError> {
        let mut document = self.read_image(crop, side)?;

        // Card-edge detection can crop to a rectangle inside the card (old back: the fingerprint box).
        // Kept only if it strictly beats the crop.
        let populated = fields::populated_fields(&document);
        if self.fallback_enabled
            && populated < fields::LOW_YIELD_FIELDS
            && let Some(original) = original
        {
            let from_original = self.read_image(original, side)?;
            let count = fields::populated_fields(&from_original);
            tracing::info!(count, crop = populated, "low-yield OCR fallback on original upload");
            if count > populated {
                document = from_original;
            }
        }
        Ok(document)
    }

    /// One image through steps 1-2. `side` only names the debug image.
    fn read_image(&self, image: &[u8], side: Option<&str>) -> Result<CitizenshipDocument, AppError> {
        let debug = std::env::var("OCR_DEBUG_TIMING").is_ok();
        let t0 = std::time::Instant::now();
        let buf = Vector::<u8>::from_slice(image);
        let bgr = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR).map_err(AppError::Decode)?;
        if bgr.empty() {
            return Err(AppError::EmptyImage);
        }

        let side_name = match side {
            Some("front") => "front",
            Some("back") => "back",
            _ => "last",
        };
        let mut params = Vector::<i32>::new();
        params.push(imgcodecs::IMWRITE_JPEG_QUALITY);
        params.push(90);
        let dump = |image: &Mat, name: &str| {
            let mut bytes = Vector::<u8>::new();
            if imgcodecs::imencode(".jpg", image, &mut bytes, &params).is_ok() {
                let _ = std::fs::write(self.engine.output_dir().join(format!("{name}_{side_name}.jpg")), bytes.as_slice());
            }
        };

        let cleaned = {
            let crop_size = bgr.size()?;
            let short_side = crop_size.width.min(crop_size.height);
            // Don't pick between cleaned/raw by field count on working scans: tried, chose a wrong result.
            if self.preprocess_enabled && short_side >= MIN_PREPROCESS_SHORT_SIDE {
                let t_pre = std::time::Instant::now();
                let cleaned = preprocess::denoise_for_ocr(&bgr)?;
                if debug {
                    eprintln!("[ocr_timing] preprocess: {:?}", t_pre.elapsed());
                }
                Some(cleaned)
            } else {
                None
            }
        };
        let preprocessed = cleaned.is_some();
        let cleaned = cleaned.as_ref().unwrap_or(&bgr);

        let t1 = std::time::Instant::now();
        let quads = self.engine.detect(cleaned, debug, "primary")?;
        let (lines, flipped) = self.engine.read_quads(cleaned, &quads, debug, "primary", Orientation::Auto)?;
        if debug {
            eprintln!("[ocr_timing] detection pipeline total: {:?}", t1.elapsed());
        }
        let mut document = fields::extract(&lines);
        let mut best = (lines, cleaned);

        // Low-yield fallback (step 2). Old cards made the orientation vote flip upright text and
        // thresholding break rows, so retry both. Detection (seconds) runs once per image and is
        // reused across orientations. A retry must strictly beat the primary.
        let populated = fields::populated_fields(&document);
        if self.fallback_enabled && populated < fields::LOW_YIELD_FIELDS {
            // Lines may be fine and only the layout old.
            let mut best_count = populated;
            let reparsed = fields::extract_fallback(&best.0);
            let reparsed_count = fields::populated_fields(&reparsed);
            if reparsed_count > best_count {
                best_count = reparsed_count;
                document = reparsed;
            }

            let opposite = if flipped { Orientation::Upright } else { Orientation::Flipped };
            let raw_quads = if preprocessed { Some(self.engine.detect(&bgr, debug, "fallback-raw")?) } else { None };
            let mut attempts = vec![(cleaned, &quads, opposite, "fallback-reoriented")];
            if let Some(raw_quads) = &raw_quads {
                attempts.push((&bgr, raw_quads, Orientation::Upright, "fallback-raw-upright"));
                attempts.push((&bgr, raw_quads, Orientation::Flipped, "fallback-raw-flipped"));
            }
            for (image, quads, orientation, label) in attempts {
                let (lines, _) = self.engine.read_quads(image, quads, debug, label, orientation)?;
                let candidate = fields::extract_fallback(&lines);
                let count = fields::populated_fields(&candidate);
                tracing::info!(label, count, primary = populated, "low-yield OCR fallback attempt");
                if count > best_count {
                    best_count = count;
                    document = candidate;
                    best = (lines, image);
                }
            }

            // Always from the raw crop: binarization flips digits (१ -> ९). An inconclusive vote clears the
            // number — a wrong number is worse than none.
            if let Some(number) = number::reread_citizenship_number(&self.engine, &bgr, &best.0)? {
                document.citizenship_number = number;
            }
        }
        let (lines, ocr_input) = best;

        // Debug overlay of every kept box with its confidence; fixed filename per side.
        if let Ok(annotated) = draw_debug_bboxes(ocr_input, &lines) {
            dump(&annotated, "bboxed");
        }

        if std::env::var("OCR_DEBUG_LINES").is_ok() {
            for (i, line) in lines.iter().enumerate() {
                tracing::info!("line {i}: {:?} conf={:.3} poly={:?}", line.text, line.confidence, line.polygon);
            }
        }

        if debug {
            eprintln!("[ocr_timing] total: {:?}", t0.elapsed());
        }
        Ok(document)
    }
}
