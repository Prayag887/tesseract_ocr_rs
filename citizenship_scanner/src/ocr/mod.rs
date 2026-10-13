//! OCR models and pipeline stages; no field or retry logic (that's `extraction`).
//! - [`OcrEngine::detect`]: image -> line boxes (`detection.rs`, the slow stage)
//! - [`OcrEngine::read_quads`]: boxes -> oriented (`orientation.rs`), recognized
//!   (`recognition.rs`) [`OcrLine`]s in reading order
//! - [`OcrEngine::recognize`]: recognize pre-cut crops
//! `tensor.rs`: shared model I/O. `debug.rs`: box overlay image.

mod debug;
mod detection;
mod orientation;
mod recognition;
mod tensor;

use std::sync::Mutex;

use opencv::core::{Mat, Point2f, Vector};
use opencv::imgcodecs;
use ort::session::Session;
use ort::session::builder::PrepackedWeights;

pub(crate) use debug::draw_debug_bboxes;
pub(crate) use orientation::Orientation;

use crate::error::AppError;
use crate::fields::{self, OcrLine};

/// Line box in page pixels, TL/TR/BR/BL.
pub(crate) type Quad = [Point2f; 4];

const MAX_BATCH_SIZE: usize = 64;

pub struct OcrEngine {
    det: Mutex<Session>,
    textline_ori: Mutex<Session>,
    rec: Vec<Mutex<Session>>,
    rec_vocab: Vec<&'static str>,
    min_confidence: f32,
    orientation_batch_size: usize,
    recognition_batch_size: usize,
    /// Debug image directory.
    output_dir: std::path::PathBuf,
}

impl OcrEngine {
    pub fn load(output_dir: std::path::PathBuf) -> Result<Self, AppError> {
        let det = crate::ort_config::session_builder()?.commit_from_file(detection::DET_MODEL_PATH)?;
        let textline_ori =
            crate::ort_config::session_builder()?.commit_from_file(orientation::TEXTLINE_ORI_MODEL_PATH)?;
        let mut rec_vocab = Vec::with_capacity(570);
        rec_vocab.push(""); // CTC blank
        rec_vocab.extend(recognition::REC_DICT.lines());
        rec_vocab.push(" ");

        let min_confidence = std::env::var("CITIZENSHIP_OCR_MIN_CONFIDENCE")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(0.45);
        let orientation_batch_size =
            env_usize("OCR_ORIENTATION_BATCH_SIZE", orientation::DEFAULT_ORIENTATION_BATCH_SIZE)
                .clamp(1, MAX_BATCH_SIZE);
        let recognition_batch_size =
            env_usize("OCR_RECOGNITION_BATCH_SIZE", recognition::DEFAULT_RECOGNITION_BATCH_SIZE)
                .clamp(1, MAX_BATCH_SIZE);
        let recognition_workers =
            env_usize("OCR_RECOGNITION_WORKERS", recognition::default_recognition_workers())
                .clamp(1, recognition::MAX_RECOGNITION_WORKERS);
        // One prepacked-weights container for the whole pool, so identical weights are stored once.
        let rec_weights = PrepackedWeights::new();
        let mut rec = Vec::with_capacity(recognition_workers);
        for _ in 0..recognition_workers {
            rec.push(Mutex::new(
                crate::ort_config::rec_session_builder(&rec_weights)?
                    .commit_from_file(recognition::REC_MODEL_PATH)?,
            ));
        }
        tracing::info!(recognition_workers, "OCR recognition workers configured");

        Ok(Self {
            det: Mutex::new(det),
            textline_ori: Mutex::new(textline_ori),
            rec,
            rec_vocab,
            min_confidence,
            orientation_batch_size,
            recognition_batch_size,
            output_dir,
        })
    }

    pub(crate) fn output_dir(&self) -> &std::path::Path {
        &self.output_dir
    }

    /// Line boxes on `image`; `label` only tags debug timing.
    pub(crate) fn detect(&self, image: &Mat, debug: bool, label: &str) -> Result<Vec<Quad>, AppError> {
        let t1 = std::time::Instant::now();
        let quads = detection::detect_text_boxes(&self.det, image)?;
        if debug {
            eprintln!("[ocr_timing] {label} detect_text_boxes: {:?} ({} boxes)", t1.elapsed(), quads.len());
        }
        Ok(quads)
    }

    /// Crop, orient, recognize; drops lines under `min_confidence`; reading order. Also returns
    /// whether crops were flipped.
    pub(crate) fn read_quads(
        &self,
        image: &Mat,
        quads: &[Quad],
        debug: bool,
        label: &str,
        orientation: Orientation,
    ) -> Result<(Vec<OcrLine>, bool), AppError> {
        let t2 = std::time::Instant::now();
        let mut crops = Vec::with_capacity(quads.len());
        for quad in quads {
            crops.push(detection::crop_quad(image, quad)?);
        }
        let (crops, flipped) = orientation::classify_and_fix_rotation_batch(
            &self.textline_ori,
            crops,
            self.orientation_batch_size,
            orientation,
        )?;
        if std::env::var("OCR_DEBUG_CROPS").is_ok() {
            for (i, crop) in crops.iter().enumerate() {
                let _ = imgcodecs::imwrite(
                    &self.output_dir.join(format!("crop_{label}_{i}.png")),
                    crop,
                    &Vector::new(),
                );
            }
        }
        if debug {
            eprintln!("[ocr_timing] {label} crop + batched orientation: {:?}", t2.elapsed());
        }

        let t3 = std::time::Instant::now();
        let recognized =
            recognition::recognize_lines_parallel(&self.rec, &self.rec_vocab, &crops, self.recognition_batch_size)?;
        if debug {
            eprintln!("[ocr_timing] {label} batched recognition: {:?}", t3.elapsed());
        }

        let mut lines = Vec::with_capacity(quads.len());
        for (quad, (text, confidence)) in quads.iter().zip(recognized) {
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
        fields::sort_reading_order(&mut lines);
        Ok((lines, flipped))
    }

    /// Recognize crops as-is (no orientation, no confidence filter).
    pub(crate) fn recognize(&self, crops: &[Mat]) -> Result<Vec<(String, f32)>, AppError> {
        recognition::recognize_lines_batch(&self.rec[0], &self.rec_vocab, crops, 1)
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
