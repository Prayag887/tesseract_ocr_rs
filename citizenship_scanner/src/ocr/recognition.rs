//! Text recognition (PP-OCRv5 Devanagari+Latin, CTC): width-batched, parallel over a session pool,
//! greedy-decoded to `(text, mean confidence)`.

use std::sync::Mutex;

use opencv::core::{Mat, Size, Vec3b};
use opencv::imgproc;
use opencv::prelude::*;
use ort::session::Session;

use super::env_usize;
use super::tensor::forward;
use crate::error::AppError;

/// Shared with the NID/passport services; reads both scripts.
pub(super) const REC_MODEL_PATH: &str = "models/devanagari_rec.onnx";

pub(super) const REC_DICT: &str = include_str!("../../models/devanagari_rec_dict.txt");

const REC_HEIGHT: i32 = 48;

const REC_MAX_WIDTH: i32 = 2000;

pub(super) const DEFAULT_RECOGNITION_BATCH_SIZE: usize = 1;

pub(super) const MAX_RECOGNITION_WORKERS: usize = 8;

pub(super) fn default_recognition_workers() -> usize {
    let logical_cpus = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let request_concurrency = env_usize("OCR_MAX_CONCURRENCY", 2).clamp(1, 64);
    logical_cpus
        .div_ceil(request_concurrency)
        .clamp(1, MAX_RECOGNITION_WORKERS)
}

pub(super) fn recognize_lines_parallel(
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

pub(super) fn recognize_lines_batch(
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
