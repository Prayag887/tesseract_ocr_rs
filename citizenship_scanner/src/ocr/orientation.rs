//! Line crop orientation: one page-wide classifier vote, or a forced [`Orientation`].

use std::sync::Mutex;

use opencv::core::{Mat, Size};
use opencv::imgproc;
use ort::session::Session;

use super::tensor::{forward, imagenet_nchw};
use crate::error::AppError;

pub(super) const TEXTLINE_ORI_MODEL_PATH: &str = "models/textline_ori.onnx";

const TEXTLINE_ORI_WIDTH: i32 = 160;

const TEXTLINE_ORI_HEIGHT: i32 = 80;

pub(super) const DEFAULT_ORIENTATION_BATCH_SIZE: usize = 8;

/// `Auto` = classifier vote; `Upright`/`Flipped` force it (fallback retries).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Orientation {
    Auto,
    Upright,
    Flipped,
}

/// Flips all crops or none, by one page-wide vote: per-line votes flipped short upright lines on
/// their own, which reads as garbage. A forced `orientation` skips the vote. Returns whether
/// crops were flipped.
pub(super) fn classify_and_fix_rotation_batch(
    textline_ori: &Mutex<Session>,
    mut crops: Vec<Mat>,
    batch_size: usize,
    orientation: Orientation,
) -> Result<(Vec<Mat>, bool), AppError> {
    let flip = match orientation {
        Orientation::Auto => page_votes_flipped(textline_ori, &crops, batch_size)?,
        Orientation::Upright => false,
        Orientation::Flipped => true,
    };
    if flip {
        for crop in crops.iter_mut() {
            let mut rotated = Mat::default();
            opencv::core::flip(crop, &mut rotated, -1)?;
            *crop = rotated;
        }
    }
    Ok((crops, flip))
}

fn page_votes_flipped(
    textline_ori: &Mutex<Session>,
    crops: &[Mat],
    batch_size: usize,
) -> Result<bool, AppError> {
    let mut total_upright = 0.0f32;
    let mut total_flipped = 0.0f32;
    for chunk in crops.chunks(batch_size) {
        let sample_len = (3 * TEXTLINE_ORI_HEIGHT * TEXTLINE_ORI_WIDTH) as usize;
        let mut data = Vec::with_capacity(chunk.len() * sample_len);
        for crop in chunk.iter() {
            let mut resized = Mat::default();
            imgproc::resize(
                crop,
                &mut resized,
                Size::new(TEXTLINE_ORI_WIDTH, TEXTLINE_ORI_HEIGHT),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;
            let mut rgb = Mat::default();
            imgproc::cvt_color_def(&resized, &mut rgb, imgproc::COLOR_BGR2RGB)?;
            data.extend(imagenet_nchw(&rgb)?);
        }
        let (_shape, out) = forward(
            textline_ori,
            [
                chunk.len(),
                3,
                TEXTLINE_ORI_HEIGHT as usize,
                TEXTLINE_ORI_WIDTH as usize,
            ],
            data,
        )?;

        for index in 0..chunk.len() {
            total_upright += out[index * 2];
            total_flipped += out[index * 2 + 1];
        }
    }

    Ok(total_flipped > total_upright)
}
