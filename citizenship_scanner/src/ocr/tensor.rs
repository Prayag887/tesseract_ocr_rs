//! Model I/O shared by all three OCR models: tensor names, ImageNet normalization, `forward`.

use std::sync::Mutex;

use opencv::core::{Mat, Vec3b};
use opencv::prelude::*;
use ort::session::Session;
use ort::value::Tensor;

use crate::error::AppError;

const NET_INPUT_NAME: &str = "x";

const NET_OUTPUT_NAME: &str = "fetch_name_0";

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub(super) fn forward(
    net: &Mutex<Session>,
    shape: [usize; 4],
    data: Vec<f32>,
) -> Result<(Vec<i64>, Vec<f32>), AppError> {
    let input = Tensor::from_array((shape, data))?;
    let mut net = net.lock().expect("ort Session mutex poisoned");
    let outputs = net.run(ort::inputs![NET_INPUT_NAME => input])?;
    let (out_shape, out_data) = outputs[NET_OUTPUT_NAME].try_extract_tensor::<f32>()?;
    Ok((out_shape.to_vec(), out_data.to_vec()))
}

pub(super) fn imagenet_nchw(rgb: &Mat) -> Result<Vec<f32>, AppError> {
    let rows = rgb.rows();
    let cols = rgb.cols();
    let pixels = rgb.data_typed::<Vec3b>()?;
    let plane = (rows * cols) as usize;
    let mut out = vec![0f32; 3 * plane];
    for (index, pixel) in pixels.iter().enumerate() {
        for channel in 0..3 {
            let value = f32::from(pixel[channel]) / 255.0;
            out[channel * plane + index] = (value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
        }
    }
    Ok(out)
}
