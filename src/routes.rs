use std::sync::Arc;

use axum::extract::{Multipart, Path, State};
use axum::Json;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::AppError;
use crate::ocr::{self, Field};
use crate::scanner;

pub struct AppState {
    pub output_dir: std::path::PathBuf,
    pub detector: scanner::DocDetector,
}

#[derive(Serialize)]
pub struct ScanResponse {
    id: String,
    filename: String,
    corners_detected: bool,
    width: i32,
    height: i32,
    /// Crop corners in the *original* upload's pixel space, ordered
    /// [top-left, top-right, bottom-right, bottom-left], as [x, y] pairs.
    corners: [[f32; 2]; 4],
    orig_width: i32,
    orig_height: i32,
}

#[derive(Deserialize)]
pub struct CropRequest {
    /// [x, y] pairs, any order intended as [top-left, top-right, bottom-right, bottom-left].
    corners: [[f32; 2]; 4],
}

#[derive(Serialize)]
pub struct CropResponse {
    filename: String,
    width: i32,
    height: i32,
}

pub async fn health() -> &'static str {
    "ok"
}

pub async fn scan(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<ScanResponse>, AppError> {
    let mut image_bytes: Option<Vec<u8>> = None;

    while let Some(field) = multipart.next_field().await? {
        if field.name() == Some("image") || image_bytes.is_none() {
            image_bytes = Some(field.bytes().await?.to_vec());
        }
    }
    let image_bytes = image_bytes.ok_or(AppError::MissingImage)?;
    if image_bytes.is_empty() {
        return Err(AppError::EmptyImage);
    }

    let id = Uuid::new_v4();

    // Keep the original around so the user can re-crop later without
    // re-uploading and without compounding JPEG re-encode loss.
    let original_path = state.output_dir.join(format!("{id}_original.jpg"));
    tokio::fs::write(&original_path, &image_bytes)
        .await
        .map_err(AppError::Io)?;

    // OpenCV/dnn work is synchronous and CPU-bound; run it on the blocking
    // pool so it never stalls the async reactor.
    let state_for_scan = state.clone();
    let output = tokio::task::spawn_blocking(move || {
        scanner::scan_document(&state_for_scan.detector, &image_bytes)
    })
    .await
    .expect("scan_document panicked")?;

    let filename = format!("{id}.jpg");
    let path = state.output_dir.join(&filename);
    tokio::fs::write(&path, &output.jpeg_bytes)
        .await
        .map_err(AppError::Io)?;

    Ok(Json(ScanResponse {
        id: id.to_string(),
        filename,
        corners_detected: output.corners_detected,
        width: output.width,
        height: output.height,
        corners: output.corners.map(|(x, y)| [x, y]),
        orig_width: output.orig_width,
        orig_height: output.orig_height,
    }))
}

pub async fn crop(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<CropRequest>,
) -> Result<Json<CropResponse>, AppError> {
    let original_path = state.output_dir.join(format!("{id}_original.jpg"));
    let image_bytes = tokio::fs::read(&original_path)
        .await
        .map_err(|_| AppError::NotFound)?;

    let corners = req.corners.map(|[x, y]| (x, y));
    let output = tokio::task::spawn_blocking(move || scanner::crop_with_corners(&image_bytes, corners))
        .await
        .expect("crop_with_corners panicked")?;

    let filename = format!("{id}.jpg");
    let path = state.output_dir.join(&filename);
    tokio::fs::write(&path, &output.jpeg_bytes)
        .await
        .map_err(AppError::Io)?;

    Ok(Json(CropResponse {
        filename,
        width: output.width,
        height: output.height,
    }))
}

pub async fn extract(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<Vec<Field>>, AppError> {
    let path = state.output_dir.join(format!("{id}.jpg"));
    if !tokio::fs::try_exists(&path).await.unwrap_or(false) {
        return Err(AppError::NotFound);
    }

    // Shells out to the `tesseract` CLI; run on the blocking pool like the
    // other CPU/process-bound work so it doesn't stall the async reactor.
    let fields = tokio::task::spawn_blocking(move || ocr::extract_fields(&path))
        .await
        .expect("extract_fields panicked")?;

    Ok(Json(fields))
}
