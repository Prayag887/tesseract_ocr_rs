use std::sync::Arc;

use axum::Json;
use axum::extract::{Multipart, Path, Query, State};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::AppError;
use crate::fields::{self, CitizenshipDocument};
use crate::local_ocr::LocalOcrEngine;
use crate::scanner;

pub struct AppState {
    pub output_dir: std::path::PathBuf,
    pub detector: scanner::DocDetector,
    pub ocr: LocalOcrEngine,
    pub processing_permits: tokio::sync::Semaphore,
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

    let _permit = state
        .processing_permits
        .acquire()
        .await
        .expect("processing semaphore is never closed");

    let state_for_scan = state.clone();
    let output = tokio::task::spawn_blocking(move || {
        scanner::scan_document(&state_for_scan.detector, &image_bytes)
    })
    .await
    .map_err(AppError::BlockingTask)??;

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
    let _permit = state
        .processing_permits
        .acquire()
        .await
        .expect("processing semaphore is never closed");
    let output =
        tokio::task::spawn_blocking(move || scanner::crop_with_corners(&image_bytes, corners))
            .await
            .map_err(AppError::BlockingTask)??;

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

#[derive(Deserialize)]
pub struct ExtractQuery {
    /// `"front"` or `"back"` — only affects which debug-preprocessed-image
    /// filename gets overwritten (see `local_ocr::extract_blocking`), not
    /// the extraction logic itself.
    side: Option<String>,
}

pub async fn extract(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Query(query): Query<ExtractQuery>,
) -> Result<Json<CitizenshipDocument>, AppError> {
    let path = state.output_dir.join(format!("{id}.jpg"));
    let image = tokio::fs::read(&path)
        .await
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::NotFound => AppError::NotFound,
            _ => AppError::Io(error),
        })?;
    let _permit = state
        .processing_permits
        .acquire()
        .await
        .expect("processing semaphore is never closed");
    let document = state.ocr.extract(&image, query.side.as_deref()).await?;

    // Nothing about a scanned citizenship certificate needs to persist past
    // its own extraction — best-effort cleanup of both the crop and the
    // original upload; a failed delete here (e.g. already gone) isn't worth
    // failing the request over.
    let _ = tokio::fs::remove_file(&path).await;
    let _ = tokio::fs::remove_file(state.output_dir.join(format!("{id}_original.jpg"))).await;

    Ok(Json(document))
}

#[derive(Deserialize)]
pub struct CombineRequest {
    front: Option<CitizenshipDocument>,
    back: Option<CitizenshipDocument>,
}

/// Merges a front-side and back-side `/extract` result into one document —
/// see `fields::combine` for which side wins per field.
pub async fn combine(Json(req): Json<CombineRequest>) -> Json<CitizenshipDocument> {
    let front = req.front.unwrap_or_default();
    let back = req.back.unwrap_or_default();
    Json(fields::combine(&front, &back))
}
