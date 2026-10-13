use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Multipart, Path, Query, State};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::AppError;
use crate::fields::{self, CitizenshipDocument};
use crate::extraction::Extractor;
use crate::scanner;

/// Upload and current crop for one scan/crop/extract cycle — memory only, never disk
/// (files orphaned by abandoned scans used to pile up on the production volume).
struct ScanEntry {
    original: Vec<u8>,
    crop: Vec<u8>,
    /// No corners detected: the crop is the whole upload, so the original-upload retry is skipped.
    crop_is_full_frame: bool,
}

pub struct AppState {
    pub detector: scanner::DocDetector,
    pub extractor: Extractor,
    pub processing_permits: tokio::sync::Semaphore,
    scans: tokio::sync::Mutex<HashMap<Uuid, ScanEntry>>,
}

impl AppState {
    pub fn new(detector: scanner::DocDetector, extractor: Extractor, processing_permits: tokio::sync::Semaphore) -> Self {
        Self { detector, extractor, processing_permits, scans: tokio::sync::Mutex::new(HashMap::new()) }
    }
}

#[derive(Serialize)]
pub struct ScanResponse {
    id: String,
    filename: String,
    corners_detected: bool,
    width: i32,
    height: i32,
    /// Original-upload pixels, TL/TR/BR/BL, as [x, y].
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

    let _permit = state
        .processing_permits
        .acquire()
        .await
        .expect("processing semaphore is never closed");

    let state_for_scan = state.clone();
    // Kept so /crop can re-warp without re-encoding loss.
    let original = image_bytes.clone();
    let output = tokio::task::spawn_blocking(move || {
        scanner::scan_document(&state_for_scan.detector, &image_bytes)
    })
    .await
    .map_err(AppError::BlockingTask)??;

    let filename = format!("{id}.jpg");
    let response = ScanResponse {
        id: id.to_string(),
        filename,
        corners_detected: output.corners_detected,
        width: output.width,
        height: output.height,
        corners: output.corners.map(|(x, y)| [x, y]),
        orig_width: output.orig_width,
        orig_height: output.orig_height,
    };
    let crop_is_full_frame = !output.corners_detected;
    state.scans.lock().await.insert(id, ScanEntry { original, crop: output.jpeg_bytes, crop_is_full_frame });

    Ok(Json(response))
}

pub async fn crop(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<CropRequest>,
) -> Result<Json<CropResponse>, AppError> {
    let image_bytes = {
        let scans = state.scans.lock().await;
        scans.get(&id).ok_or(AppError::NotFound)?.original.clone()
    };

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
    let response = CropResponse { filename, width: output.width, height: output.height };
    // Gone only if a concurrent /extract removed it; ignored.
    if let Some(entry) = state.scans.lock().await.get_mut(&id) {
        entry.crop = output.jpeg_bytes;
        entry.crop_is_full_frame = false;
    }

    Ok(Json(response))
}

#[derive(Deserialize)]
pub struct ExtractQuery {
    /// "front" or "back"; only names the debug image.
    side: Option<String>,
}

pub async fn extract(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Query(query): Query<ExtractQuery>,
) -> Result<Json<CitizenshipDocument>, AppError> {
    let (image, original, crop_is_full_frame) = {
        let scans = state.scans.lock().await;
        let entry = scans.get(&id).ok_or(AppError::NotFound)?;
        (entry.crop.clone(), entry.original.clone(), entry.crop_is_full_frame)
    };
    let _permit = state
        .processing_permits
        .acquire()
        .await
        .expect("processing semaphore is never closed");
    let original = (!crop_is_full_frame).then_some(original.as_slice());
    let document = tokio::task::block_in_place(|| {
        state.extractor.extract_document(&image, original, query.side.as_deref())
    })?;

    state.scans.lock().await.remove(&id);

    Ok(Json(document))
}

#[derive(Deserialize)]
pub struct CombineRequest {
    front: Option<CitizenshipDocument>,
    back: Option<CitizenshipDocument>,
}

/// Merges front and back `/extract` results (see `fields::combine`).
pub async fn combine(Json(req): Json<CombineRequest>) -> Json<CitizenshipDocument> {
    let front = req.front.unwrap_or_default();
    let back = req.back.unwrap_or_default();
    Json(fields::combine(&front, &back))
}
