use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Multipart, Path, Query, State};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::AppError;
use crate::fields::{self, CitizenshipDocument};
use crate::local_ocr::LocalOcrEngine;
use crate::scanner;

/// A scan's original upload and current crop, held in memory for the
/// lifetime of one scan/crop/extract session — never written to disk. An
/// earlier version persisted these as `{id}.jpg`/`{id}_original.jpg` files
/// under `output_dir`, deleted again by `extract`; that meant every
/// abandoned scan (uploaded, never extracted) left an orphaned file behind
/// permanently, and — worse — `output_dir` in production is a persistent
/// Docker volume, so those orphans survived container recreates too. This
/// data was never meant to outlive one request cycle; it shouldn't have
/// been touching disk (or a volume-backed mount) in the first place.
struct ScanEntry {
    original: Vec<u8>,
    crop: Vec<u8>,
}

pub struct AppState {
    pub detector: scanner::DocDetector,
    pub ocr: LocalOcrEngine,
    pub processing_permits: tokio::sync::Semaphore,
    scans: tokio::sync::Mutex<HashMap<Uuid, ScanEntry>>,
}

impl AppState {
    pub fn new(detector: scanner::DocDetector, ocr: LocalOcrEngine, processing_permits: tokio::sync::Semaphore) -> Self {
        Self { detector, ocr, processing_permits, scans: tokio::sync::Mutex::new(HashMap::new()) }
    }
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

    let _permit = state
        .processing_permits
        .acquire()
        .await
        .expect("processing semaphore is never closed");

    let state_for_scan = state.clone();
    // Kept for /crop to re-derive the crop from without re-uploading and
    // without compounding JPEG re-encode loss — cloned once here since the
    // blocking task below consumes `image_bytes`.
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
    state.scans.lock().await.insert(id, ScanEntry { original, crop: output.jpeg_bytes });

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
    // The id must still exist (checked above) unless it was deleted by a
    // concurrent /extract for the same id — an unlikely enough race for
    // this single-user flow that silently no-op'ing beats adding locking
    // just to guard it.
    if let Some(entry) = state.scans.lock().await.get_mut(&id) {
        entry.crop = output.jpeg_bytes;
    }

    Ok(Json(response))
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
    let image = {
        let scans = state.scans.lock().await;
        scans.get(&id).ok_or(AppError::NotFound)?.crop.clone()
    };
    let _permit = state
        .processing_permits
        .acquire()
        .await
        .expect("processing semaphore is never closed");
    let document = state.ocr.extract(&image, query.side.as_deref()).await?;

    // Nothing about a scanned citizenship certificate needs to persist past
    // its own extraction.
    state.scans.lock().await.remove(&id);

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
