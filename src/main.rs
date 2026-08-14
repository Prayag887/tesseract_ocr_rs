mod error;
mod llm_client;
mod local_ocr;
mod ocr;
mod routes;
mod scanner;

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;

use routes::{AppState, OcrBackend};

const MAX_UPLOAD_BYTES: usize = 25 * 1024 * 1024; // 25 MB

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let output_dir = std::path::PathBuf::from("scanned_document");
    std::fs::create_dir_all(&output_dir).expect("failed to create scanned_document directory");

    let detector = scanner::DocDetector::load().expect("failed to load document detection model");
    let ocr_backend = std::env::var("OCR_BACKEND").unwrap_or_else(|_| "local".to_owned());
    let ocr = match ocr_backend.as_str() {
        "local" => OcrBackend::Local(
            local_ocr::LocalOcrEngine::load().expect("failed to load native OCR models"),
        ),
        "paddlex" => OcrBackend::Paddle(
            ocr::PaddleOcrClient::from_env().expect("failed to configure PP-OCRv5 client"),
        ),
        other => panic!("invalid OCR_BACKEND {other:?}, expected \"local\" or \"paddlex\""),
    };
    tracing::info!(backend = %ocr_backend, "OCR backend selected");

    let llm = match std::env::var("FIELD_EXTRACTOR").as_deref() {
        Ok("rules") => None,
        _ => match llm_client::LlmVerifier::spawn() {
            Ok(verifier) => Some(verifier),
            Err(error) => {
                tracing::warn!(%error, "failed to start llm_verifier sidecar, falling back to rule-based field extraction");
                None
            }
        },
    };
    tracing::info!(field_extractor = if llm.is_some() { "llm" } else { "rules" }, "field extraction backend selected");

    let state = Arc::new(AppState {
        output_dir: output_dir.clone(),
        detector,
        ocr,
        llm,
    });

    let app = Router::new()
        .route("/health", get(routes::health))
        .route("/scan", post(routes::scan))
        .route("/crop/{id}", post(routes::crop))
        .route("/extract/{id}", post(routes::extract))
        .nest_service("/scanned_document", ServeDir::new(&output_dir))
        .fallback_service(ServeDir::new("static"))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .expect("failed to bind to 0.0.0.0:3000");
    tracing::info!("listening on {}", listener.local_addr().unwrap());

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install Ctrl+C handler");
    tracing::info!("shutdown signal received");
}
