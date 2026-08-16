mod error;
mod fields;
mod local_ocr;
mod mrz;
mod routes;
mod scanner;

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;

use routes::AppState;

const MAX_UPLOAD_BYTES: usize = 25 * 1024 * 1024; // 25 MB
/// Distinct from the NID service's port (3002) so both can run side by side
/// on one host during local dev.
const DEFAULT_PORT: u16 = 3003;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let output_dir = std::path::PathBuf::from("scanned_document");
    std::fs::create_dir_all(&output_dir).expect("failed to create scanned_document directory");

    let detector = scanner::DocDetector::load().expect("failed to load document detection model");
    let ocr = local_ocr::MrzOcrEngine::load().expect("failed to load MRZ OCR models");

    let state = Arc::new(AppState {
        output_dir: output_dir.clone(),
        detector,
        ocr,
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
        // See the NID service's src/main.rs (repo root) for why this is permissive.
        .layer(CorsLayer::permissive())
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_PORT);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .expect("failed to bind");
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
