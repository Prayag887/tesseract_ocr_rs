mod error;
mod local_ocr;
mod ocr;
mod routes;
mod scanner;

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tower_http::cors::CorsLayer;
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
    let max_concurrency = std::env::var("OCR_MAX_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2)
        .clamp(1, 64);
    tracing::info!(max_concurrency, "image-processing concurrency configured");

    let state = Arc::new(AppState {
        output_dir: output_dir.clone(),
        detector,
        ocr,
        processing_permits: tokio::sync::Semaphore::new(max_concurrency),
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
        // Permissive: the shared UI switches tabs between this service
        // (port 3002) and passport-scan-service (port 3003) via absolute
        // fetch URLs, so both need to answer cross-origin. Both are
        // internal, unauthenticated local-dev services already — tighten
        // this to an explicit origin allowlist before any public exposure.
        .layer(CorsLayer::permissive())
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(3000);
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
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    tracing::info!("shutdown signal received");
}
