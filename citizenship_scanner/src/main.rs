mod error;
mod fields;
mod local_ocr;
mod ort_config;
mod preprocess;
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
/// Distinct from the NID service's port (3000) and the passport service's
/// port (3003) so all three can run side by side on one host during local
/// dev.
const DEFAULT_PORT: u16 = 3004;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // Before any session is built — `ort` installs a default environment
    // lazily on first use, and this is a no-op afterwards.
    ort_config::init_runtime().expect("failed to configure ONNX Runtime");

    let output_dir = std::path::PathBuf::from("scanned_document");
    std::fs::create_dir_all(&output_dir).expect("failed to create scanned_document directory");

    let detector = scanner::DocDetector::load().expect("failed to load document detection model");
    let ocr = local_ocr::LocalOcrEngine::load(output_dir.clone()).expect("failed to load OCR models");
    let max_concurrency = std::env::var("OCR_MAX_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2)
        .clamp(1, 64);
    tracing::info!(max_concurrency, "image-processing concurrency configured");

    let state = Arc::new(AppState::new(detector, ocr, tokio::sync::Semaphore::new(max_concurrency)));

    let app = Router::new()
        .route("/health", get(routes::health))
        .route("/scan", post(routes::scan))
        .route("/crop/{id}", post(routes::crop))
        .route("/extract/{id}", post(routes::extract))
        .route("/combine", post(routes::combine))
        .nest_service("/scanned_document", ServeDir::new(&output_dir))
        .fallback_service(ServeDir::new("static"))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES))
        .layer(TraceLayer::new_for_http())
        // Permissive for the same reason as the NID/passport services: this
        // is reached only through the shared gateway today, not exposed
        // directly. Tighten to an explicit origin allowlist before that
        // changes.
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
