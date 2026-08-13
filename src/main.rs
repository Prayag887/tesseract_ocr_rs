mod error;
mod routes;
mod scanner;

use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use axum::Router;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;

use routes::AppState;

const MAX_UPLOAD_BYTES: usize = 25 * 1024 * 1024; // 25 MB

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let output_dir = std::path::PathBuf::from("scanned_document");
    std::fs::create_dir_all(&output_dir).expect("failed to create scanned_document directory");

    let detector = scanner::DocDetector::load().expect("failed to load document detection model");

    let state = Arc::new(AppState {
        output_dir: output_dir.clone(),
        detector,
    });

    let app = Router::new()
        .route("/health", get(routes::health))
        .route("/scan", post(routes::scan))
        .route("/crop/{id}", post(routes::crop))
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
