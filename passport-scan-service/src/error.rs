use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(thiserror::Error, Debug)]
pub enum AppError {
    #[error("no image field found in multipart body")]
    MissingImage,
    #[error("uploaded image is empty")]
    EmptyImage,
    #[error("scan id not found")]
    NotFound,
    #[error("failed to decode image: {0}")]
    Decode(#[source] opencv::Error),
    #[error("image processing failed: {0}")]
    Processing(#[source] opencv::Error),
    #[error("invalid crop: {0}")]
    InvalidCrop(&'static str),
    #[error("failed to persist scanned image: {0}")]
    Io(#[source] std::io::Error),
    #[error("multipart error: {0}")]
    Multipart(#[from] axum::extract::multipart::MultipartError),
    #[error("blocking image task failed: {0}")]
    BlockingTask(#[source] tokio::task::JoinError),
    #[error("MRZ not found or unreadable on this document")]
    MrzNotFound,
    #[error("MRZ checksum validation failed: {0}")]
    MrzChecksum(&'static str),
}

impl From<opencv::Error> for AppError {
    fn from(e: opencv::Error) -> Self {
        AppError::Processing(e)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match &self {
            AppError::MissingImage
            | AppError::EmptyImage
            | AppError::Decode(_)
            | AppError::InvalidCrop(_)
            | AppError::MrzNotFound
            | AppError::MrzChecksum(_) => StatusCode::BAD_REQUEST,
            AppError::NotFound => StatusCode::NOT_FOUND,
            AppError::Multipart(_) => StatusCode::BAD_REQUEST,
            AppError::Processing(_) | AppError::Io(_) | AppError::BlockingTask(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };
        let message = match &self {
            AppError::MissingImage => "no image field found in multipart body",
            AppError::EmptyImage => "uploaded image is empty",
            AppError::NotFound => "scan id not found",
            AppError::Decode(_) => "failed to decode image",
            AppError::InvalidCrop(message) => message,
            AppError::Multipart(_) => "invalid multipart upload",
            AppError::MrzNotFound => "MRZ not found or unreadable on this document",
            AppError::MrzChecksum(message) => message,
            AppError::Processing(_) | AppError::Io(_) | AppError::BlockingTask(_) => {
                "internal server error"
            }
        };
        if status.is_server_error() {
            tracing::error!(error = ?self, "request failed");
        }
        let body = Json(json!({ "error": message }));
        (status, body).into_response()
    }
}
