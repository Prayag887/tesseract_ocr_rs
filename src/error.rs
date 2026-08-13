use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
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
    #[error("failed to persist scanned image: {0}")]
    Io(#[source] std::io::Error),
    #[error("multipart error: {0}")]
    Multipart(#[from] axum::extract::multipart::MultipartError),
}

impl From<opencv::Error> for AppError {
    fn from(e: opencv::Error) -> Self {
        AppError::Processing(e)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match &self {
            AppError::MissingImage | AppError::EmptyImage | AppError::Decode(_) => {
                StatusCode::BAD_REQUEST
            }
            AppError::NotFound => StatusCode::NOT_FOUND,
            AppError::Multipart(_) => StatusCode::BAD_REQUEST,
            AppError::Processing(_) | AppError::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = Json(json!({ "error": self.to_string() }));
        (status, body).into_response()
    }
}
