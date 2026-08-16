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
    #[error("invalid PP-OCRv5 configuration: {0}")]
    OcrConfig(String),
    #[error("PP-OCRv5 request failed: {0}")]
    OcrTransport(#[source] reqwest::Error),
    #[error("PP-OCRv5 rejected the request with code {0}")]
    OcrRejected(u16),
    #[error("invalid PP-OCRv5 response: {0}")]
    OcrProtocol(&'static str),
    #[error("PP-OCRv5 request waited too long for capacity")]
    OcrQueueTimeout,
    #[error("PP-OCRv5 client is unavailable")]
    OcrUnavailable,
    #[error("image is too large to encode for PP-OCRv5")]
    OcrInputTooLarge,
    #[error("blocking image task failed: {0}")]
    BlockingTask(#[source] tokio::task::JoinError),
    #[error("OCR recognition worker panicked")]
    OcrWorkerPanicked,
    #[error("ONNX Runtime error: {0}")]
    Onnx(#[source] ort::Error),
}

impl From<opencv::Error> for AppError {
    fn from(e: opencv::Error) -> Self {
        AppError::Processing(e)
    }
}

impl<R> From<ort::Error<R>> for AppError {
    fn from(e: ort::Error<R>) -> Self {
        AppError::Onnx(ort::Error::new_with_code(e.code(), e.message().to_owned()))
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match &self {
            AppError::MissingImage
            | AppError::EmptyImage
            | AppError::Decode(_)
            | AppError::InvalidCrop(_) => StatusCode::BAD_REQUEST,
            AppError::NotFound => StatusCode::NOT_FOUND,
            AppError::Multipart(_) => StatusCode::BAD_REQUEST,
            AppError::OcrQueueTimeout => StatusCode::SERVICE_UNAVAILABLE,
            AppError::OcrTransport(error) if error.is_timeout() => StatusCode::GATEWAY_TIMEOUT,
            AppError::OcrTransport(_) | AppError::OcrRejected(_) | AppError::OcrProtocol(_) => {
                StatusCode::BAD_GATEWAY
            }
            AppError::Processing(_)
            | AppError::Io(_)
            | AppError::OcrConfig(_)
            | AppError::OcrUnavailable
            | AppError::OcrInputTooLarge
            | AppError::BlockingTask(_)
            | AppError::OcrWorkerPanicked
            | AppError::Onnx(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let message = match &self {
            AppError::MissingImage => "no image field found in multipart body",
            AppError::EmptyImage => "uploaded image is empty",
            AppError::NotFound => "scan id not found",
            AppError::Decode(_) => "failed to decode image",
            AppError::InvalidCrop(message) => message,
            AppError::Multipart(_) => "invalid multipart upload",
            AppError::OcrQueueTimeout => "OCR service is busy",
            AppError::OcrTransport(error) if error.is_timeout() => "OCR service timed out",
            AppError::OcrTransport(_)
            | AppError::OcrRejected(_)
            | AppError::OcrProtocol(_)
            | AppError::OcrUnavailable => "OCR service is unavailable",
            AppError::Processing(_)
            | AppError::Io(_)
            | AppError::OcrConfig(_)
            | AppError::OcrInputTooLarge
            | AppError::BlockingTask(_)
            | AppError::OcrWorkerPanicked
            | AppError::Onnx(_) => "internal server error",
        };
        if status.is_server_error() {
            tracing::error!(error = ?self, "request failed");
        }
        let body = Json(json!({ "error": message }));
        (status, body).into_response()
    }
}
