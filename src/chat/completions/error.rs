use crate::error;
use serde::{Deserialize, Serialize};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    ReqwestError(#[from] reqwest::Error),
    #[error(transparent)]
    OpenRouterProviderError(#[from] OpenRouterProviderError),
    #[error("received an empty stream")]
    EmptyStream,
    #[error(transparent)]
    DeserializationError(#[from] serde_path_to_error::Error<serde_json::Error>),
    #[error("received bad status code: {code}, body: {body}")]
    BadStatus {
        code: reqwest::StatusCode,
        body: serde_json::Value,
    },
    #[error("error fetching stream: {0}")]
    StreamError(#[from] reqwest_eventsource::Error),
    #[error("error fetching stream: timeout")]
    StreamTimeout,
    #[error(transparent)]
    CtxError(error::ResponseError),
    #[error(transparent)]
    CompletionsArchiveError(error::ResponseError),
    #[error("invalid choice_index for completion {0}: {1}")]
    InvalidCompletionChoiceIndex(String, u64),
}

impl error::StatusError for Error {
    fn status(&self) -> u16 {
        match self {
            Error::ReqwestError(e) => {
                e.status().map(|s| s.as_u16()).unwrap_or(500)
            }
            Error::OpenRouterProviderError(e) => e.status(),
            Error::EmptyStream => 500,
            Error::DeserializationError(_) => 500,
            Error::BadStatus { code, .. } => code.as_u16(),
            Error::StreamError(reqwest_eventsource::Error::Transport(e)) => {
                e.status().map(|s| s.as_u16()).unwrap_or(500)
            }
            Error::StreamError(
                reqwest_eventsource::Error::InvalidStatusCode(code, _),
            ) => code.as_u16(),
            Error::StreamError(_) => 500,
            Error::StreamTimeout => 500,
            Error::CtxError(e) => e.status(),
            Error::CompletionsArchiveError(e) => e.status(),
            Error::InvalidCompletionChoiceIndex(_, _) => 400,
        }
    }

    fn message(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "kind": "chat",
            "error": match self {
                Error::ReqwestError(e) => serde_json::json!({
                    "kind": "reqwest",
                    "error": e.to_string(),
                }),
                Error::OpenRouterProviderError(e) => e.message().unwrap_or_default(),
                Error::EmptyStream => serde_json::json!({
                    "kind": "empty_stream",
                    "error": "received an empty stream",
                }),
                Error::DeserializationError(e) => serde_json::json!({
                    "kind": "deserialization",
                    "error": e.to_string(),
                }),
                Error::BadStatus { body, .. } => serde_json::json!({
                    "kind": "bad_status",
                    "error": body,
                }),
                Error::StreamError(e) => serde_json::json!({
                    "kind": "stream_error",
                    "error": e.to_string(),
                }),
                Error::StreamTimeout => serde_json::json!({
                    "kind": "stream_timeout",
                    "error": "error fetching stream: timeout",
                }),
                Error::CtxError(e) => e.message().unwrap_or_else(|| {
                    serde_json::Value::String("ctx error".to_string())
                }),
                Error::CompletionsArchiveError(e) => e.message().unwrap_or_else(|| {
                    serde_json::Value::String("completions archive error".to_string())
                }),
                Error::InvalidCompletionChoiceIndex(id, index) => serde_json::json!({
                    "kind": "invalid_completion_choice_index",
                    "error": format!("invalid choice_index for completion {}: {}", id, index),
                }),
            }
        }))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[error("{}", &serde_json::to_string(self).unwrap_or_default())]
pub struct OpenRouterProviderError {
    pub error: OpenRouterProviderErrorInner,
    pub user_id: Option<String>,
}

impl error::StatusError for OpenRouterProviderError {
    fn status(&self) -> u16 {
        self.error.status()
    }

    fn message(&self) -> Option<serde_json::Value> {
        self.error.message()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[error("{}", &serde_json::to_string(self).unwrap_or_default())]
pub struct OpenRouterProviderErrorInner {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl error::StatusError for OpenRouterProviderErrorInner {
    fn status(&self) -> u16 {
        self.code
            .unwrap_or(reqwest::StatusCode::INTERNAL_SERVER_ERROR.as_u16())
    }

    fn message(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "kind": "provider",
            "message": self.message,
            "metadata": self.metadata,
        }))
    }
}
