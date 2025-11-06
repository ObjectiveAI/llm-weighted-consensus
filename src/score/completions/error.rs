use crate::{chat, error};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("error fetching score model: {0}")]
    FetchModel(error::ResponseError),
    #[error("error fetching score model weights: {0}")]
    FetchModelWeights(error::ResponseError),
    #[error("invalid score model: {0}")]
    InvalidModel(String),
    #[error("expected 2 or more provided choices but got {0}")]
    ExpectedTwoOrMoreChoices(usize),
    #[error("expected a valid response key")]
    InvalidContent,
    #[error(transparent)]
    Chat(#[from] chat::completions::Error),
    #[error("all votes failed, see choices for further details")]
    AllVotesFailed(Option<u16>),
}

impl error::StatusError for Error {
    fn status(&self) -> u16 {
        match self {
            Error::FetchModel(e) => e.status(),
            Error::FetchModelWeights(e) => e.status(),
            Error::InvalidModel(_) => 400,
            Error::ExpectedTwoOrMoreChoices(_) => 400,
            Error::InvalidContent => 500,
            Error::Chat(e) => e.status(),
            Error::AllVotesFailed(Some(code)) => *code,
            Error::AllVotesFailed(None) => 500,
        }
    }

    fn message(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "kind": "score",
            "error": match self {
                Error::FetchModel(e) => e.message(),
                Error::FetchModelWeights(e) => e.message(),
                Error::InvalidModel(e) => Some(serde_json::json!({
                    "kind": "invalid_model",
                    "error": e,
                })),
                Error::ExpectedTwoOrMoreChoices(n) => Some(serde_json::json!({
                    "kind": "expected_two_or_more_choices",
                    "error": format!("expected 2 or more provided choices but got {}", n),
                })),
                Error::InvalidContent => Some(serde_json::json!({
                    "kind": "invalid_content",
                    "error": "expected a valid response key",
                })),
                Error::Chat(e) => e.message(),
                Error::AllVotesFailed(_) => Some(serde_json::json!({
                    "kind": "all_votes_failed",
                    "error": "all votes failed, see choices for further details",
                })),
            }
        }))
    }
}
