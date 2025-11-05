use crate::chat;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateEmbeddingResponse {
    pub data: Vec<Embedding>,
    pub model: String,
    pub object: CreateEmbeddingResponseObject,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<chat::completions::response::Usage>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CreateEmbeddingResponseObject {
    #[serde(rename = "list")]
    List,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub embedding: Vec<f64>,
    pub index: u64,
    pub object: EmbeddingObject,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EmbeddingObject {
    #[serde(rename = "embedding")]
    Embedding,
}
