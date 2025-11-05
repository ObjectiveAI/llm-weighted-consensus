pub mod completions;
pub mod llm;
pub mod model;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightType {
    Static,
    TrainingTable,
}

impl std::fmt::Display for WeightType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            WeightType::Static => write!(f, "static"),
            WeightType::TrainingTable => write!(f, "training_table"),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightStaticType {
    Static,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightTrainingTableType {
    TrainingTable,
}
