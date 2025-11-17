use crate::{chat, score};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionCreateParams {
    pub messages: Vec<chat::completions::request::Message>,
    pub model: Model,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<chat::completions::request::ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<chat::completions::request::StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<chat::completions::request::Tool>>, // readonly

    // openrouter fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<chat::completions::request::Usage>,

    // custom fields
    pub choices: Vec<Choice>,
}

impl ChatCompletionCreateParams {
    pub fn template_content(&self) -> String {
        let mut content = String::new();
        let mut first = true;
        for message in &self.messages {
            if !first {
                content.push_str("\n");
            }
            message.write_template_content(&mut content);
            first = false;
        }
        content
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Model {
    Id(String),
    Provided(score::model::ModelBase),
}

impl Model {
    fn unwrap_kind_str(&self) -> &'static str {
        match self {
            Model::Id(_) => "Id",
            Model::Provided(_) => "Provided",
        }
    }

    pub fn unwrap_id(&self) -> &str {
        match self {
            Model::Id(id) => id,
            Model::Provided(_) => panic!(
                "called `Model::unwrap_id` on a `{}` value",
                self.unwrap_kind_str()
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Choice {
    Text(String),
    ChatCompletionMessage(chat::completions::response::unary::Message),
    ChatCompletion {
        r#type: ChatCompletionChoiceType,
        id: String,
        #[serde(default)]
        choice_index: u64,
    },
    ScoreCompletion {
        r#type: ScoreCompletionChoiceType,
        id: String,
        #[serde(default)]
        choice_index: u64,
    },
    MultichatCompletion {
        r#type: ChatCompletionChoiceType,
        id: String,
        #[serde(default)]
        choice_index: u64,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatCompletionChoiceType {
    ChatCompletion,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreCompletionChoiceType {
    ScoreCompletion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultichatCompletionChoiceType {
    MultichatCompletion,
}
