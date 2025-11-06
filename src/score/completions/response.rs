pub mod streaming {
    use crate::chat;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChatCompletionChunk {
        pub id: String,
        pub choices: Vec<Choice>,
        pub created: u64,
        pub model: String,
        pub object: chat::completions::response::streaming::Object,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub usage: Option<chat::completions::response::Usage>,

        // custom fields
        #[serde(skip_serializing_if = "Option::is_none")]
        pub weight_data: Option<super::super::weight::Data>,
    }

    impl ChatCompletionChunk {
        pub fn tool_as_content(&mut self) {
            self.choices.iter_mut().for_each(Choice::tool_as_content);
        }

        pub fn push(
            &mut self,
            ChatCompletionChunk {
                choices,
                usage,
                weight_data,
                ..
            }: &ChatCompletionChunk,
        ) {
            self.push_choices(choices);
            match (&mut self.usage, usage) {
                (Some(self_usage), Some(other_usage)) => {
                    self_usage.push(other_usage);
                }
                (None, Some(other_usage)) => {
                    self.usage = Some(other_usage.clone());
                }
                _ => {}
            }
            if self.weight_data.is_none() {
                self.weight_data = weight_data.clone();
            }
        }

        fn push_choices(&mut self, other_choices: &[Choice]) {
            fn push_choice(choices: &mut Vec<Choice>, other: &Choice) {
                fn find_choice(
                    choices: &mut Vec<Choice>,
                    index: u64,
                ) -> Option<&mut Choice> {
                    for choice in choices {
                        if choice.index == index {
                            return Some(choice);
                        }
                    }
                    None
                }
                if let Some(choice) = find_choice(choices, other.index) {
                    choice.push(other);
                } else {
                    choices.push(other.clone());
                }
            }
            for other_choice in other_choices {
                push_choice(&mut self.choices, other_choice);
            }
        }

        pub fn clone_without_choices(&self, capacity: usize) -> Self {
            Self {
                id: self.id.clone(),
                choices: Vec::with_capacity(capacity),
                created: self.created,
                model: self.model.clone(),
                object: self.object.clone(),
                usage: self.usage.clone(),
                weight_data: self.weight_data.clone(),
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Choice {
        pub delta: Delta,
        pub finish_reason: Option<chat::completions::response::FinishReason>,
        pub index: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logprobs: Option<chat::completions::response::Logprobs>,

        // custom fields
        #[serde(skip_serializing_if = "Option::is_none")]
        pub weight: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub confidence: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<crate::error::ResponseError>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model_index: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub completion_metadata: Option<super::CompletionMetadata>,
    }

    impl Choice {
        pub fn tool_as_content(&mut self) {
            if matches!(
                self.finish_reason,
                Some(chat::completions::response::FinishReason::ToolCalls)
            ) {
                self.finish_reason =
                    Some(chat::completions::response::FinishReason::Stop);
            }
            self.delta.tool_as_content();
        }

        pub fn push(
            &mut self,
            Choice {
                delta,
                finish_reason,
                logprobs,
                weight,
                confidence,
                error,
                model,
                model_index,
                completion_metadata,
                ..
            }: &Choice,
        ) {
            self.delta.push(delta);
            if self.finish_reason.is_none() {
                self.finish_reason = finish_reason.clone();
            }
            match (&mut self.logprobs, logprobs) {
                (Some(self_logprobs), Some(other_logprobs)) => {
                    self_logprobs.push(other_logprobs);
                }
                (None, Some(other_logprobs)) => {
                    self.logprobs = Some(other_logprobs.clone());
                }
                _ => {}
            }
            if self.weight.is_none() {
                self.weight = weight.clone();
            }
            if self.confidence.is_none() {
                self.confidence = confidence.clone();
            }
            if self.error.is_none() {
                self.error = error.clone();
            }
            if self.model.is_none() {
                self.model = model.clone();
            }
            if self.model_index.is_none() {
                self.model_index = model_index.clone();
            }
            match (&mut self.completion_metadata, completion_metadata) {
                (Some(self_metadata), Some(other_metadata)) => {
                    self_metadata.push(other_metadata);
                }
                (None, Some(other_metadata)) => {
                    self.completion_metadata = Some(other_metadata.clone());
                }
                _ => {}
            }
        }

        pub fn has_finish_reason_or_usage(&self) -> bool {
            self.finish_reason.is_some()
                || self
                    .completion_metadata
                    .as_ref()
                    .is_some_and(|m| m.usage.is_some())
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Delta {
        #[serde(flatten)]
        pub inner: chat::completions::response::streaming::Delta,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub vote: Option<Vec<f64>>,
    }

    impl std::default::Default for Delta {
        fn default() -> Self {
            Self {
                inner: chat::completions::response::streaming::Delta::default(),
                vote: None,
            }
        }
    }

    impl Delta {
        pub fn tool_as_content(&mut self) {
            self.inner.tool_as_content();
        }

        pub fn push(&mut self, Delta { inner, vote }: &Delta) {
            self.inner.push(inner);
            if self.vote.is_none() {
                self.vote = vote.clone();
            }
        }
    }
}

pub mod unary {
    use crate::chat;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChatCompletion {
        pub id: String,
        pub choices: Vec<Choice>,
        pub created: u64,
        pub model: String,
        pub object: chat::completions::response::unary::Object,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub usage: Option<chat::completions::response::Usage>,

        // custom fields
        pub weight_data: Option<super::super::weight::Data>,
    }

    impl From<super::streaming::ChatCompletionChunk> for ChatCompletion {
        fn from(
            super::streaming::ChatCompletionChunk {
                id,
                choices,
                created,
                model,
                usage,
                weight_data,
                ..
            }: super::streaming::ChatCompletionChunk,
        ) -> Self {
            ChatCompletion {
                id,
                choices: choices.into_iter().map(Choice::from).collect(),
                created,
                model,
                object:
                    chat::completions::response::unary::Object::ChatCompletion,
                usage,
                weight_data,
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Choice {
        pub message: Message,
        pub finish_reason: chat::completions::response::FinishReason,
        pub index: u64,
        pub logprobs: Option<chat::completions::response::Logprobs>,

        // custom fields
        pub weight: Option<f64>,
        pub confidence: Option<f64>,
        pub error: Option<crate::error::ResponseError>,
        pub model: Option<String>,
        pub model_index: Option<usize>,
        pub completion_metadata: Option<super::CompletionMetadata>,
    }

    impl From<super::streaming::Choice> for Choice {
        fn from(
            super::streaming::Choice {
                delta,
                finish_reason,
                index,
                logprobs,
                weight,
                confidence,
                error,
                model,
                model_index,
                completion_metadata,
            }: super::streaming::Choice,
        ) -> Self {
            Choice {
                message: Message::from(delta),
                finish_reason: finish_reason.unwrap_or_default(),
                index,
                logprobs,
                weight,
                confidence,
                error,
                model,
                model_index,
                completion_metadata,
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Message {
        #[serde(flatten)]
        pub inner: chat::completions::response::unary::Message,
        pub vote: Option<Vec<f64>>,
    }

    impl From<super::streaming::Delta> for Message {
        fn from(
            super::streaming::Delta { inner, vote }: super::streaming::Delta,
        ) -> Self {
            Self {
                inner: chat::completions::response::unary::Message::from(inner),
                vote,
            }
        }
    }
}

use crate::chat;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionMetadata {
    pub id: String,
    pub created: u64,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<chat::completions::response::ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<chat::completions::response::Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

impl std::default::Default for CompletionMetadata {
    fn default() -> Self {
        Self {
            id: String::new(),
            created: 0,
            model: String::new(),
            service_tier: None,
            system_fingerprint: None,
            usage: None,
            provider: None,
        }
    }
}

impl CompletionMetadata {
    pub fn push(
        &mut self,
        CompletionMetadata {
            service_tier,
            system_fingerprint,
            usage,
            provider,
            ..
        }: &CompletionMetadata,
    ) {
        if self.service_tier.is_none() {
            self.service_tier = service_tier.clone();
        }
        if self.system_fingerprint.is_none() {
            self.system_fingerprint = system_fingerprint.clone();
        }
        match (&mut self.usage, usage) {
            (Some(self_usage), Some(other_usage)) => {
                self_usage.push(other_usage);
            }
            (None, Some(other_usage)) => {
                self.usage = Some(other_usage.clone());
            }
            _ => {}
        }
        if self.provider.is_none() {
            self.provider = provider.clone();
        }
    }
}
