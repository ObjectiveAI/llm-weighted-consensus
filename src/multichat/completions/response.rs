pub mod streaming {
    use crate::{chat, score};
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
    }

    impl ChatCompletionChunk {
        pub fn push(
            &mut self,
            ChatCompletionChunk { choices, usage, .. }: &ChatCompletionChunk,
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
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Choice {
        pub delta: chat::completions::response::streaming::Delta,
        pub finish_reason: Option<chat::completions::response::FinishReason>,
        pub index: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logprobs: Option<chat::completions::response::Logprobs>,

        // custom fields
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<crate::error::ResponseError>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model_index: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub completion_metadata:
            Option<score::completions::response::CompletionMetadata>,
    }

    impl Choice {
        pub fn push(
            &mut self,
            Choice {
                delta,
                finish_reason,
                logprobs,
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
}

pub mod unary {
    use crate::{chat, score};
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
    }

    impl From<super::streaming::ChatCompletionChunk> for ChatCompletion {
        fn from(
            super::streaming::ChatCompletionChunk {
                id,
                choices,
                created,
                model,
                usage,
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
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Choice {
        pub message: chat::completions::response::unary::Message,
        pub finish_reason: chat::completions::response::FinishReason,
        pub index: u64,
        pub logprobs: Option<chat::completions::response::Logprobs>,

        // custom fields
        pub error: Option<crate::error::ResponseError>,
        pub model: Option<String>,
        pub model_index: Option<usize>,
        pub completion_metadata:
            Option<score::completions::response::CompletionMetadata>,
    }

    impl From<super::streaming::Choice> for Choice {
        fn from(
            super::streaming::Choice {
                delta,
                finish_reason,
                index,
                logprobs,
                error,
                model,
                model_index,
                completion_metadata,
            }: super::streaming::Choice,
        ) -> Self {
            Choice {
                message: chat::completions::response::unary::Message::from(
                    delta,
                ),
                finish_reason: finish_reason.unwrap_or_default(),
                index,
                logprobs,
                error,
                model,
                model_index,
                completion_metadata,
            }
        }
    }
}
