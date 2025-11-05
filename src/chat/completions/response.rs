pub mod streaming {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChatCompletionChunk {
        pub id: String,
        pub choices: Vec<Choice>,
        pub created: u64,
        pub model: String,
        pub object: Object,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub service_tier: Option<super::ServiceTier>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub system_fingerprint: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub usage: Option<super::Usage>,

        // openrouter fields
        #[serde(skip_serializing_if = "Option::is_none")]
        pub provider: Option<String>,
    }

    impl ChatCompletionChunk {
        pub fn push(
            &mut self,
            ChatCompletionChunk {
                choices,
                service_tier,
                system_fingerprint,
                usage,
                provider,
                ..
            }: &ChatCompletionChunk,
        ) {
            self.push_choices(choices);
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
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Choice {
        pub delta: Delta,
        pub finish_reason: Option<super::FinishReason>,
        pub index: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logprobs: Option<super::Logprobs>,
    }

    impl Choice {
        pub fn push(
            &mut self,
            Choice {
                delta,
                finish_reason,
                logprobs,
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
        }
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub enum Object {
        #[serde(rename = "chat.completion.chunk")]
        ChatCompletionChunk,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Delta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub refusal: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub role: Option<super::Role>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_calls: Option<Vec<ToolCall>>,

        // openrouter fields
        #[serde(skip_serializing_if = "Option::is_none")]
        pub reasoning: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub images: Option<Vec<super::Image>>,
    }

    impl std::default::Default for Delta {
        fn default() -> Self {
            Self {
                content: None,
                refusal: None,
                role: None,
                tool_calls: None,
                reasoning: None,
                images: None,
            }
        }
    }

    impl Delta {
        // convert tool call arguments to content
        pub fn tool_as_content(&mut self) {
            if let Some(tool_calls) = self.tool_calls.take() {
                for tool_call in tool_calls {
                    if let Some(ToolCallFunction {
                        arguments: Some(arguments),
                        ..
                    }) = tool_call.function
                    {
                        if let Some(content) = &mut self.content {
                            content.push_str(&arguments);
                        } else {
                            self.content = Some(arguments);
                        }
                    }
                }
            }
        }

        pub fn push(
            &mut self,
            Delta {
                content,
                refusal,
                role,
                tool_calls,
                reasoning,
                images,
            }: &Delta,
        ) {
            super::util::push_option_string(&mut self.content, content);
            super::util::push_option_string(&mut self.refusal, refusal);
            if self.role.is_none() {
                self.role = role.clone();
            }
            self.push_tool_calls(tool_calls);
            super::util::push_option_string(&mut self.reasoning, reasoning);
            super::util::push_option_vec(&mut self.images, images);
        }

        fn push_tool_calls(
            &mut self,
            other_tool_calls: &Option<Vec<ToolCall>>,
        ) {
            fn push_tool_call(
                tool_calls: &mut Vec<ToolCall>,
                other: &ToolCall,
            ) {
                fn find_tool_call(
                    tool_calls: &mut Vec<ToolCall>,
                    index: u64,
                ) -> Option<&mut ToolCall> {
                    for tool_call in tool_calls {
                        if tool_call.index == index {
                            return Some(tool_call);
                        }
                    }
                    None
                }
                if let Some(tool_call) = find_tool_call(tool_calls, other.index)
                {
                    tool_call.push(other);
                } else {
                    tool_calls.push(other.clone());
                }
            }
            match (self.tool_calls.as_mut(), other_tool_calls) {
                (Some(self_tool_calls), Some(other_tool_calls)) => {
                    for other_tool_call in other_tool_calls {
                        push_tool_call(self_tool_calls, other_tool_call);
                    }
                }
                (None, Some(other_tool_calls)) => {
                    self.tool_calls = Some(other_tool_calls.clone());
                }
                _ => {}
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ToolCall {
        pub index: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub function: Option<ToolCallFunction>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub r#type: Option<super::ToolCallType>,
    }

    impl ToolCall {
        pub fn push(
            &mut self,
            ToolCall {
                id,
                function,
                r#type,
                ..
            }: &ToolCall,
        ) {
            if self.id.is_none() {
                self.id = id.clone();
            }
            match (&mut self.function, &function) {
                (Some(self_function), Some(other_function)) => {
                    self_function.push(other_function);
                }
                (None, Some(other_function)) => {
                    self.function = Some(other_function.clone());
                }
                _ => {}
            }
            if self.r#type.is_none() {
                self.r#type = r#type.clone();
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ToolCallFunction {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub arguments: Option<String>,
    }

    impl ToolCallFunction {
        pub fn push(&mut self, other: &ToolCallFunction) {
            if self.name.is_none() {
                self.name = other.name.clone();
            }
            match (&mut self.arguments, &other.arguments) {
                (Some(self_arguments), Some(other_arguments)) => {
                    self_arguments.push_str(other_arguments);
                }
                (None, Some(other_arguments)) => {
                    self.arguments = Some(other_arguments.clone());
                }
                _ => {}
            }
        }
    }
}

pub mod unary {
    use serde::{Deserialize, Serialize};
    use std::default::Default;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChatCompletion {
        pub id: String,
        pub choices: Vec<Choice>,
        pub created: u64,
        pub model: String,
        pub object: Object,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub service_tier: Option<super::ServiceTier>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub system_fingerprint: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub usage: Option<super::Usage>,

        // openrouter fields
        #[serde(skip_serializing_if = "Option::is_none")]
        pub provider: Option<String>,
    }

    impl std::default::Default for ChatCompletion {
        fn default() -> Self {
            Self {
                id: String::new(),
                choices: Vec::new(),
                created: 0,
                model: String::new(),
                object: Object::ChatCompletion,
                service_tier: None,
                system_fingerprint: None,
                usage: None,
                provider: None,
            }
        }
    }

    impl From<super::streaming::ChatCompletionChunk> for ChatCompletion {
        fn from(
            super::streaming::ChatCompletionChunk {
                id,
                choices,
                created,
                model,
                service_tier,
                system_fingerprint,
                usage,
                provider,
                ..
            }: super::streaming::ChatCompletionChunk,
        ) -> Self {
            ChatCompletion {
                id,
                choices: choices.into_iter().map(Choice::from).collect(),
                created,
                model,
                object: Object::ChatCompletion,
                service_tier,
                system_fingerprint,
                usage,
                provider,
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Choice {
        pub message: Message,
        pub finish_reason: super::FinishReason,
        pub index: u64,
        pub logprobs: Option<super::Logprobs>,
    }

    impl From<super::streaming::Choice> for Choice {
        fn from(
            super::streaming::Choice {
                delta,
                finish_reason,
                index,
                logprobs,
            }: super::streaming::Choice,
        ) -> Self {
            Choice {
                message: Message::from(delta),
                finish_reason: finish_reason.unwrap_or_default(),
                index,
                logprobs,
            }
        }
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub enum Object {
        #[serde(rename = "chat.completion")]
        ChatCompletion,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Message {
        pub content: Option<String>,
        pub refusal: Option<String>,
        pub role: super::Role,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub annotations: Option<Vec<Annotation>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub audio: Option<Audio>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_calls: Option<Vec<ToolCall>>,

        // openrouter fields
        #[serde(skip_serializing_if = "Option::is_none")]
        pub reasoning: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub images: Option<Vec<super::Image>>,
    }

    impl From<super::streaming::Delta> for Message {
        fn from(
            super::streaming::Delta {
                content,
                refusal,
                role,
                tool_calls,
                reasoning,
                images,
            }: super::streaming::Delta,
        ) -> Self {
            Message {
                content,
                refusal,
                role: role.unwrap_or_default(),
                annotations: None,
                audio: None,
                tool_calls: tool_calls.map(|tool_calls| {
                    tool_calls.into_iter().map(ToolCall::from).collect()
                }),
                reasoning,
                images,
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum Annotation {
        #[serde(rename = "url_citation")]
        UrlCitation { url_citation: AnnotationUrlCitation },
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AnnotationUrlCitation {
        pub end_index: u64,
        pub start_index: u64,
        pub title: String,
        pub url: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Audio {
        pub id: String,
        pub data: String,
        pub expires_at: u64,
        pub transcript: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ToolCall {
        pub id: String,
        pub function: ToolCallFunction,
        pub r#type: super::ToolCallType,
    }

    impl From<super::streaming::ToolCall> for ToolCall {
        fn from(
            super::streaming::ToolCall {
                id,
                function,
                r#type,
                ..
            }: super::streaming::ToolCall,
        ) -> Self {
            ToolCall {
                id: id.unwrap_or_default(),
                function: function
                    .map(ToolCallFunction::from)
                    .unwrap_or_default(),
                r#type: r#type.unwrap_or_default(),
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub struct ToolCallFunction {
        pub name: String,
        pub arguments: String,
    }

    impl From<super::streaming::ToolCallFunction> for ToolCallFunction {
        fn from(
            super::streaming::ToolCallFunction { name, arguments }: super::streaming::ToolCallFunction,
        ) -> Self {
            ToolCallFunction {
                name: name.unwrap_or_default(),
                arguments: arguments.unwrap_or_default(),
            }
        }
    }
}

use serde::{Deserialize, Serialize};
use std::default::Default;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ServiceTier {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "default")]
    Default,
    #[serde(rename = "flex")]
    Flex,
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq,
)]
pub enum FinishReason {
    #[serde(rename = "stop")]
    Stop,
    #[serde(rename = "length")]
    Length,
    #[serde(rename = "tool_calls")]
    ToolCalls,
    #[serde(rename = "content_filter")]
    ContentFilter,

    // custom fields
    #[serde(rename = "error")]
    #[default]
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub completion_tokens: u64,
    pub prompt_tokens: u64,
    pub total_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,

    // openrouter fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<rust_decimal::Decimal>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_details: Option<CostDetails>,
}

impl std::default::Default for Usage {
    fn default() -> Self {
        Self {
            completion_tokens: 0,
            prompt_tokens: 0,
            total_tokens: 0,
            completion_tokens_details: None,
            prompt_tokens_details: None,
            cost: None,
            cost_details: None,
        }
    }
}

impl Usage {
    pub fn push(&mut self, other: &Usage) {
        self.completion_tokens += other.completion_tokens;
        self.prompt_tokens += other.prompt_tokens;
        self.total_tokens += other.total_tokens;
        match (
            &mut self.completion_tokens_details,
            &other.completion_tokens_details,
        ) {
            (Some(self_value), Some(other_value)) => {
                self_value.push(other_value);
            }
            (None, Some(other_value)) => {
                self.completion_tokens_details = Some(other_value.clone());
            }
            _ => {}
        }
        match (
            &mut self.prompt_tokens_details,
            &other.prompt_tokens_details,
        ) {
            (Some(self_value), Some(other_value)) => {
                self_value.push(other_value);
            }
            (None, Some(other_value)) => {
                self.prompt_tokens_details = Some(other_value.clone());
            }
            _ => {}
        }
        util::push_option_decimal(&mut self.cost, &other.cost);
        match (&mut self.cost_details, &other.cost_details) {
            (Some(self_value), Some(other_value)) => {
                self_value.push(other_value);
            }
            (None, Some(other_value)) => {
                self.cost_details = Some(other_value.clone());
            }
            _ => {}
        }
    }

    pub fn is_empty(&self) -> bool {
        self.completion_tokens == 0
            && self.prompt_tokens == 0
            && self.total_tokens == 0
            && self.completion_tokens_details.is_none()
            && self.prompt_tokens_details.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_prediction_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected_prediction_tokens: Option<u64>,
}

impl CompletionTokensDetails {
    pub fn push(&mut self, other: &CompletionTokensDetails) {
        util::push_option_u64(
            &mut self.accepted_prediction_tokens,
            &other.accepted_prediction_tokens,
        );
        util::push_option_u64(&mut self.audio_tokens, &other.audio_tokens);
        util::push_option_u64(
            &mut self.reasoning_tokens,
            &other.reasoning_tokens,
        );
        util::push_option_u64(
            &mut self.rejected_prediction_tokens,
            &other.rejected_prediction_tokens,
        );
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
}

impl PromptTokensDetails {
    pub fn push(&mut self, other: &PromptTokensDetails) {
        util::push_option_u64(&mut self.audio_tokens, &other.audio_tokens);
        util::push_option_u64(&mut self.cached_tokens, &other.cached_tokens);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream_inference_cost: Option<rust_decimal::Decimal>,

    // custom fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream_upstream_inference_cost: Option<rust_decimal::Decimal>,
}

impl CostDetails {
    pub fn push(&mut self, other: &CostDetails) {
        util::push_option_decimal(
            &mut self.upstream_inference_cost,
            &other.upstream_inference_cost,
        );
        util::push_option_decimal(
            &mut self.upstream_upstream_inference_cost,
            &other.upstream_upstream_inference_cost,
        );
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logprobs {
    pub content: Option<Vec<Logprob>>,
    pub refusal: Option<Vec<Logprob>>,
}

impl Logprobs {
    pub fn push(&mut self, other: &Logprobs) {
        match (&mut self.content, &other.content) {
            (Some(self_content), Some(other_content)) => {
                self_content.extend(other_content.clone());
            }
            (None, Some(other_content)) => {
                self.content = Some(other_content.clone());
            }
            _ => {}
        }
        match (&mut self.refusal, &other.refusal) {
            (Some(self_refusal), Some(other_refusal)) => {
                self_refusal.extend(other_refusal.clone());
            }
            (None, Some(other_refusal)) => {
                self.refusal = Some(other_refusal.clone());
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logprob {
    pub token: String,
    pub bytes: Option<Vec<u8>>,
    pub logprob: f64,
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopLogprob {
    pub token: String,
    pub bytes: Option<Vec<u8>>,
    pub logprob: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum Role {
    #[serde(rename = "assistant")]
    #[default]
    Assistant,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum ToolCallType {
    #[serde(rename = "function")]
    #[default]
    Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    pub r#type: ImageType,
    pub image_url: ImageUrl,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ImageType {
    #[serde(rename = "image_url")]
    #[default]
    ImageUrl,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

pub mod util {
    pub fn push_option_u64(
        self_value: &mut Option<u64>,
        other_value: &Option<u64>,
    ) {
        match (self_value.as_mut(), other_value) {
            (Some(self_value), Some(other_value)) => {
                *self_value += other_value;
            }
            (None, Some(other_value)) => {
                *self_value = Some(*other_value);
            }
            _ => {}
        }
    }

    pub fn push_option_decimal(
        self_value: &mut Option<rust_decimal::Decimal>,
        other_value: &Option<rust_decimal::Decimal>,
    ) {
        match (self_value.as_mut(), other_value) {
            (Some(self_value), Some(other_value)) => {
                *self_value += other_value;
            }
            (None, Some(other_value)) => {
                *self_value = Some(*other_value);
            }
            _ => {}
        }
    }

    pub fn push_option_string(
        self_value: &mut Option<String>,
        other_value: &Option<String>,
    ) {
        match (self_value.as_mut(), other_value) {
            (Some(self_value), Some(other_value)) => {
                self_value.push_str(other_value);
            }
            (None, Some(other_value)) => {
                *self_value = Some(other_value.clone());
            }
            _ => {}
        }
    }

    pub fn push_option_vec<T: Clone>(
        self_value: &mut Option<Vec<T>>,
        other_value: &Option<Vec<T>>,
    ) {
        match (self_value.as_mut(), other_value) {
            (Some(self_value), Some(other_value)) => {
                self_value.extend(other_value.clone());
            }
            (None, Some(other_value)) => {
                *self_value = Some(other_value.clone());
            }
            _ => {}
        }
    }
}
