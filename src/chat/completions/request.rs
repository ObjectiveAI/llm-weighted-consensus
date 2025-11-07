use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionCreateParams {
    pub messages: Vec<Message>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<IndexMap<String, i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<Prediction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Stop>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<WebSearchOptions>,

    // openrouter fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plugins: Option<Vec<Plugin>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderPreferences>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_a: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<Verbosity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ServiceTier {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "default")]
    Default,
    #[serde(rename = "flex")]
    Flex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    String(String),
    Strings(Vec<String>),
}

impl Stop {
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            Stop::String(s) => vec![s.clone()],
            Stop::Strings(v) => v.clone(),
        }
    }
}

// #[derive(Debug, Clone, Serialize, Deserialize)]
// pub struct Audio {
//     pub format: AudioFormat,
//     pub voice: String,
// }

// #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
// pub enum AudioFormat {
//     #[serde(rename = "wav")]
//     Wav,
//     #[serde(rename = "aac")]
//     Aac,
//     #[serde(rename = "mp3")]
//     Mp3,
//     #[serde(rename = "flac")]
//     Flac,
//     #[serde(rename = "opus")]
//     Opus,
//     #[serde(rename = "pcm16")]
//     Pcm16,
// }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub content: PredictionContent,
    pub r#type: PredictionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PredictionContent {
    Text(String),
    Parts(Vec<PredictionContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionContentPart {
    pub text: String,
    pub r#type: PredictionContentPartType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PredictionContentPartType {
    #[serde(rename = "text")]
    Text,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PredictionType {
    #[serde(rename = "content")]
    Content,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ReasoningEffort {
    #[serde(rename = "minimal")]
    Minimal,
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "high")]
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: JsonSchema },
}

impl ResponseFormat {
    pub fn is_json(&self) -> bool {
        matches!(
            self,
            ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. }
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct JsonSchema {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolChoice {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "required")]
    Required,
    #[serde(untagged)]
    Function(ToolChoiceFunction),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub r#type: ToolChoiceFunctionType,
    pub function: ToolChoiceFunctionFunction,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ToolChoiceFunctionType {
    #[serde(rename = "function")]
    Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunctionFunction {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<SearchContextSize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<UserLocation>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SearchContextSize {
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "high")]
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserLocation {
    pub approximate: UserLocationApproximate,
    pub r#type: UserLocationType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserLocationApproximate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>, // IANA
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum UserLocationType {
    #[serde(rename = "approximate")]
    Approximate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub function: FunctionDefinition,
    pub r#type: ToolType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ToolType {
    #[serde(rename = "function")]
    Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    #[serde(rename = "developer")]
    Developer(DeveloperMessage),
    #[serde(rename = "system")]
    System(SystemMessage),
    #[serde(rename = "user")]
    User(UserMessage),
    #[serde(rename = "assistant")]
    Assistant(AssistantMessage),
    #[serde(rename = "tool")]
    Tool(ToolMessage),
}

impl Message {
    pub fn write_template_content(&self, s: &mut String) {
        match self {
            Message::Developer(msg) => msg.write_template_content(s),
            Message::System(msg) => msg.write_template_content(s),
            Message::User(msg) => msg.write_template_content(s),
            Message::Assistant(msg) => msg.write_template_content(s),
            Message::Tool(msg) => msg.write_template_content(s),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeveloperMessage {
    pub content: SimpleContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl DeveloperMessage {
    pub fn write_template_content(&self, s: &mut String) {
        s.push_str("developer");
        if let Some(name) = &self.name {
            s.push_str(" (");
            s.push_str(name);
            s.push_str(")");
        }
        s.push_str(": ");
        self.content.write_template_content(s);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMessage {
    pub content: SimpleContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl SystemMessage {
    pub fn write_template_content(&self, s: &mut String) {
        s.push_str("system");
        if let Some(name) = &self.name {
            s.push_str(" (");
            s.push_str(name);
            s.push_str(")");
        }
        s.push_str(": ");
        self.content.write_template_content(s);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMessage {
    pub content: UserContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl UserMessage {
    pub fn write_template_content(&self, s: &mut String) {
        s.push_str("user");
        if let Some(name) = &self.name {
            s.push_str(" (");
            s.push_str(name);
            s.push_str(")");
        }
        s.push_str(": ");
        self.content.write_template_content(s);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMessage {
    pub content: SimpleContent,
    pub tool_call_id: String,
}

impl ToolMessage {
    pub fn write_template_content(&self, s: &mut String) {
        s.push_str("tool (");
        s.push_str(&self.tool_call_id);
        s.push_str("): ");
        self.content.write_template_content(s);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantMessage {
    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub audio: Option<AssistantAudio>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<AssistantContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<AssistantToolCall>>,
}

impl AssistantMessage {
    pub fn write_template_content(&self, s: &mut String) {
        let mut wrote = false;
        if let Some(content) = &self.content {
            self.write_template_content_role(s);
            content.write_template_content(s);
            wrote = true;
        }
        if let Some(refusal) = &self.refusal {
            if wrote {
                s.push_str("\n");
            }
            self.write_template_content_role(s);
            s.push_str(refusal);
            wrote = true;
        }
        if let Some(tool_calls) = &self.tool_calls {
            if wrote {
                s.push_str("\n");
            }
            self.write_template_content_role(s);
            tool_calls
                .iter()
                .for_each(|tool_call| tool_call.write_template_content(s));
        }
    }

    fn write_template_content_role(&self, s: &mut String) {
        s.push_str("assistant");
        if let Some(name) = &self.name {
            s.push_str(" (");
            s.push_str(name);
            s.push_str(")");
        }
        s.push_str(": ");
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SimpleContent {
    Text(String),
    Parts(Vec<SimpleContentPart>),
}

impl SimpleContent {
    pub fn write_template_content(&self, s: &mut String) {
        match self {
            SimpleContent::Text(text) => s.push_str(text),
            SimpleContent::Parts(parts) => {
                parts.iter().for_each(|part| part.write_template_content(s))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleContentPart {
    pub text: String,
    pub r#type: SimpleContentPartType,
}

impl SimpleContentPart {
    pub fn write_template_content(&self, s: &mut String) {
        s.push_str(&self.text);
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SimpleContentPartType {
    #[serde(rename = "text")]
    Text,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl UserContent {
    pub fn write_template_content(&self, s: &mut String) {
        match self {
            UserContent::Text(text) => s.push_str(text),
            UserContent::Parts(parts) => {
                parts.iter().for_each(|part| part.write_template_content(s))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    #[serde(rename = "input_audio")]
    InputAudio { input_audio: InputAudio },
    #[serde(rename = "file")]
    File { file: File },
}

impl ContentPart {
    pub fn write_template_content(&self, s: &mut String) {
        match self {
            ContentPart::Text { text } => s.push_str(text),
            _ => {}
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageUrlDetail>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ImageUrlDetail {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "high")]
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudio {
    pub data: String,
    pub format: InputAudioFormat,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InputAudioFormat {
    #[serde(rename = "wav")]
    Wav,
    #[serde(rename = "mp3")]
    Mp3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct File {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

// #[derive(Debug, Clone, Serialize, Deserialize)]
// pub struct AssistantAudio {
//     pub id: String,
// }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AssistantContent {
    Text(String),
    Parts(Vec<AssistantContentPart>),
}

impl AssistantContent {
    pub fn write_template_content(&self, s: &mut String) {
        match self {
            AssistantContent::Text(text) => s.push_str(text),
            AssistantContent::Parts(parts) => {
                parts.iter().for_each(|part| part.write_template_content(s))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AssistantContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
}

impl AssistantContentPart {
    pub fn write_template_content(&self, s: &mut String) {
        s.push_str(match self {
            AssistantContentPart::Text { text } => text,
            AssistantContentPart::Refusal { refusal } => refusal,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantToolCall {
    pub id: String,
    pub function: AssistantToolCallFunction,
    pub r#type: AssistantToolCallType,
}

impl AssistantToolCall {
    pub fn write_template_content(&self, s: &mut String) {
        s.push_str(&format!(
            "<tool_call>{}</tool_call>",
            serde_json::to_string(self).unwrap()
        ));
    }
}

impl From<super::response::unary::ToolCall> for AssistantToolCall {
    fn from(
        super::response::unary::ToolCall {
            id,
            function:
                super::response::unary::ToolCallFunction { name, arguments },
            ..
        }: super::response::unary::ToolCall,
    ) -> Self {
        AssistantToolCall {
            id,
            function: AssistantToolCallFunction { name, arguments },
            r#type: AssistantToolCallType::Function,
        }
    }
}

impl From<super::response::streaming::ToolCall> for AssistantToolCall {
    fn from(tool_call: super::response::streaming::ToolCall) -> Self {
        super::response::unary::ToolCall::from(tool_call).into()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AssistantToolCallType {
    #[serde(rename = "function")]
    Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderPreferences {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<ProviderPreferencesDataCollection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantizations: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<String>,
}

impl ProviderPreferences {
    pub fn is_empty(&self) -> bool {
        self.order.is_none()
            && self.allow_fallbacks.is_none()
            && self.require_parameters.is_none()
            && self.data_collection.is_none()
            && self.only.is_none()
            && self.ignore.is_none()
            && self.quantizations.is_none()
            && self.sort.is_none()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ProviderPreferencesDataCollection {
    #[serde(rename = "allow")]
    Allow,
    #[serde(rename = "deny")]
    Deny,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plugin {
    pub id: String,
    #[serde(flatten)]
    pub fields: IndexMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Reasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Usage {
    pub include: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Verbosity {
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "high")]
    High,
}
