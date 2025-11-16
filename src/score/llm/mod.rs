use crate::chat;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use twox_hash::XxHash3_128;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlmBase {
    // the upstream language model to use
    pub model: String,

    // the scoring weight of this LLM
    #[serde(default)]
    pub weight: Weight,

    // output mode
    #[serde(default)]
    pub output_mode: OutputMode,

    // whether to use synthetic reasoning for non-reasoning LLMs
    // excludes output_mode: instruction
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synthetic_reasoning: Option<bool>,

    // whether to use logprobs
    // changes `vote` to probabilities (if upstream actually provides them)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u64>,

    // messages which will precede the user prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_messages: Option<Vec<chat::completions::request::Message>>,

    // messages which will follow the user prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix_messages: Option<Vec<chat::completions::request::Message>>,

    // openai fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<IndexMap<String, i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<chat::completions::request::Stop>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    // openrouter fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<chat::completions::request::ProviderPreferences>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<chat::completions::request::Reasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_a: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<chat::completions::request::Verbosity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
}

impl LlmBase {
    pub fn prepare(&mut self) {
        fn prepare_f64(value: &mut Option<f64>, default: f64) {
            if let Some(v) = value
                && *v == default
            {
                *value = None;
            }
        }
        fn prepare_u64(value: &mut Option<u64>, default: u64) {
            if let Some(v) = value
                && *v == default
            {
                *value = None;
            }
        }
        fn prepare_logit_bias(logit_bias: &mut Option<IndexMap<String, i64>>) {
            if let Some(lb) = logit_bias
                && lb.is_empty()
            {
                *logit_bias = None;
            }
        }
        fn prepare_verbosity(
            verbosity: &mut Option<chat::completions::request::Verbosity>,
        ) {
            if let Some(v) = verbosity {
                // medium is default
                if matches!(v, chat::completions::request::Verbosity::Medium) {
                    *verbosity = None;
                }
            }
        }
        fn prepare_stop(stop: &mut Option<chat::completions::request::Stop>) {
            if let Some(s) = stop {
                match s {
                    chat::completions::request::Stop::Strings(s) => {
                        if s.is_empty() {
                            *stop = None;
                        } else if s.len() == 1 {
                            *stop =
                                Some(chat::completions::request::Stop::String(
                                    s.pop().unwrap(),
                                ));
                        } else {
                            s.sort();
                        }
                    }
                    _ => {}
                }
            }
        }
        fn prepare_reasoning(
            reasoning: &mut Option<chat::completions::request::Reasoning>,
        ) {
            if let Some(r) = reasoning {
                if r.max_tokens.is_some_and(|mt| mt == 0) {
                    // 0 => None
                    r.max_tokens = None;
                }
                // r.enabled defaults to true or false based on r.effort and r.max_tokens
                // unset it unless it conflicts
                match r.enabled {
                    Some(true)
                        if r.effort.is_some() || r.max_tokens.is_some() =>
                    {
                        r.enabled = None;
                    }
                    Some(false)
                        if r.effort.is_none() && r.max_tokens.is_none() =>
                    {
                        r.enabled = None;
                    }
                    _ => {}
                }
                if r.max_tokens.is_none()
                    && r.enabled.is_none()
                    && r.effort.is_none()
                {
                    *reasoning = None;
                }
            }
        }
        fn prepare_provider(
            provider: &mut Option<
                chat::completions::request::ProviderPreferences,
            >,
        ) {
            if let Some(p) = provider {
                if p.is_empty() {
                    *provider = None;
                } else {
                    if let Some(order) = &mut p.order {
                        if order.is_empty() {
                            p.order = None;
                        }
                    }
                    if let Some(true) = p.allow_fallbacks {
                        p.allow_fallbacks = None;
                    }
                    if let Some(false) = p.require_parameters {
                        p.require_parameters = None;
                    }
                    if let Some(
                            chat::completions::request::ProviderPreferencesDataCollection::Allow
                        ) = p.data_collection
                        {
                            p.data_collection = None;
                        }
                    if let Some(only) = &mut p.only {
                        only.sort();
                        if only.is_empty() {
                            p.only = None;
                        }
                    }
                    if let Some(ignore) = &mut p.ignore {
                        ignore.sort();
                        if ignore.is_empty() {
                            p.ignore = None;
                        }
                    }
                    if let Some(quantizations) = &mut p.quantizations {
                        quantizations.sort();
                        if quantizations.is_empty() {
                            p.quantizations = None;
                        }
                    }
                    if p.is_empty() {
                        *provider = None;
                    }
                }
            }
        }
        fn prepare_top_logprobs(select_top_logprobs: &mut Option<u64>) {
            if let Some(v) = select_top_logprobs
                && *v == 0
            {
                *select_top_logprobs = None;
            }
        }
        fn prepare_models(models: &mut Option<Vec<String>>) {
            if let Some(ms) = models
                && ms.is_empty()
            {
                *models = None;
            }
        }
        fn prepare_synthetic_reasoning(synthetic_reasoning: &mut Option<bool>) {
            if let Some(v) = synthetic_reasoning
                && *v == false
            {
                *synthetic_reasoning = None;
            }
        }
        fn prepare_messages(
            messages: &mut Option<Vec<chat::completions::request::Message>>,
        ) {
            if let Some(msgs) = messages
                && msgs.is_empty()
            {
                *messages = None;
            }
        }
        prepare_synthetic_reasoning(&mut self.synthetic_reasoning);
        prepare_top_logprobs(&mut self.top_logprobs);
        prepare_messages(&mut self.prefix_messages);
        prepare_messages(&mut self.suffix_messages);
        prepare_f64(&mut self.frequency_penalty, 0.0);
        prepare_logit_bias(&mut self.logit_bias);
        prepare_u64(&mut self.max_completion_tokens, 0);
        prepare_f64(&mut self.presence_penalty, 0.0);
        prepare_stop(&mut self.stop);
        prepare_f64(&mut self.temperature, 1.0);
        prepare_f64(&mut self.top_p, 1.0);
        prepare_u64(&mut self.max_tokens, 0);
        prepare_f64(&mut self.min_p, 0.0);
        prepare_provider(&mut self.provider);
        prepare_reasoning(&mut self.reasoning);
        prepare_f64(&mut self.repetition_penalty, 1.0);
        prepare_f64(&mut self.top_a, 0.0);
        prepare_u64(&mut self.top_k, 0);
        prepare_verbosity(&mut self.verbosity);
        prepare_models(&mut self.models);
    }

    pub fn validate(&self, expect: super::WeightType) -> Result<(), String> {
        fn validate_f64(
            value: Option<f64>,
            name: &str,
            min: f64,
            max: f64,
        ) -> Result<(), String> {
            if let Some(v) = value {
                if !v.is_finite() {
                    return Err(format!(
                        "`{}` must be a finite number: `{}`={}",
                        name, name, v,
                    ));
                }
                if v < min || v > max {
                    return Err(format!(
                        "`{}` must be between {} and {}: `{}`={}",
                        name, min, max, name, v,
                    ));
                }
            }
            Ok(())
        }
        fn validate_u64(
            value: Option<u64>,
            name: &str,
            min: u64,
            max: u64,
        ) -> Result<(), String> {
            if let Some(v) = value {
                if v < min || v > max {
                    return Err(format!(
                        "`{}` must be between {} and {}: `{}`={}",
                        name, min, max, name, v
                    ));
                }
            }
            Ok(())
        }
        fn validate_strings(
            value: Option<&[String]>,
            name: &str,
        ) -> Result<(), String> {
            if let Some(value) = value {
                let mut seen = HashSet::with_capacity(value.len());
                for s in value {
                    if s.is_empty() {
                        return Err(format!(
                            "`{}` cannot contain empty strings",
                            name
                        ));
                    } else if !seen.insert(s.as_str()) {
                        return Err(format!(
                            "`{}` cannot contain duplicate strings: `{}`",
                            name, s
                        ));
                    }
                }
            }
            Ok(())
        }
        fn validate_reasoning(
            reasoning: Option<chat::completions::request::Reasoning>,
        ) -> Result<(), String> {
            if let Some(r) = reasoning {
                if r.max_tokens.is_some_and(|mt| mt > i32::MAX as u64) {
                    return Err(format!(
                        "`reasoning.max_tokens` must be at most {}: `reasoning.max_tokens`={}",
                        i32::MAX,
                        r.max_tokens.unwrap()
                    ));
                }
                if r.effort.is_some() && r.max_tokens.is_some() {
                    return Err("`reasoning.max_tokens` and `reasoning.effort` cannot be set at the same time".to_string());
                }
                match (r.enabled, r.max_tokens, r.effort) {
                    (Some(false), Some(_), None) => {
                        return Err("`reasoning.enabled` cannot be false when `reasoning.max_tokens` is set".to_string());
                    }
                    (Some(false), None, Some(_)) => {
                        return Err("`reasoning.enabled` cannot be false when `reasoning.effort` is set".to_string());
                    }
                    _ => {}
                }
            }
            Ok(())
        }
        fn validate_model(model: &str) -> Result<(), String> {
            if model.is_empty() {
                return Err("`model` cannot be empty".to_string());
            }
            Ok(())
        }
        fn validate_models(
            primary_model: &str,
            models: &Option<Vec<String>>,
        ) -> Result<(), String> {
            if let Some(models) = models {
                let mut seen = HashSet::with_capacity(models.len());
                for model in models {
                    if model.is_empty() {
                        return Err(
                            "models cannot contain empty strings".to_string()
                        );
                    } else if model.as_str() == primary_model
                        || !seen.insert(model.as_str())
                    {
                        return Err(format!(
                            "models cannot contain duplicate strings: `models`={}",
                            model
                        ));
                    }
                }
            }
            Ok(())
        }
        fn validate_logit_bias(
            logit_bias: &Option<IndexMap<String, i64>>,
        ) -> Result<(), String> {
            if let Some(lb) = logit_bias {
                for (token, weight) in lb {
                    if token.is_empty() {
                        return Err(
                            "`logit_bias` keys cannot be empty".to_string()
                        );
                    } else if !token.chars().all(|c| c.is_ascii_digit()) {
                        return Err(format!(
                            "`logit_bias` keys must be numeric: `logit_bias`={}",
                            token
                        ));
                    } else if token.chars().next().unwrap() == '0'
                        && token.len() > 1
                    {
                        return Err(format!(
                            "`logit_bias` keys cannot have leading zeroes: `logit_bias`={}",
                            token
                        ));
                    } else if *weight > 100 || *weight < -100 {
                        return Err(format!(
                            "`logit_bias` values must be between -100 and 100: `logit_bias[{}]`={}",
                            token, weight
                        ));
                    }
                }
            }
            Ok(())
        }
        fn validate_stop(
            stop: &Option<chat::completions::request::Stop>,
        ) -> Result<(), String> {
            if let Some(s) = stop {
                match s {
                    chat::completions::request::Stop::Strings(strings) => {
                        validate_strings(Some(strings), "stop")?;
                    }
                    chat::completions::request::Stop::String(string) => {
                        if string.is_empty() {
                            return Err(
                                "`stop` cannot be an empty string".to_string()
                            );
                        }
                    }
                }
            }
            Ok(())
        }
        fn validate_provider(
            provider: &Option<chat::completions::request::ProviderPreferences>,
        ) -> Result<(), String> {
            if let Some(p) = provider {
                if let Some(order) = &p.order {
                    validate_strings(Some(order), "provider.order")?;
                }
                if let Some(only) = &p.only {
                    validate_strings(Some(only), "provider.only")?;
                }
                if let Some(ignore) = &p.ignore {
                    validate_strings(Some(ignore), "provider.ignore")?;
                }
                if let Some(quantizations) = &p.quantizations {
                    validate_strings(
                        Some(quantizations),
                        "provider.quantizations",
                    )?;
                }
                if let Some(sort) = &p.sort {
                    if sort.is_empty() {
                        return Err(
                            "`provider.sort` cannot be empty".to_string()
                        );
                    }
                }
            }
            Ok(())
        }
        fn validate_top_logprobs(
            top_logprobs: &Option<u64>,
        ) -> Result<(), String> {
            if let Some(v) = top_logprobs {
                if *v > 20 {
                    return Err(format!(
                        "`top_logprobs` must be between 0 and 20: `top_logprobs`={}",
                        v
                    ));
                }
            }
            Ok(())
        }
        fn validate_synthetic_reasoning(
            synthetic_reasoning: &Option<bool>,
            output_mode: OutputMode,
        ) -> Result<(), String> {
            if let Some(sr) = synthetic_reasoning {
                if *sr {
                    if matches!(output_mode, OutputMode::Instruction) {
                        return Err(
                            "`synthetic_reasoning` cannot be true when `output_mode` is `instruction`".to_string()
                        );
                    }
                }
            }
            Ok(())
        }
        validate_model(&self.model)?;
        self.weight.validate(expect)?;
        validate_synthetic_reasoning(
            &self.synthetic_reasoning,
            self.output_mode,
        )?;
        validate_top_logprobs(&self.top_logprobs)?;
        validate_f64(self.frequency_penalty, "frequency_penalty", -2.0, 2.0)?;
        validate_logit_bias(&self.logit_bias)?;
        validate_u64(
            self.max_completion_tokens,
            "max_completion_tokens",
            0,
            i32::MAX as u64,
        )?;
        validate_f64(self.presence_penalty, "presence_penalty", -2.0, 2.0)?;
        validate_stop(&self.stop)?;
        validate_f64(self.temperature, "temperature", 0.0, 2.0)?;
        validate_f64(self.top_p, "top_p", 0.0, 1.0)?;
        validate_u64(self.max_tokens, "max_tokens", 0, i32::MAX as u64)?;
        validate_f64(self.min_p, "min_p", 0.0, 1.0)?;
        validate_provider(&self.provider)?;
        validate_reasoning(self.reasoning)?;
        validate_f64(self.repetition_penalty, "repetition_penalty", 0.0, 2.0)?;
        validate_f64(self.top_a, "top_a", 0.0, 1.0)?;
        validate_u64(self.top_k, "top_k", 0, i32::MAX as u64)?;
        validate_models(&self.model, &self.models)?;
        Ok(())
    }

    pub fn id_number(&self) -> u128 {
        let json = serde_json::to_string(self).unwrap();
        let mut hasher = XxHash3_128::with_seed(0);
        hasher.write(json.as_bytes());
        hasher.finish_128()
    }

    pub fn id_string(&self) -> String {
        format!("{:0>22}", base62::encode(self.id_number()))
    }

    pub fn training_table_id_number(&self) -> Option<u128> {
        if !matches!(self.weight.r#type(), super::WeightType::TrainingTable) {
            return None;
        }
        let mut clone = self.clone();
        clone.weight = Weight::default();
        Some(clone.id_number())
    }

    pub fn training_table_id_string(&self) -> Option<String> {
        self.training_table_id_number()
            .map(|id| format!("{:0>22}", base62::encode(id)))
    }

    pub fn multichat_id_number(&self) -> u128 {
        let mut clone = self.clone();
        clone.weight = Weight::default();
        clone.output_mode = OutputMode::default();
        clone.synthetic_reasoning = None;
        clone.top_logprobs = None;
        clone.id_number()
    }

    pub fn multichat_id_string(&self) -> String {
        format!("{:0>22}", base62::encode(self.multichat_id_number()))
    }

    pub fn into_llm(
        self,
        id: String,
        training_table_id: Option<String>,
        multichat_id: String,
        index: usize,
        training_table_index: Option<usize>,
        multichat_index: usize,
        expect: super::WeightType,
    ) -> Result<Llm, String> {
        self.validate(expect)?;
        Ok(Llm {
            base: self,
            id,
            training_table_id,
            multichat_id,
            index,
            training_table_index,
            multichat_index,
        })
    }

    pub fn into_llm_without_indices(
        mut self,
    ) -> Result<LlmWithoutIndices, String> {
        self.prepare();
        self.validate(self.weight.r#type())?;
        let id = self.id_string();
        let training_table_id = self.training_table_id_string();
        let multichat_id = self.multichat_id_string();
        Ok(LlmWithoutIndices {
            base: self,
            id,
            training_table_id,
            multichat_id,
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Weight {
    Static(WeightStatic),
    TrainingTable(WeightTrainingTable),
}

// NEVER change this implementation
impl std::default::Default for Weight {
    fn default() -> Self {
        Weight::Static(WeightStatic {
            r#type: super::WeightStaticType::Static,
            weight: rust_decimal::dec!(1.0),
        })
    }
}

impl Weight {
    pub fn r#type(&self) -> super::WeightType {
        match self {
            Weight::Static(_) => super::WeightType::Static,
            Weight::TrainingTable(_) => super::WeightType::TrainingTable,
        }
    }

    pub fn validate(&self, expect: super::WeightType) -> Result<(), String> {
        match (self, expect) {
            (Weight::Static(w), super::WeightType::Static) => w.validate(),
            (Weight::TrainingTable(w), super::WeightType::TrainingTable) => {
                w.validate()
            }
            _ => Err(format!(
                "expected weight of type `{}`, found `{}`",
                expect,
                self.r#type()
            )),
        }
    }

    pub fn weight_static(&self) -> Option<WeightStatic> {
        match self {
            Weight::Static(w) => Some(*w),
            _ => None,
        }
    }

    pub fn weight_training_table(&self) -> Option<WeightTrainingTable> {
        match self {
            Weight::TrainingTable(w) => Some(*w),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct WeightStatic {
    pub r#type: super::WeightStaticType,
    pub weight: rust_decimal::Decimal,
}

impl WeightStatic {
    pub fn validate(&self) -> Result<(), String> {
        if self.weight <= rust_decimal::Decimal::ZERO {
            Err(format!(
                "`weight` must be a normal positive number: `weight`={}",
                self.weight
            ))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct WeightTrainingTable {
    pub r#type: super::WeightTrainingTableType,
    pub base_weight: rust_decimal::Decimal,
    pub min_weight: rust_decimal::Decimal,
    pub max_weight: rust_decimal::Decimal,
}

impl WeightTrainingTable {
    pub fn validate(&self) -> Result<(), String> {
        if self.base_weight < self.min_weight
            || self.base_weight > self.max_weight
            || self.min_weight > self.max_weight
            || self.base_weight <= rust_decimal::Decimal::ZERO
            || self.min_weight <= rust_decimal::Decimal::ZERO
            || self.max_weight <= rust_decimal::Decimal::ZERO
        {
            Err(format!(
                "LLM must have normal positive base, min, and max weights for training table weights mode: `base_weight={}`, `min_weight={}`, `max_weight={}`",
                self.base_weight, self.min_weight, self.max_weight
            ))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputMode {
    Instruction, // instructed to output a key
    JsonSchema, // response format json schema containing an enum of possible keys
    ToolCall, // forced tool with argument schema containing an enum of possible keys
}

impl std::default::Default for OutputMode {
    fn default() -> Self {
        OutputMode::Instruction
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlmWithoutIndices {
    // a hash of the LLM configuration
    pub id: String,

    // a hash of the LLM configuration, excluding weight, output_mode, synthetic_reasoning, and top_logprobs
    pub multichat_id: String,

    // a hash of the LLM configuration, excluding weight
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_table_id: Option<String>,

    #[serde(flatten)]
    pub base: LlmBase,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Llm {
    // a hash of the LLM configuration
    pub id: String,

    // the index of the LLM in the Score Model
    pub index: usize,

    // a hash of the LLM configuration, excluding weight, output_mode, synthetic_reasoning, and top_logprobs
    pub multichat_id: String,

    // the index of the LLM in the MultiChat Model
    pub multichat_index: usize,

    // a hash of the LLM configuration, excluding weight
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_table_id: Option<String>,

    // the index of the LLM in the training table
    // same for every LLM with the same training_table_id
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_table_index: Option<usize>,

    #[serde(flatten)]
    pub base: LlmBase,
}
