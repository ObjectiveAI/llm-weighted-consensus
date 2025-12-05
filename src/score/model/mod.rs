mod fetcher;

pub use fetcher::*;

use crate::chat;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use twox_hash::XxHash3_128;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelBase {
    pub llms: Vec<super::llm::LlmBase>,
    #[serde(default)]
    pub weight: Weight,
}

impl ModelBase {
    fn prepare(&mut self) {
        self.weight.prepare();
        self.llms.iter_mut().for_each(super::llm::LlmBase::prepare);
    }

    pub fn validate_llms_len(&self) -> Result<(), String> {
        if self.llms.len() < 1 {
            return Err("query model must have at least 1 llm".to_string());
        }
        if self.llms.len() > 128 {
            return Err(format!(
                "query model must have at most {} llms: llms_len={}",
                128,
                self.llms.len()
            ));
        }
        Ok(())
    }

    pub fn into_model_validate(mut self) -> Result<Model, String> {
        self.prepare();
        self.validate_llms_len()?;
        self.weight.validate()?;

        // convert & validate llms
        let mut llms = Vec::with_capacity(self.llms.len());

        // collect unique training table IDs
        let mut training_table_ids = match self.weight.r#type() {
            super::WeightType::TrainingTable => {
                Some(Vec::with_capacity(self.llms.len()))
            }
            _ => None,
        };

        // collect all multichat IDs
        let mut multichat_ids = Vec::with_capacity(self.llms.len());

        for llm in self.llms {
            // compute IDs
            let id = llm.id_string();
            let training_table_id = llm.training_table_id_string();
            let multichat_id = llm.multichat_id_string();

            // append training table ID, unique oneof each
            if let Some(training_table_ids) = &mut training_table_ids
                && let Some(training_table_id) = &training_table_id
            {
                if training_table_ids.len() == 0
                    || !training_table_ids.contains(training_table_id)
                {
                    training_table_ids.push(training_table_id.clone());
                }
            }

            // append multichat ID
            multichat_ids.push(multichat_id.clone());

            // append LLM
            llms.push(llm.into_llm(
                id,
                training_table_id,
                multichat_id,
                0,
                None,
                usize::MAX,
                self.weight.r#type(),
            )?);
        }

        // sort by IDs
        // this ensures the same order is always used for the same LLMs
        llms.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        if let Some(training_table_ids) = &mut training_table_ids {
            training_table_ids.sort_unstable_by(|a, b| a.cmp(b));
        }
        multichat_ids.sort_unstable_by(|a, b| a.cmp(b));

        // prepare ID hasher
        let mut hasher = XxHash3_128::with_seed(0);
        let weight_json = serde_json::to_string(&self.weight).unwrap();
        hasher.write(weight_json.as_bytes());

        // prepare training table ID hasher
        let mut training_table_hasher = if training_table_ids.is_some() {
            let mut training_table_hasher = XxHash3_128::with_seed(0);
            training_table_hasher.write(
                serde_json::to_string(
                    &self.weight.weight_training_table().unwrap().embeddings,
                )
                .unwrap()
                .as_bytes(),
            );
            Some(training_table_hasher)
        } else {
            None
        };
        let mut multichat_hasher = XxHash3_128::with_seed(0);
        let mut multichat_seen = HashMap::with_capacity(multichat_ids.len());

        // iterate over LLMs to:
        // - set index
        // - set training_table_index
        // - hash ID
        // - hash training table ID
        let mut i = 0;
        for super::llm::Llm {
            id,
            training_table_id,
            multichat_id,
            index,
            training_table_index,
            multichat_index,
            ..
        } in &mut llms
        {
            // hash ID and set index
            hasher.write(id.as_bytes());
            *index = i;
            i += 1;

            // hash training table ID and set training table index
            if let Some(training_table_hasher) = &mut training_table_hasher {
                let training_table_id = training_table_id.as_deref().unwrap();
                training_table_hasher.write(training_table_id.as_bytes());
                *training_table_index = Some(
                    training_table_ids
                        .as_ref()
                        .unwrap()
                        .iter()
                        .position(|n| n == training_table_id)
                        .unwrap(),
                );
            }

            // hash multichat ID and set multichat index
            multichat_seen
                .entry(multichat_id.clone())
                .and_modify(|e| *e += 1)
                .or_insert(1);
            multichat_hasher.write(multichat_id.as_bytes());
            *multichat_index = multichat_ids
                .iter()
                .position(|n| n == multichat_id)
                .unwrap();
            *multichat_index += multichat_seen[multichat_id.as_str()] - 1;
        }

        // iterate over multichat IDs to hash them and set multichat index for LLMs
        for (multichat_index, multichat_id) in
            multichat_ids.into_iter().enumerate()
        {
            multichat_hasher.write(multichat_id.as_bytes());
            for llm in &mut llms {
                if llm.multichat_id == multichat_id
                    && llm.multichat_index == usize::MAX
                {
                    llm.multichat_index = multichat_index;
                }
            }
        }

        // finalize the IDs
        let id = format!("{:0>22}", base62::encode(hasher.finish_128()));
        let training_table_id = match training_table_hasher {
            Some(hasher) => {
                Some(format!("{:0>22}", base62::encode(hasher.finish_128())))
            }
            None => None,
        };
        let multichat_id =
            format!("{:0>22}", base62::encode(multichat_hasher.finish_128()));

        // finalize the conversion
        Ok(Model {
            id,
            multichat_id,
            training_table_id,
            llms,
            weight: self.weight,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub multichat_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_table_id: Option<String>,
    pub llms: Vec<super::llm::Llm>,
    #[serde(default)]
    pub weight: Weight,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Weight {
    Static(WeightStatic),
    TrainingTable(WeightTrainingTable),
}

impl std::default::Default for Weight {
    fn default() -> Self {
        Weight::Static(WeightStatic {
            r#type: super::WeightStaticType::Static,
        })
    }
}

impl Weight {
    pub fn weight_static(&self) -> Option<WeightStatic> {
        match self {
            Weight::Static(w) => Some(*w),
            _ => None,
        }
    }

    pub fn weight_training_table(&self) -> Option<&WeightTrainingTable> {
        match self {
            Weight::Static(_) => None,
            Weight::TrainingTable(w) => Some(w),
        }
    }

    pub fn r#type(&self) -> super::WeightType {
        match self {
            Weight::Static(_) => super::WeightType::Static,
            Weight::TrainingTable(_) => super::WeightType::TrainingTable,
        }
    }

    pub fn prepare(&mut self) {
        match self {
            Weight::Static(w) => w.prepare(),
            Weight::TrainingTable(w) => w.prepare(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        match self {
            Weight::Static(w) => w.validate(),
            Weight::TrainingTable(w) => w.validate(),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct WeightStatic {
    pub r#type: super::WeightStaticType,
}

impl WeightStatic {
    pub fn prepare(&mut self) {}

    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightTrainingTable {
    pub r#type: super::WeightTrainingTableType,
    pub embeddings: WeightTrainingTableEmbeddings,
    pub top: usize,
}

impl WeightTrainingTable {
    pub fn prepare(&mut self) {
        self.embeddings.prepare();
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.top < 1 {
            Err(format!(
                "training table weight `top` must be at least 1: `top`={}",
                self.top
            ))
        } else if self.top > i32::MAX as usize {
            Err(format!(
                "training table weight `top` must be at most {}: `top`={}",
                i32::MAX as usize,
                self.top
            ))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightTrainingTableEmbeddings {
    pub model: String,
    pub max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<chat::completions::request::ProviderPreferences>,
}

impl WeightTrainingTableEmbeddings {
    pub fn prepare(&mut self) {
        if let Some(p) = &mut self.provider {
            if p.is_empty() {
                self.provider = None;
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
                    self.provider = None;
                }
            }
        }
    }

    pub fn validate(&self) -> Result<(), String> {
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
        if self.model.is_empty() {
            return Err("`embeddings.model` cannot be empty".to_string());
        }
        if self.max_tokens > i32::MAX as usize {
            return Err(format!(
                "`embeddings.max_tokens` must be at most {}: got {}",
                i32::MAX as usize,
                self.max_tokens
            ));
        }
        validate_provider(&self.provider)?;
        Ok(())
    }
}
