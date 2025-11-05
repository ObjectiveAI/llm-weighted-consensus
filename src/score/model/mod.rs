mod fetcher;

pub use fetcher::*;

use serde::{Deserialize, Serialize};
use twox_hash::XxHash3_128;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelBase {
    pub llms: Vec<super::llm::LlmBase>,
    pub weight: Weight,
}

impl ModelBase {
    fn prepare(&mut self) {
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
        let mut training_table_ids = match self.weight.r#type() {
            super::WeightType::TrainingTable => {
                Some(Vec::with_capacity(self.llms.len()))
            }
            _ => None,
        };
        for llm in self.llms {
            let id = llm.id_string();
            let training_table_id = llm.training_table_id_string();
            if let Some(training_table_ids) = &mut training_table_ids
                && let Some(training_table_id) = &training_table_id
            {
                if training_table_ids.len() == 0
                    || !training_table_ids.contains(training_table_id)
                {
                    training_table_ids.push(training_table_id.clone());
                }
            }
            llms.push(llm.into_llm(
                id,
                training_table_id,
                0,
                None,
                self.weight.r#type(),
            )?);
        }

        // sort the models by their names
        // this ensures the same order is always used for the same models
        llms.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        if let Some(training_table_ids) = &mut training_table_ids {
            training_table_ids.sort_unstable_by(|a, b| a.cmp(b));
        }

        // compute IDs and fix indices
        let mut i = 0;
        let mut hasher = XxHash3_128::with_seed(0);
        let mut training_table_hasher = if training_table_ids.is_some() {
            Some(XxHash3_128::with_seed(0))
        } else {
            None
        };
        let weight_json = serde_json::to_string(&self.weight).unwrap();
        hasher.write(weight_json.as_bytes());
        if let Some(training_table_hasher) = &mut training_table_hasher {
            training_table_hasher.write(
                self.weight
                    .weight_training_table()
                    .unwrap()
                    .embeddings_model
                    .to_string()
                    .as_bytes(),
            );
        }
        for super::llm::Llm {
            id,
            training_table_id,
            index,
            training_table_index,
            ..
        } in &mut llms
        {
            hasher.write(id.as_bytes());
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
            *index = i;
            i += 1;
        }
        let id = format!("{:0>22}", base62::encode(hasher.finish_128()));
        let training_table_id = match training_table_hasher {
            Some(hasher) => {
                Some(format!("{:0>22}", base62::encode(hasher.finish_128())))
            }
            None => None,
        };

        // finalize the conversion
        Ok(Model {
            id,
            training_table_id,
            llms,
            weight: self.weight,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_table_id: Option<String>,
    pub llms: Vec<super::llm::Llm>,
    pub weight: Weight,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Weight {
    Static(WeightStatic),
    TrainingTable(WeightTrainingTable),
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
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightTrainingTable {
    pub r#type: super::WeightTrainingTableType,
    pub embeddings_model: String,
    pub top: usize,
}

impl WeightTrainingTable {
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
