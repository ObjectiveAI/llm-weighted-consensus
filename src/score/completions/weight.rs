use crate::{embeddings, error, score};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Data {
    Static(StaticData),
    TrainingTable(TrainingTableData),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StaticData;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingTableData {
    pub embeddings_response: embeddings::response::CreateEmbeddingResponse,
}

#[derive(Debug, Clone)]
pub struct Fetchers<CTX, STATIC, TRAININGTABLE> {
    pub r#static: STATIC,
    pub training_table: TRAININGTABLE,
    _ctx: std::marker::PhantomData<CTX>,
}

impl<CTX, STATIC, TRAININGTABLE> Fetchers<CTX, STATIC, TRAININGTABLE> {
    pub fn new(
        static_fetcher: STATIC,
        training_table_fetcher: TRAININGTABLE,
    ) -> Self {
        Self {
            r#static: static_fetcher,
            training_table: training_table_fetcher,
            _ctx: std::marker::PhantomData,
        }
    }
}

impl<CTX, STATIC, TRAININGTABLE> Fetchers<CTX, STATIC, TRAININGTABLE>
where
    STATIC: Fetcher<CTX, StaticData>,
    TRAININGTABLE: Fetcher<CTX, TrainingTableData>,
{
    pub async fn fetch(
        &self,
        ctx: CTX,
        request: Arc<super::request::ChatCompletionCreateParams>,
        model: score::model::Model,
    ) -> Result<(Vec<f64>, Data), error::ResponseError> {
        match model.weight.r#type() {
            score::WeightType::Static => self
                .r#static
                .fetch(ctx, request, model)
                .await
                .map(|(weights, data)| (weights, Data::Static(data))),
            score::WeightType::TrainingTable => self
                .training_table
                .fetch(ctx, request, model)
                .await
                .map(|(weights, data)| (weights, Data::TrainingTable(data))),
        }
    }
}

#[async_trait::async_trait]
pub trait Fetcher<CTX, T> {
    async fn fetch(
        &self,
        ctx: CTX,
        request: Arc<super::request::ChatCompletionCreateParams>,
        model: score::model::Model,
    ) -> Result<(Vec<f64>, T), error::ResponseError>;
}

#[derive(Debug, Clone, Copy)]
pub struct StaticFetcher;

#[async_trait::async_trait]
impl<CTX: Send + Sync + 'static> Fetcher<CTX, StaticData> for StaticFetcher {
    async fn fetch(
        &self,
        _ctx: CTX,
        _request: Arc<super::request::ChatCompletionCreateParams>,
        model: score::model::Model,
    ) -> Result<(Vec<f64>, StaticData), error::ResponseError> {
        Ok((
            model
                .llms
                .iter()
                .map(|llm| llm.base.weight.weight_static().unwrap().weight)
                .collect(),
            StaticData,
        ))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UnimplementedTrainingTableFetcher;

#[async_trait::async_trait]
impl<CTX: Send + Sync + 'static> Fetcher<CTX, TrainingTableData>
    for UnimplementedTrainingTableFetcher
{
    async fn fetch(
        &self,
        _ctx: CTX,
        _request: Arc<super::request::ChatCompletionCreateParams>,
        _model: score::model::Model,
    ) -> Result<(Vec<f64>, TrainingTableData), error::ResponseError> {
        unimplemented!()
    }
}
