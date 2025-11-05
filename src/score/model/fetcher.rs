use crate::error;

#[async_trait::async_trait]
pub trait Fetcher {
    async fn fetch(
        &self,
        model_id: &str,
    ) -> Result<super::Model, error::ResponseError>;
}

pub struct UnimplementedFetcher;

#[async_trait::async_trait]
impl Fetcher for UnimplementedFetcher {
    async fn fetch(
        &self,
        _model_id: &str,
    ) -> Result<super::Model, error::ResponseError> {
        unimplemented!()
    }
}
