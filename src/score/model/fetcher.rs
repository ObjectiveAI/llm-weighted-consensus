use crate::error;

#[async_trait::async_trait]
pub trait Fetcher<CTX> {
    async fn fetch(
        &self,
        ctx: CTX,
        model_id: &str,
    ) -> Result<super::Model, error::ResponseError>;
}

pub struct UnimplementedFetcher;

#[async_trait::async_trait]
impl<CTX: Send + Sync + 'static> Fetcher<CTX> for UnimplementedFetcher {
    async fn fetch(
        &self,
        _ctx: CTX,
        _model_id: &str,
    ) -> Result<super::Model, error::ResponseError> {
        unimplemented!()
    }
}
