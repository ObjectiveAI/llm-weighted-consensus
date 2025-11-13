use crate::{chat, error, score};

#[async_trait::async_trait]
pub trait Fetcher<CTX> {
    async fn fetch_chat_completion(
        &self,
        ctx: CTX,
        id: &str,
    ) -> Result<
        chat::completions::response::unary::ChatCompletion,
        error::ResponseError,
    >;
    async fn fetch_score_completion(
        &self,
        ctx: CTX,
        id: &str,
    ) -> Result<
        score::completions::response::unary::ChatCompletion,
        error::ResponseError,
    >;
}

pub struct UnimplementedFetcher;

#[async_trait::async_trait]
impl<CTX: Send + Sync + 'static> Fetcher<CTX> for UnimplementedFetcher {
    async fn fetch_chat_completion(
        &self,
        _ctx: CTX,
        _id: &str,
    ) -> Result<
        chat::completions::response::unary::ChatCompletion,
        error::ResponseError,
    > {
        unimplemented!()
    }
    async fn fetch_score_completion(
        &self,
        _ctx: CTX,
        _id: &str,
    ) -> Result<
        score::completions::response::unary::ChatCompletion,
        error::ResponseError,
    > {
        unimplemented!()
    }
}
