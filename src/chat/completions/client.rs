use crate::{completions_archive, util::StreamOnce};
use backoff::ExponentialBackoff;
use eventsource_stream::Event as MessageEvent;
use futures::{Stream, StreamExt, TryFutureExt, TryStreamExt};
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiBase {
    pub api_base: String, // OpenAI API URL
    pub api_key: String,  // OpenAI API Key
}

#[derive(Debug, Clone)]
struct Attempt<'ab> {
    api_base: &'ab ApiBase,
    model: String,
}

#[async_trait::async_trait]
pub trait CtxHandler<CTX> {
    async fn handle(
        &self,
        ctx: CTX,
        api_bases: Vec<ApiBase>,
    ) -> Result<Vec<ApiBase>, crate::error::ResponseError>;
}

pub struct NoOpCtxHandler<CTX>(std::marker::PhantomData<CTX>);

impl<CTX> NoOpCtxHandler<CTX> {
    pub fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

#[async_trait::async_trait]
impl<CTX> CtxHandler<CTX> for NoOpCtxHandler<CTX>
where
    CTX: Send + Sync + 'static,
{
    async fn handle(
        &self,
        _ctx: CTX,
        api_bases: Vec<ApiBase>,
    ) -> Result<Vec<ApiBase>, crate::error::ResponseError> {
        Ok(api_bases)
    }
}

#[async_trait::async_trait]
pub trait Client<CTX> {
    async fn create_unary(
        self: Arc<Self>,
        ctx: CTX,
        request: super::request::ChatCompletionCreateParams,
    ) -> Result<super::response::unary::ChatCompletion, super::Error>;

    async fn create_streaming(
        self: Arc<Self>,
        ctx: CTX,
        request: super::request::ChatCompletionCreateParams,
    ) -> Result<
        impl Stream<
            Item = Result<
                super::response::streaming::ChatCompletionChunk,
                super::Error,
            >,
        > + Send
        + Unpin
        + 'static,
        super::Error,
    >;
}

#[derive(Debug, Clone)]
pub struct DefaultClient<CTX, HNDLCTX, FCOMPLETIONS> {
    pub http_client: reqwest::Client,
    pub backoff: ExponentialBackoff,
    pub api_bases: Vec<ApiBase>, // try each in order
    pub user_agent: Option<String>, // user-agent header
    pub x_title: Option<String>, // x-title header
    pub referer: Option<String>, // referer and http-referer headers
    pub first_chunk_timeout: Duration, // timeout for first stream chunk
    pub other_chunk_timeout: Duration, // timeout for other stream chunks
    pub ctx_handler: Arc<HNDLCTX>,
    pub completions_archive_fetcher: Arc<FCOMPLETIONS>,
    _ctx: std::marker::PhantomData<CTX>,
}

impl<CTX, HNDLCTX, FCOMPLETIONS> DefaultClient<CTX, HNDLCTX, FCOMPLETIONS> {
    pub fn new(
        http_client: reqwest::Client,
        backoff: ExponentialBackoff,
        api_bases: Vec<ApiBase>,
        user_agent: Option<String>,
        x_title: Option<String>,
        referer: Option<String>,
        first_chunk_timeout: Duration,
        other_chunk_timeout: Duration,
        ctx_handler: Arc<HNDLCTX>,
        completions_archive_fetcher: Arc<FCOMPLETIONS>,
    ) -> Self {
        Self {
            http_client,
            backoff,
            api_bases,
            user_agent,
            x_title,
            referer,
            first_chunk_timeout,
            other_chunk_timeout,
            ctx_handler,
            completions_archive_fetcher,
            _ctx: std::marker::PhantomData,
        }
    }
}

#[async_trait::async_trait]
impl<CTX, HNDLCTX, FCOMPLETIONS> Client<CTX>
    for DefaultClient<CTX, HNDLCTX, FCOMPLETIONS>
where
    CTX: Clone + Send + Sync + 'static,
    HNDLCTX: CtxHandler<CTX> + Send + Sync + 'static,
    FCOMPLETIONS: completions_archive::Fetcher<CTX> + Send + Sync + 'static,
{
    async fn create_unary(
        self: Arc<Self>,
        ctx: CTX,
        request: super::request::ChatCompletionCreateParams,
    ) -> Result<super::response::unary::ChatCompletion, super::Error> {
        let (response, _api_base) =
            self.create_unary_return_api_base(ctx, request).await?;
        Ok(response)
    }

    async fn create_streaming(
        self: Arc<Self>,
        ctx: CTX,
        request: super::request::ChatCompletionCreateParams,
    ) -> Result<
        impl Stream<
            Item = Result<
                super::response::streaming::ChatCompletionChunk,
                super::Error,
            >,
        > + Send
        + Unpin
        + 'static,
        super::Error,
    > {
        let (stream, _api_base) =
            self.create_streaming_return_api_base(ctx, request).await?;
        Ok(stream)
    }
}

impl<CTX, HNDLCTX, FCOMPLETIONS> DefaultClient<CTX, HNDLCTX, FCOMPLETIONS>
where
    CTX: Clone + Send + Sync + 'static,
    HNDLCTX: CtxHandler<CTX> + Send + Sync + 'static,
    FCOMPLETIONS: completions_archive::Fetcher<CTX> + Send + Sync + 'static,
{
    pub async fn create_unary_return_api_base(
        self: Arc<Self>,
        ctx: CTX,
        request: super::request::ChatCompletionCreateParams,
    ) -> Result<(super::response::unary::ChatCompletion, ApiBase), super::Error>
    {
        let mut aggregate: Option<
            super::response::streaming::ChatCompletionChunk,
        > = None;
        let (mut stream, api_base) =
            self.create_streaming_return_api_base(ctx, request).await?;
        while let Some(response) = stream.try_next().await? {
            match aggregate {
                Some(ref mut aggregate) => aggregate.push(&response),
                None => aggregate = Some(response),
            }
        }
        match aggregate {
            Some(response) => Ok((response.into(), api_base)),
            None => Err(super::Error::EmptyStream),
        }
    }

    pub async fn create_streaming_return_api_base(
        self: Arc<Self>,
        ctx: CTX,
        mut request: super::request::ChatCompletionCreateParams,
    ) -> Result<
        (
            impl Stream<
                Item = Result<
                    super::response::streaming::ChatCompletionChunk,
                    super::Error,
                >,
            > + Send
            + Unpin
            + 'static,
            ApiBase,
        ),
        super::Error,
    > {
        // handle ctx and fetch completions
        let (api_bases, completions) = tokio::try_join!(
            self.ctx_handler
                .handle(ctx.clone(), self.api_bases.clone())
                .map_err(super::Error::CtxError),
            fetch_completion_futs_from_messages(
                self.completions_archive_fetcher.clone(),
                ctx.clone(),
                &request.messages,
            )
            .map_err(super::Error::CompletionsArchiveError),
        )?;

        // replace completion messages with assistant messages
        replace_completion_messages_with_assistant_messages(
            &completions,
            &mut request.messages,
        )?;

        // force streaming
        if request.stream.is_none_or(|s| !s) {
            request.stream_options = Some(super::request::StreamOptions {
                include_usage: Some(true),
            });
        }
        request.stream = Some(true);

        // set up attempts - for each model, try each api_base
        let mut attempts = Vec::with_capacity(
            request.models.as_ref().map_or(1, |m| m.len() + 1)
                * api_bases.len(),
        );
        for api_base in &api_bases {
            attempts.push(Attempt {
                api_base,
                model: request.model.clone(),
            });
        }
        if let Some(models) = request.models.take() {
            for model in models {
                for api_base in &self.api_bases {
                    attempts.push(Attempt {
                        api_base,
                        model: model.clone(),
                    });
                }
            }
        }

        // fetch
        let first_chunk_timeout = self.first_chunk_timeout;
        let other_chunk_timeout = self.other_chunk_timeout;
        backoff::future::retry(self.backoff.clone(), || async {
            let mut request = request.clone();
            let mut i = 0;
            loop {
                let Attempt { api_base, model } = attempts[i].clone();
                request.model = model;
                // event source for this fetch
                let event_source =
                    self.create_streaming_event_source(api_base, &request);
                // stream for this fetch (consume event source)
                let mut stream = Self::create_streaming_stream(
                    event_source,
                    first_chunk_timeout,
                    other_chunk_timeout,
                )
                .boxed();
                i += 1;
                // check first chunk
                match stream.next().await {
                    Some(Ok(response)) => {
                        // first chunk is good, return stream with this chunk prepended
                        break Ok((
                            StreamOnce::new(Ok(response)).chain(stream),
                            api_base.clone(),
                        ));
                    }
                    Some(Err(e)) if i == attempts.len() => {
                        // last attempt error
                        break Err(backoff::Error::transient(e));
                    }
                    None if i == attempts.len() => {
                        // last attempt empty stream
                        break Err(backoff::Error::transient(
                            super::Error::EmptyStream,
                        ));
                    }
                    _ => {
                        // try next attempt
                    }
                }
            }
        })
        .await
    }

    fn create_streaming_event_source(
        &self,
        ApiBase { api_base, api_key }: &ApiBase,
        request: &super::request::ChatCompletionCreateParams,
    ) -> EventSource {
        let mut http_request = self
            .http_client
            .post(format!("{}/chat/completions", api_base))
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", api_key),
            );
        if let Some(ref user_agent) = self.user_agent {
            http_request = http_request.header("user-agent", user_agent);
        }
        if let Some(ref x_title) = self.x_title {
            http_request = http_request.header("x-title", x_title);
        }
        if let Some(ref referer) = self.referer {
            http_request = http_request
                .header("referer", referer)
                .header("http-referer", referer);
        }
        http_request.json(request).eventsource().unwrap()
    }

    fn create_streaming_stream(
        mut event_source: EventSource,
        first_chunk_timeout: Duration,
        other_chunk_timeout: Duration,
    ) -> impl Stream<
        Item = Result<
            super::response::streaming::ChatCompletionChunk,
            super::Error,
        >,
    > + Send
    + 'static {
        async_stream::stream! {
            let mut first = true;
            while let Some(event) = tokio::time::timeout(
                if first {
                    first_chunk_timeout
                } else {
                    other_chunk_timeout
                },
                event_source.next(),
            ).await.transpose() {
                first = false;
                match event {
                    Ok(Ok(Event::Open)) => continue,
                    Ok(Ok(Event::Message(MessageEvent { data, .. }))) => {
                        if data == "[DONE]" {
                            break;
                        } else if data.starts_with(":") {
                            continue; // skip comments
                        } else if data.is_empty() {
                            continue; // skip empty messages
                        }
                        let mut de = serde_json::Deserializer::from_str(&data);
                        match serde_path_to_error::deserialize::<
                            _,
                            super::response::streaming::ChatCompletionChunk,
                        >(&mut de)
                        {
                            Ok(mut response) => {
                                response.with_total_cost();
                                yield Ok(response)
                            },
                            Err(e) => {
                                de = serde_json::Deserializer::from_str(&data);
                                match serde_path_to_error::deserialize::<
                                    _,
                                    super::OpenRouterProviderError,
                                >(&mut de)
                                {
                                    Ok(provider_error) => yield Err(
                                        super::Error::OpenRouterProviderError(
                                            provider_error,
                                        ),
                                    ),
                                    Err(_) => yield Err(
                                        super::Error::DeserializationError(
                                            e,
                                        ),
                                    ),
                                }
                            }
                        }
                    }
                    Ok(Err(reqwest_eventsource::Error::InvalidStatusCode(
                        code,
                        response,
                    ))) => {
                        match response.text().await {
                            Ok(body) => {
                                yield Err(super::Error::BadStatus {
                                    code,
                                    body: match serde_json::from_str::<
                                        serde_json::Value,
                                    >(
                                        &body,
                                    ) {
                                        Ok(value) => value,
                                        Err(_) => serde_json::Value::String(
                                            body,
                                        ),
                                    },
                                });
                            }
                            Err(_) => {
                                yield Err(super::Error::BadStatus {
                                    code,
                                    body: serde_json::Value::Null,
                                });
                            }
                        }
                    }
                    Ok(Err(e)) => {
                        yield Err(super::Error::from(e));
                    }
                    Err(_) => {
                        yield Err(super::Error::StreamTimeout);
                    }
                }
            }
        }
    }
}

pub async fn fetch_completion_futs_from_messages<CTX: Clone>(
    completions_archive_fetcher: Arc<
        impl completions_archive::Fetcher<CTX> + Send + Sync + 'static,
    >,
    ctx: CTX,
    messages: &[super::request::Message],
) -> Result<
    HashMap<String, completions_archive::Completion>,
    crate::error::ResponseError,
> {
    // first, create a future for each unique completion in choices and messages
    let mut completions_futs = Vec::new();
    let mut ids = HashSet::new();
    for message in messages {
        match message {
            super::request::Message::ChatCompletion(
                super::request::ChatCompletionMessage { id, .. },
            ) => {
                if !ids.insert(id.as_str()) {
                    continue;
                }
                completions_futs.push(futures::future::Either::Left(
                    completions_archive_fetcher
                        .fetch_chat_completion(ctx.clone(), id)
                        .map_ok(completions_archive::Completion::Chat),
                ));
            }
            super::request::Message::ScoreCompletion(
                super::request::ScoreCompletionMessage { id, .. },
            ) => {
                if !ids.insert(id.as_str()) {
                    continue;
                }
                completions_futs.push(futures::future::Either::Right(
                    futures::future::Either::Left(
                        completions_archive_fetcher
                            .fetch_score_completion(ctx.clone(), id)
                            .map_ok(completions_archive::Completion::Score),
                    ),
                ));
            }
            super::request::Message::MultichatCompletion(
                super::request::MultichatCompletionMessage { id, .. },
            ) => {
                if !ids.insert(id.as_str()) {
                    continue;
                }
                completions_futs.push(futures::future::Either::Right(
                    futures::future::Either::Right(
                        completions_archive_fetcher
                            .fetch_multichat_completion(ctx.clone(), id)
                            .map_ok(completions_archive::Completion::Multichat),
                    ),
                ));
            }
            _ => {}
        }
    }
    if completions_futs.is_empty() {
        Ok(HashMap::new())
    } else {
        let completions =
            futures::future::try_join_all(completions_futs).await?;

        // map from id to completion
        let mut id_to_completion = HashMap::with_capacity(completions.len());
        for completion in completions {
            let id = match &completion {
                completions_archive::Completion::Chat(c) => &c.id,
                completions_archive::Completion::Score(c) => &c.id,
                completions_archive::Completion::Multichat(c) => &c.id,
            };
            id_to_completion.insert(id.clone(), completion);
        }

        Ok(id_to_completion)
    }
}

pub fn replace_completion_messages_with_assistant_messages(
    completions: &HashMap<String, completions_archive::Completion>,
    messages: &mut Vec<super::request::Message>,
) -> Result<(), super::Error> {
    if completions.len() == 0 {
        return Ok(());
    }
    for message in messages {
        let (id, choice_index, name) = match message {
            super::request::Message::ChatCompletion(
                super::request::ChatCompletionMessage {
                    id,
                    choice_index,
                    name,
                },
            ) => (id, *choice_index, name.clone()),
            super::request::Message::ScoreCompletion(
                super::request::ScoreCompletionMessage {
                    id,
                    choice_index,
                    name,
                },
            ) => (id, *choice_index, name.clone()),
            super::request::Message::MultichatCompletion(
                super::request::MultichatCompletionMessage {
                    id,
                    choice_index,
                    name,
                },
            ) => (id, *choice_index, name.clone()),
            _ => continue,
        };
        // return error if the choice_index is invalid
        let completion_choice_message = match &completions[id.as_str()] {
            completions_archive::Completion::Chat(completion) => completion
                .choices
                .iter()
                .find(|choice| choice.index == choice_index)
                .map(|choice| choice.message.clone()),
            completions_archive::Completion::Score(completion) => completion
                .choices
                .iter()
                .find(|choice| choice.index == choice_index)
                .map(|choice| choice.message.inner.clone()),
            completions_archive::Completion::Multichat(completion) => {
                completion
                    .choices
                    .iter()
                    .find(|choice| choice.index == choice_index)
                    .map(|choice| choice.message.clone())
            }
        }
        .ok_or(super::Error::InvalidCompletionChoiceIndex(
            id.clone(),
            choice_index,
        ))?;
        // replace the completion message with an assistant message
        *message = super::request::Message::Assistant(
            convert_completion_choice_message_to_assistant_message(
                completion_choice_message,
                name,
            ),
        );
    }
    Ok(())
}

pub fn convert_completion_choice_message_to_assistant_message(
    message: super::response::unary::Message,
    name: Option<String>,
) -> super::request::AssistantMessage {
    // convert generated images to input image parts
    let image_parts = message
        .images
        .map(|images| {
            let images_len = images.len();
            if images_len > 0 {
                Some((
                    images.len(),
                    images.into_iter().map(|image| {
                        super::request::RichContentPart::ImageUrl {
                            image_url: super::request::ImageUrl {
                                url: image.image_url.url,
                                detail: None,
                            },
                        }
                    }),
                ))
            } else {
                None
            }
        })
        .flatten();
    // content will be parts if there are images
    let content = match (message.content, image_parts) {
        (Some(content), Some((image_parts_len, image_parts))) => {
            let mut parts = Vec::with_capacity(1 + image_parts_len);
            parts.push(super::request::RichContentPart::Text { text: content });
            parts.extend(image_parts);
            Some(super::request::RichContent::Parts(parts))
        }
        (Some(content), None) => {
            Some(super::request::RichContent::Text(content))
        }
        (None, Some((image_parts_len, image_parts))) => {
            let mut parts = Vec::with_capacity(image_parts_len);
            parts.extend(image_parts);
            Some(super::request::RichContent::Parts(parts))
        }
        (None, None) => None,
    };
    let refusal = message.refusal;
    // convert response tool calls to assistant request tool calls
    let tool_calls = message.tool_calls.map(|tcs| {
        tcs.into_iter()
            .map(super::request::AssistantToolCall::from)
            .collect()
    });
    // OpenRouter has a field for this, should we use it?
    // Extra tokens, idk, maybe make it configurable
    let reasoning = None;

    super::request::AssistantMessage {
        content,
        name,
        refusal,
        tool_calls,
        reasoning,
    }
}
