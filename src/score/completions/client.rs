use super::weight;
use crate::{
    chat, score,
    util::{ChoiceIndexer, StreamOnce},
};
use futures::{
    Stream, StreamExt, TryStreamExt, 
};
use indexmap::IndexMap;
use rand::{Rng, seq::SliceRandom};
use regex::Regex;
use serde::ser::SerializeMap;
use std::{
    collections::HashMap,
    hash::Hash,
    sync::{Arc, LazyLock},
    time,
};

pub fn response_id(created: u64) -> String {
    let uuid = uuid::Uuid::new_v4();
    format!("rnkcpl-{}-{}", uuid.simple(), created)
}

pub struct Client<CTX, HNDLCTX, FMODEL, FSTATIC, FTRAININGTABLE> {
    pub chat_client: Arc<chat::completions::Client<CTX, HNDLCTX>>,
    pub model_fetcher: Arc<FMODEL>,
    pub weight_fetchers: Arc<
        score::completions::weight::Fetchers<CTX, FSTATIC, FTRAININGTABLE>,
    >,
}

impl<CTX, HNDLCTX, FMODEL, FSTATIC, FTRAININGTABLE> Client<CTX, HNDLCTX, FMODEL, FSTATIC, FTRAININGTABLE> {
    pub fn new(
        chat_client: Arc<chat::completions::Client<CTX, HNDLCTX>>,
        model_fetcher: Arc<FMODEL>,
        weight_fetchers: Arc<
            score::completions::weight::Fetchers<CTX, FSTATIC, FTRAININGTABLE>,
        >,
    ) -> Self {
        Self {
            chat_client,
            model_fetcher,
            weight_fetchers,
        }
    }
}

impl<CTX, HNDLCTX, FMODEL, FSTATIC, FTRAININGTABLE>
    Client<CTX, HNDLCTX, FMODEL, FSTATIC, FTRAININGTABLE>
where
    CTX: Clone + Send + Sync + 'static,
    HNDLCTX: chat::completions::CtxHandler<CTX> + Send + Sync + 'static,
    FMODEL: score::model::Fetcher + Send + Sync + 'static,
    FSTATIC: score::completions::weight::Fetcher<CTX, weight::StaticData>
        + Send
        + Sync
        + 'static,
    FTRAININGTABLE: score::completions::weight::Fetcher<CTX, weight::TrainingTableData>
        + Send
        + Sync
        + 'static,
{
    pub async fn create_unary(
        self: Arc<Self>,
        ctx: CTX,
        request: super::request::ChatCompletionCreateParams,
    ) -> Result<super::response::unary::ChatCompletion, super::Error> {
        let mut aggregate: Option<
            super::response::streaming::ChatCompletionChunk,
        > = None;
        let stream = self.create_streaming(ctx, request).await?;
        futures::pin_mut!(stream);
        while let Some(response) = stream.try_next().await? {
            match aggregate {
                Some(ref mut aggregate) => aggregate.push(&response),
                None => aggregate = Some(response),
            }
        }
        match aggregate {
            Some(response) => Ok(response.into()),
            None => unreachable!(),
        }
    }

    pub async fn create_streaming(
        self: Arc<Self>,
        ctx: CTX,
        mut request: super::request::ChatCompletionCreateParams,
    ) -> Result<
        impl Stream<
            Item = Result<
                super::response::streaming::ChatCompletionChunk,
                super::Error,
            >,
        > + Send
        + 'static,
        super::Error,
    > {
        // timestamp and identify the completion
        let created = time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let response_id = response_id(created);

        // validate choice count
        if request.choices.len() < 2 {
            return Err(super::Error::ExpectedTwoOrMoreChoices(
                request.choices.len(),
            ));
        }
        let indexer =
            Arc::new(ChoiceIndexer::new(request.choices.len() as u64));

        // fetch or validate the score model
        let model = match request.model {
            super::request::Model::Id(id) => {
                // 22 character id, fetch model
                if id.len() == 22 {
                    self.model_fetcher
                        .fetch(&id)
                        .await
                        .map_err(|e| super::Error::FetchModel(e))?
                } else {
                    match id.split("/").last() {
                        // 22 character id with author prefix, fetch model
                        Some(slug) if slug.len() == 22 => {
                            self.model_fetcher
                                .fetch(slug)
                                .await
                                .map_err(|e| super::Error::FetchModel(e))?
                        },
                        // JSON string model, parse and validate
                        _ => match serde_json::from_str::<score::model::ModelBase>(&id) {
                            Ok(provided) => provided
                                .into_model_validate()
                                .map_err(|e| super::Error::InvalidModel(e))?,
                            Err(_) => return Err(super::Error::InvalidModel(id)),
                        },
                    }
                }
            }
            // JSON body model, parse and validate
            super::request::Model::Provided(provided) => provided
                .into_model_validate()
                .map_err(|e| super::Error::InvalidModel(e))?,
        };
        request.model = super::request::Model::Id(model.id.clone());
        let request = Arc::new(request);

        // fetch weights
        let (weights, weight_data) = self
            .weight_fetchers
            .fetch(ctx.clone(), request.clone(), model.clone())
            .await
            .map_err(|e| super::Error::FetchModelWeights(e))?;

        // track usage
        let mut usage: chat::completions::response::Usage =
            chat::completions::response::Usage::default();

        // create the first chunk, containing the provided choices
        let mut aggregate = super::response::streaming::ChatCompletionChunk {
            id: response_id.clone(),
            choices: {
                let mut choices = Vec::with_capacity(request.choices.len() + model.llms.len());
                for (i, choice_) in request.choices.iter().enumerate() {
                    choices.push(super::response::streaming::Choice {
                        delta: super::response::streaming::Delta {
                            inner: chat::completions::response::streaming::Delta {
                                content: Some(choice_.clone()),
                                refusal: None,
                                role: Some(chat::completions::response::Role::Assistant),
                                tool_calls: None,
                                reasoning: None,
                                images: None,
                            },
                            vote: None,
                        },
                        finish_reason: Some(chat::completions::response::FinishReason::Stop),
                        index: i as u64,
                        logprobs: None,
                        weight: None,
                        confidence: None,
                        error: None,
                        model: None,
                        model_index: None,
                        completion_metadata: None,
                    });
                }
                choices
            },
            created,
            model: model.id.clone(),
            object: chat::completions::response::streaming::Object::ChatCompletionChunk,
            usage: None,
            weight_data: None,
        };
        let mut initial_chunk = Some(aggregate.clone());

        // stream
        Ok(async_stream::stream! {
            let mut vote_stream = futures::stream::select_all(model.llms
                .iter()
                .map(|llm| {
                    futures::stream::once(self.clone().llm_create_streaming(
                        ctx.clone(),
                        response_id.clone(),
                        created,
                        indexer.clone(),
                        llm.clone(),
                        weights[llm.index],
                        request.clone(),
                    )).flat_map(|s| s).boxed()
                })
            );
            while let Some(mut chunk) = vote_stream.next().await {
                // yield provided choices first
                if let Some(initial_chunk) = initial_chunk.take() {
                    yield Ok(initial_chunk);
                }
                for super::response::streaming::Choice {
                    completion_metadata,
                    ..
                } in &mut chunk.choices {
                    if let Some(super::response::CompletionMetadata {
                        usage: Some(llm_usage),
                        ..
                    }) = completion_metadata.as_mut() {
                        usage.push(llm_usage);
                    }
                }
                aggregate.push(&chunk);
                for super::response::streaming::Choice {
                    completion_metadata,
                    ..
                } in &mut chunk.choices {
                    if let Some(
                        completion_metadata
                    ) = completion_metadata.as_mut() {
                        // include usage only in the last chunk
                        completion_metadata.usage = None;
                    }
                }
                yield Ok(chunk);
            }

            // tally all votes and check for all-error
            let mut choice_weight = vec![0.0; request.choices.len()];
            let mut all_choices_error = true;
            let mut all_choices_error_code = None;
            for choice in &aggregate.choices {
                if all_choices_error {
                    match (&choice.error, all_choices_error_code) {
                        (None, _) => {
                            all_choices_error = false;
                        }
                        (Some(e), None) => {
                            all_choices_error_code = Some(e.code);
                        }
                        // if there's a mismatch in codes
                        (Some(e), Some(code)) if e.code != code => {
                            if e.code >= 400 && e.code < 500 && code >= 400 && code < 500 {
                                // if all codes are 400-499, set to 400
                                all_choices_error_code = Some(400);
                            } else {
                                // otherwise, set to 500
                                all_choices_error_code = Some(500);
                            }
                        }
                        _ => {}
                    }
                }
                if let Some(vote) = &choice.delta.vote {
                    for (i, v) in vote.iter().enumerate() {
                        choice_weight[i] += *v * choice.weight.unwrap_or(0.0);
                    }
                }
            }

            // yield final chunk with: 
            // - weight data
            // - usage
            // - confidence for each choice
            let choice_weight_sum = choice_weight.iter().sum::<f64>();
            aggregate.weight_data = Some(weight_data);
            aggregate.usage = Some(usage);
            for choice in &mut aggregate.choices {
                if choice.index < request.choices.len() as u64 {
                    let confidence = if choice_weight_sum > 0.0 {
                        choice_weight[choice.index as usize] / choice_weight_sum
                    } else {
                        0.0
                    };
                    choice.confidence = Some(confidence);
                } else if let Some(vote) = choice.delta.vote.take() {
                    for (i, v) in vote.into_iter().enumerate() {
                        let vote_confidence = if choice_weight_sum > 0.0 {
                            choice_weight[i] / choice_weight_sum
                        } else {
                            0.0
                        } * v;
                        choice.confidence = match choice.confidence {
                            Some(c) => Some(c + vote_confidence),
                            None => Some(vote_confidence),
                        };
                    }
                }
                choice.delta = super::response::streaming::Delta::default();
                choice.finish_reason = None;
                choice.logprobs = None;
                choice.error = None;
            }
            yield Ok(aggregate);

            // if all choices errored, yield an error
            if all_choices_error {
                yield Err(super::Error::AllVotesFailed(
                    all_choices_error_code,
                ));
            }
        })
    }

    async fn llm_create_streaming(
        self: Arc<Self>,
        ctx: CTX,
        response_id: String,
        created: u64,
        indexer: Arc<ChoiceIndexer>,
        llm: score::llm::Llm,
        weight: f64,
        request: Arc<super::request::ChatCompletionCreateParams>,
    ) -> impl Stream<Item = super::response::streaming::ChatCompletionChunk>
    + Send
    + Unpin
    + 'static {
        let super::request::ChatCompletionCreateParams {
            messages,
            seed,
            service_tier,
            tools: readonly_tools,
            choices,
            ..
        } = &*request;
        let mut messages = messages.clone();

        // create prefixes and get choices string
        let (pfx_tree, pfx_indices, choices_string) = {
            let mut rng = rand::rng();
            // create the prefixes
            let pfx_tree = SelectPfxTree::new(
                &mut rng,
                choices.len(),
                match llm.base.top_logprobs {
                    Some(top_logprobs) => top_logprobs as usize,
                    None => 20,
                },
            );
            // map prefix to choice index
            let pfx_indices = pfx_tree.pfx_indices(&mut rng, choices.len());
            // serialize choices
            let choices_string = SelectPfxTree::json_serialize_select_choices(
                &choices,
                &pfx_indices,
            );
            (pfx_tree, pfx_indices, choices_string)
        };

        // add selection to prompt
        let content = format!(
            "Select the response:\n\n{}",
            choices_string
        );
        if let Some(chat::completions::request::Message::System(
            last_message,
        )) = messages.last_mut()
        {
            match last_message.content {
                chat::completions::request::SimpleContent::Text(
                    ref mut text,
                ) => {
                    text.push_str("\n\n");
                    text.push_str(&content);
                }
                chat::completions::request::SimpleContent::Parts(
                    ref mut parts,
                ) => {
                    parts.push(chat::completions::request::SimpleContentPart {
                        text: format!("\n\n{}", content),
                        r#type: chat::completions::request::SimpleContentPartType::Text,
                    });
                }
            }
        } else {
            messages.push(chat::completions::request::Message::System(
                chat::completions::request::SystemMessage {
                    content: chat::completions::request::SimpleContent::Text(
                        content,
                    ),
                    name: None,
                },
            ));
        }

        // force assistant to output the prefix of a choice
        let response_format = ResponseKey::response_format(
            pfx_indices.into_iter().map(|(pfx, _)| pfx).collect(),
            llm.base.synthetic_reasoning.unwrap_or(false),
        );

        // if 'tool_response_format' use a required tool instead of response_format
        let (tool_response_format, response_format, tools, tool_choice) = match (
            llm.base.tool_response_format,
            response_format,
            readonly_tools,
        ) {
            (
                Some(true),
                chat::completions::request::ResponseFormat::JsonSchema {
                    json_schema: chat::completions::request::JsonSchema {
                        name,
                        description,
                        schema,
                        strict,
                    }
                },
                tools_param,
            ) => (
                true,
                None,
                Some({
                    let mut tools = tools_param.clone().unwrap_or_default();
                    tools.push(chat::completions::request::Tool {
                        r#type: chat::completions::request::ToolType::Function,
                        function: chat::completions::request::FunctionDefinition {
                            name: name.clone(),
                            description,
                            parameters: schema,
                            strict,
                        },
                    });
                    tools
                }),
                Some(chat::completions::request::ToolChoice::Function(
                    chat::completions::request::ToolChoiceFunction {
                        r#type: chat::completions::request::ToolChoiceFunctionType::Function,
                        function: chat::completions::request::ToolChoiceFunctionFunction {
                            name,
                        }
                    }
                )),
            ),
            (
                _,
                response_format,
                Some(tools_param),
            ) if !tools_param.is_empty() => {
                (
                    false,
                    Some(response_format),
                    Some(tools_param.clone()),
                    Some(chat::completions::request::ToolChoice::None),
                )
            },
            (
                _,
                response_format,
                _,
            ) => (false, Some(response_format), None, None),
        };

        // stream
        let mut stream = match self
            .chat_client
            .clone()
            .create_streaming(
                ctx,
                chat::completions::request::ChatCompletionCreateParams {
                    messages,
                    model: llm.base.model.clone(),
                    frequency_penalty: llm.base.frequency_penalty,
                    logit_bias: llm.base.logit_bias.clone(),
                    logprobs: if llm.base.top_logprobs.is_some() {
                        Some(true)
                    } else {
                        None
                    },
                    max_completion_tokens: llm.base.max_completion_tokens,
                    modalities: None,
                    n: None,
                    parallel_tool_calls: None,
                    prediction: None,
                    presence_penalty: llm.base.presence_penalty,
                    reasoning_effort: None,
                    response_format,
                    seed: *seed,
                    service_tier: *service_tier,
                    stop: llm.base.stop.clone(),
                    stream: request.stream,
                    stream_options: request.stream_options,
                    temperature: llm.base.temperature,
                    tool_choice,
                    tools,
                    top_logprobs: llm.base.top_logprobs,
                    top_p: llm.base.top_p,
                    web_search_options: None,
                    max_tokens: llm.base.max_tokens,
                    min_p: llm.base.min_p,
                    plugins: None,
                    provider: llm.base.provider.clone(),
                    reasoning: llm.base.reasoning,
                    repetition_penalty: llm.base.repetition_penalty,
                    top_a: llm.base.top_a,
                    top_k: llm.base.top_k,
                    usage: request.usage,
                    verbosity: llm.base.verbosity,
                    models: llm.base.models.clone(),
                },
            )
            .await
        {
            Ok(stream) => stream.boxed(),
            Err(e) => {
                return StreamOnce::new(
                    super::response::streaming::ChatCompletionChunk {
                        id: response_id,
                        choices: vec![
                            super::response::streaming::Choice {
                                delta: super::response::streaming::Delta::default(),
                                finish_reason: Some(
                                    chat::completions::response::FinishReason::Error,
                                ),
                                index: indexer.get(llm.index, 0),
                                logprobs: None,
                                weight: Some(weight),
                                confidence: None,
                                error: Some(crate::error::ResponseError::from(
                                    &super::Error::from(e),
                                )),
                                model: Some(llm.id),
                                model_index: Some(llm.index),
                                completion_metadata: None,
                            },
                        ],
                        created,
                        model: request.model.unwrap_id().to_owned(),
                        object: chat::completions::response::streaming::Object::ChatCompletionChunk,
                        usage: None,
                        weight_data: None,
                    }
                )
                .boxed();
            }
        };

        // only return error if the very first stream item is an error
        let mut current = Some(match stream.try_next().await {
            Ok(Some(chunk)) => chunk,
            // chat client will always yield at least 1 chunk or error
            Ok(None) => unreachable!(),
            Err(e) => {
                return StreamOnce::new(
                    super::response::streaming::ChatCompletionChunk {
                        id: response_id,
                        choices: vec![
                            super::response::streaming::Choice {
                                delta: super::response::streaming::Delta::default(),
                                finish_reason: Some(
                                    chat::completions::response::FinishReason::Error,
                                ),
                                index: indexer.get(llm.index, 0),
                                logprobs: None,
                                weight: Some(weight),
                                confidence: None,
                                error: Some(crate::error::ResponseError::from(
                                    &e,
                                )),
                                model: Some(llm.id),
                                model_index: Some(llm.index),
                                completion_metadata: None,
                            },
                        ],
                        created,
                        model: request.model.unwrap_id().to_owned(),
                        object: chat::completions::response::streaming::Object::ChatCompletionChunk,
                        usage: None,
                        weight_data: None,
                    }
                )
                .boxed();
            }
        });

        // the aggregate of all chunks
        let mut aggregate: Option<
            super::response::streaming::ChatCompletionChunk,
        > = None;

        // select chunkers
        let mut chunkers: HashMap<u64, SelectChunker> =
            HashMap::with_capacity(1);

        async_stream::stream! {
            while let Some(result) = stream.next().await {
                // errors go into individual choices
                let (prev, error) = match result {
                    Ok(chunk) => (current.replace(chunk).unwrap(), None),
                    Err(e) => (current.take().unwrap(), Some(crate::error::ResponseError::from(&e))),
                };
                let error_is_some = error.is_some();
                let mut chunk = super::response::streaming::ChatCompletionChunk {
                    id: response_id.clone(),
                    choices: prev.choices.into_iter().map(|choice| {
                        super::response::streaming::Choice {
                            delta: super::response::streaming::Delta {
                                inner: choice.delta,
                                vote: None,
                            },
                            finish_reason: if error.is_some() {
                                Some(chat::completions::response::FinishReason::Error)
                            } else {
                                choice.finish_reason
                            },
                            index: indexer.get(llm.index, choice.index),
                            logprobs: choice.logprobs,
                            weight: Some(weight),
                            confidence: None,
                            error: error.clone(),
                            model: Some(llm.id.clone()),
                            model_index: Some(llm.index),
                            completion_metadata: Some(super::response::CompletionMetadata {
                                id: prev.id.clone(),
                                created: prev.created,
                                model: prev.model.clone(),
                                service_tier: prev.service_tier,
                                system_fingerprint: prev.system_fingerprint.clone(),
                                usage: prev.usage.clone(),
                                provider: prev.provider.clone(),
                            }),
                        }
                    }).collect(),
                    created,
                    model: request.model.unwrap_id().to_owned(),
                    object: chat::completions::response::streaming::Object::ChatCompletionChunk,
                    usage: None,
                    weight_data: None,
                };
                if tool_response_format {
                    chunk.tool_as_content();
                }
                for choice in &mut chunk.choices {
                    let chunker = chunkers.entry(choice.index).or_insert_with(||
                        SelectChunker::new(
                            pfx_tree.clone(),
                            request.choices.len(),
                        )
                    );
                    chunker.handle(choice);
                }
                match aggregate {
                    Some(ref mut aggregate) => {
                        aggregate.push(&chunk);
                    }
                    None => {
                        aggregate = Some(chunk.clone());
                    }
                }
                if error_is_some {
                    // once we encounter an error, the stream is over
                    for choice in &mut chunk.choices {
                        let chunker = match chunkers.remove(&choice.index) {
                            Some(chunker) => chunker,
                            None => SelectChunker::new(
                                pfx_tree.clone(),
                                request.choices.len(),
                            )
                        };
                        chunker.handle_final(choice);
                    }
                    yield chunk;
                    return;
                }
                yield chunk;
            }
            let current = current.unwrap();
            let mut chunk = super::response::streaming::ChatCompletionChunk {
                id: response_id.clone(),
                choices: current.choices.into_iter().map(|choice| {
                    super::response::streaming::Choice {
                        delta: super::response::streaming::Delta {
                            inner: choice.delta,
                            vote: None,
                        },
                        finish_reason: 
                            choice.finish_reason,
                        index: indexer.get(llm.index, choice.index),
                        logprobs: choice.logprobs,
                        weight: Some(weight),
                        confidence: None,
                        error: None,
                        model: Some(llm.id.clone()),
                        model_index: Some(llm.index),
                        completion_metadata: Some(super::response::CompletionMetadata {
                            id: current.id.clone(),
                            created: current.created,
                            model: current.model.clone(),
                            service_tier: current.service_tier,
                            system_fingerprint: current.system_fingerprint.clone(),
                            usage: current.usage.clone(),
                            provider: current.provider.clone(),
                        }),
                    }
                }).collect(),
                created,
                model: request.model.unwrap_id().to_owned(),
                object: chat::completions::response::streaming::Object::ChatCompletionChunk,
                usage: None,
                weight_data: None,
            };
            if tool_response_format {
                chunk.tool_as_content();
            }
            for choice in &mut chunk.choices {
                let chunker = chunkers.entry(choice.index).or_insert_with(||
                    SelectChunker::new(
                        pfx_tree.clone(),
                        request.choices.len(),
                    )
                );
                chunker.handle(choice);
            }
            let mut aggregate = match aggregate {
                Some(mut aggregate) => {
                    aggregate.push(&chunk);
                    aggregate
                }
                None => {
                    chunk.clone()
                }
            };
            yield chunk;

            // yield synthetic reasonings and votes
            for choice in &mut aggregate.choices {
                choice.delta.inner.refusal = None;
                choice.delta.inner.tool_calls = None;
                choice.delta.inner.reasoning = None;
                choice.delta.inner.images = None;
                let chunker = match chunkers.remove(&choice.index) {
                    Some(chunker) => chunker,
                    None => SelectChunker::new(
                        pfx_tree.clone(),
                        request.choices.len(),
                    )
                };
                chunker.handle_final(choice);
                choice.delta.inner.content = None;
            }
            yield aggregate;
        }.boxed()
    }
}

#[derive(Debug, serde::Deserialize)]
struct ResponseKey {
    _think: Option<String>,
    response_key: String,
}

impl ResponseKey {
    fn response_format(
        ids: Vec<String>,
        think: bool,
    ) -> chat::completions::request::ResponseFormat {
        chat::completions::request::ResponseFormat::JsonSchema {
            json_schema: chat::completions::request::JsonSchema {
                name: "response_key".to_string(),
                description: None,
                strict: Some(true),
                schema: Some(if think {
                    serde_json::json!({
                        "type": "object",
                        "properties": {
                            "_think": {
                                "type": "string",
                                "description": "The assistant's internal reasoning.",
                            },
                            "response_key": {
                                "type": "string",
                                "enum": ids
                            }
                        },
                        "required": ["_think", "response_key"],
                        "additionalProperties": false,
                    })
                } else {
                    serde_json::json!({
                        "type": "object",
                        "properties": {
                            "response_key": {
                                "type": "string",
                                "enum": ids
                            }
                        },
                        "required": ["response_key"],
                        "additionalProperties": false,
                    })
                }),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SelectPfx {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
}

impl SelectPfx {
    fn to_char(&self) -> char {
        match self {
            SelectPfx::A => 'A',
            SelectPfx::B => 'B',
            SelectPfx::C => 'C',
            SelectPfx::D => 'D',
            SelectPfx::E => 'E',
            SelectPfx::F => 'F',
            SelectPfx::G => 'G',
            SelectPfx::H => 'H',
            SelectPfx::I => 'I',
            SelectPfx::J => 'J',
            SelectPfx::K => 'K',
            SelectPfx::L => 'L',
            SelectPfx::M => 'M',
            SelectPfx::N => 'N',
            SelectPfx::O => 'O',
            SelectPfx::P => 'P',
            SelectPfx::Q => 'Q',
            SelectPfx::R => 'R',
            SelectPfx::S => 'S',
            SelectPfx::T => 'T',
        }
    }

    fn from_char(c: char) -> Option<Self> {
        match c {
            'A' => Some(SelectPfx::A),
            'B' => Some(SelectPfx::B),
            'C' => Some(SelectPfx::C),
            'D' => Some(SelectPfx::D),
            'E' => Some(SelectPfx::E),
            'F' => Some(SelectPfx::F),
            'G' => Some(SelectPfx::G),
            'H' => Some(SelectPfx::H),
            'I' => Some(SelectPfx::I),
            'J' => Some(SelectPfx::J),
            'K' => Some(SelectPfx::K),
            'L' => Some(SelectPfx::L),
            'M' => Some(SelectPfx::M),
            'N' => Some(SelectPfx::N),
            'O' => Some(SelectPfx::O),
            'P' => Some(SelectPfx::P),
            'Q' => Some(SelectPfx::Q),
            'R' => Some(SelectPfx::R),
            'S' => Some(SelectPfx::S),
            'T' => Some(SelectPfx::T),
            _ => None,
        }
    }

    fn rng_vec(rng: &mut impl Rng) -> Vec<Self> {
        let mut vec = vec![
            SelectPfx::A,
            SelectPfx::B,
            SelectPfx::C,
            SelectPfx::D,
            SelectPfx::E,
            SelectPfx::F,
            SelectPfx::G,
            SelectPfx::H,
            SelectPfx::I,
            SelectPfx::J,
            SelectPfx::K,
            SelectPfx::L,
            SelectPfx::M,
            SelectPfx::N,
            SelectPfx::O,
            SelectPfx::P,
            SelectPfx::Q,
            SelectPfx::R,
            SelectPfx::S,
            SelectPfx::T,
        ];
        vec.shuffle(rng);
        vec
    }
}

impl std::fmt::Display for SelectPfx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

#[derive(Debug, Clone)]
enum SelectPfxTree {
    Branch(Arc<IndexMap<SelectPfx, SelectPfxTree>>),
    Leaf(usize),
}

impl SelectPfxTree {
    fn new(
        rng: &mut impl Rng,
        source_len: usize,
        max_branch_len: usize,
    ) -> Self {
        let mut source: Vec<usize> = (0..source_len).collect();
        source.shuffle(rng);
        Self::new_inner(rng, &source, max_branch_len, false)
    }

    fn new_inner(
        rng: &mut impl Rng,
        source: &[usize],
        max_branch_len: usize,
        force_sub_branch: bool,
    ) -> Self {
        let pfxs = SelectPfx::rng_vec(rng);
        if !force_sub_branch && source.len() <= max_branch_len {
            // return a single branch containing all leaves
            let mut branch = IndexMap::with_capacity(source.len());
            for (i, source_index) in source.iter().enumerate() {
                branch.insert(pfxs[i], SelectPfxTree::Leaf(*source_index));
            }
            Self::Branch(Arc::new(branch))
        } else {
            // split into sub-branches
            let n = {
                let candidate =
                    (source.len() + max_branch_len - 1) / max_branch_len;
                if candidate <= max_branch_len {
                    candidate
                } else {
                    max_branch_len
                }
            };
            let base_per = source.len() / n;
            let extra = source.len() % n;
            let force_sub_branch =
                base_per + { if extra > 0 { 1 } else { 0 } } > max_branch_len;
            let mut branch = IndexMap::with_capacity(n);
            let mut i = 0;
            let mut count = 0;
            while i < n {
                let branch_len = base_per + if i < extra { 1 } else { 0 };
                branch.insert(
                    pfxs[i],
                    SelectPfxTree::new_inner(
                        rng,
                        &source[count..count + branch_len],
                        max_branch_len,
                        force_sub_branch,
                    ),
                );
                count += branch_len;
                i += 1;
            }
            Self::Branch(Arc::new(branch))
        }
    }

    fn pfx_indices(
        &self,
        rng: &mut impl Rng,
        source_len: usize,
    ) -> Vec<(String, usize)> {
        let mut indices = Vec::with_capacity(source_len);
        self.pfx_indices_inner(None, &mut indices);
        indices.shuffle(rng);
        indices
    }

    fn pfx_indices_inner(
        &self,
        parent_pfx: Option<String>,
        indices: &mut Vec<(String, usize)>,
    ) {
        match self {
            SelectPfxTree::Branch(branch) => {
                for (pfx, child) in branch.as_ref() {
                    let parent_pfx = Some(match &parent_pfx {
                        Some(parent_pfx) => format!("{}`{}`", parent_pfx, pfx),
                        None => format!("`{}`", pfx),
                    });
                    child.pfx_indices_inner(parent_pfx, indices);
                }
            }
            SelectPfxTree::Leaf(index) => {
                indices.push((parent_pfx.unwrap(), *index));
            }
        }
    }

    fn json_serialize_select_choices(
        choices: &[String],
        indices: &[(String, usize)],
    ) -> String {
        struct OrderedChoices<'a> {
            indices: &'a [(String, usize)],
            choices: &'a [String],
        }
        impl<'a> serde::Serialize for OrderedChoices<'a> {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                let mut map =
                    serializer.serialize_map(Some(self.indices.len()))?;
                for (pfx_key, index) in self.indices {
                    map.serialize_entry(pfx_key, &self.choices[*index])?;
                }
                map.end()
            }
        }
        serde_json::to_string_pretty(&OrderedChoices { indices, choices })
            .unwrap()
    }

    fn get(&self, pfx: SelectPfx) -> Option<SelectPfxTree> {
        match self {
            SelectPfxTree::Branch(branch) => branch.get(&pfx).cloned(),
            SelectPfxTree::Leaf(_) => None,
        }
    }

    fn is_leaf_branch(&self) -> bool {
        match self {
            SelectPfxTree::Branch(branch) => branch
                .values()
                .any(|v| matches!(v, SelectPfxTree::Leaf { .. })),
            SelectPfxTree::Leaf(_) => false,
        }
    }
}

struct SelectChunker {
    pfx_tree: SelectPfxTree,
    pfx_tree_depth: usize,
    choice_len: usize,
    native_reasoning: bool,
    mode: Option<SelectChunkerMode>,
    key: Option<String>,
    prev_char_escaped: bool,
    logprobs: Option<chat::completions::response::Logprobs>,
    finish_reason: Option<chat::completions::response::FinishReason>,
    completion_metadata: Option<super::response::CompletionMetadata>,
}

impl SelectChunker {
    pub fn new(pfx_tree: SelectPfxTree, choice_len: usize) -> Self {
        Self {
            pfx_tree,
            pfx_tree_depth: 0,
            choice_len,
            native_reasoning: false,
            mode: None,
            key: None,
            prev_char_escaped: false,
            logprobs: None,
            finish_reason: None,
            completion_metadata: None,
        }
    }

    // returns true if the choice should be yielded
    pub fn handle(
        &mut self,
        super::response::streaming::Choice {
            delta:
                super::response::streaming::Delta {
                    inner:
                        chat::completions::response::streaming::Delta {
                            content,
                            reasoning,
                            ..
                        },
                    ..
                },
            logprobs,
            finish_reason,
            completion_metadata,
            ..
        }: &mut super::response::streaming::Choice,
    ) {
        if reasoning.as_ref().is_some_and(|r| !r.is_empty()) {
            self.native_reasoning = true;
        }
        if let Some(completion_metadata) = completion_metadata {
            self.completion_metadata = Some(completion_metadata.clone());
            completion_metadata.usage = None;
        }
        self.finish_reason = self.finish_reason.or(finish_reason.take());
        if matches!(self.mode, Some(SelectChunkerMode::Done)) {
            return;
        }
        let content = match content.as_deref() {
            Some(content) => content,
            None => return,
        };
        for c in content.chars() {
            match (c, self.mode, self.prev_char_escaped) {
                ('\\', Some(SelectChunkerMode::InKey), false) => {
                    self.prev_char_escaped = true;
                }
                ('\\', Some(SelectChunkerMode::InOtherValue), false) => {
                    self.prev_char_escaped = true;
                }
                ('"', None, _) => {
                    self.mode = Some(SelectChunkerMode::InKey);
                    self.prev_char_escaped = false;
                }
                ('"', Some(SelectChunkerMode::BetweenKeyAndValue), _) => {
                    self.mode = Some(match self.key.as_deref() {
                        Some("response_key") => {
                            SelectChunkerMode::InResponseKeyValue
                        }
                        Some(_) => SelectChunkerMode::InOtherValue,
                        None => unreachable!(),
                    });
                    self.key = None;
                    self.prev_char_escaped = false;
                }
                ('"', Some(SelectChunkerMode::InKey), false) => {
                    self.mode = Some(SelectChunkerMode::BetweenKeyAndValue);
                }
                ('"', Some(SelectChunkerMode::InOtherValue), false) => {
                    self.mode = None;
                }
                (
                    '"',
                    Some(SelectChunkerMode::InResponseKeyValue),
                    false,
                ) => {
                    self.mode = Some(SelectChunkerMode::Done);
                    return;
                }
                (c, Some(SelectChunkerMode::InKey), _) => {
                    self.push_to_key(c);
                    self.prev_char_escaped = false;
                }
                (_, Some(SelectChunkerMode::InOtherValue), _) => {
                    self.prev_char_escaped = false;
                }
                // enter tick
                (
                    '`',
                    Some(SelectChunkerMode::InResponseKeyValue),
                    _,
                ) => {
                    self.mode = Some(
                        SelectChunkerMode::InResponseKeyValueInTick,
                    );
                    self.prev_char_escaped = false;
                }
                // exit tick
                (
                    '`',
                    Some(SelectChunkerMode::InResponseKeyValueInTick),
                    _,
                ) => {
                    self.mode =
                        Some(SelectChunkerMode::InResponseKeyValue);
                }
                // select pfx
                (
                    c,
                    Some(SelectChunkerMode::InResponseKeyValueInTick),
                    _,
                ) => {
                    let pfx = match SelectPfx::from_char(c) {
                        Some(pfx) => pfx,
                        _ => {
                            self.mode = Some(SelectChunkerMode::Done);
                            return;
                        }
                    };
                    if self.pfx_tree.is_leaf_branch() {
                        self.logprobs = logprobs.clone();
                    } else {
                        match self.pfx_tree.get(pfx) {
                            Some(pfx_tree) => {
                                self.pfx_tree = pfx_tree;
                                self.pfx_tree_depth += 1;
                            }
                            None => {
                                self.mode = Some(SelectChunkerMode::Done);
                                return;
                            }
                        }
                    }
                }
                (_, Some(SelectChunkerMode::InResponseKeyValue), _) => {
                    self.prev_char_escaped = false;
                }
                _ => {}
            }
        }
    }

    pub fn handle_final(
        mut self,
        super::response::streaming::Choice {
            delta:
                super::response::streaming::Delta {
                    inner:
                        chat::completions::response::streaming::Delta {
                            content,
                            reasoning,
                            ..
                        },
                    vote,
                },
            error,
            finish_reason,
            completion_metadata,
            ..
        }: &mut super::response::streaming::Choice,
    ) {
        static RE_PFX: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r#"^(?:`[A-T]`)+$"#).unwrap());
        static RE_PFX_CAP: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r#"`([A-T])`"#).unwrap());
        *finish_reason = self.finish_reason;
        *completion_metadata = self.completion_metadata;
        let content = match content.as_deref() {
            Some(content) => {
                // remove everything preceding the first '{' or following the final '}'
                let start = content.find('{').unwrap_or(0);
                let end = content.rfind('}').map(|i| i + 1).unwrap_or(content.len());
                if end < start {
                    // safeguard for if final '}' comes before first '{'
                    content
                } else {
                    &content[start..end]
                }
            },
            None => return,
        };
        if !matches!(
            self.finish_reason,
            Some(chat::completions::response::FinishReason::Stop),
        ) {
            return;
        }
        let mut de = serde_json::Deserializer::from_str(content);
        let ResponseKey {
            _think,
            response_key,
        } = match serde_path_to_error::deserialize(&mut de) {
            Ok(key) => key,
            Err(e) => {
                println!("{}\n{}", content, e);
                *error = Some(crate::error::ResponseError::from(
                    &super::Error::InvalidChoiceContent(e),
                ));
                *finish_reason =
                    Some(chat::completions::response::FinishReason::Error);
                return;
            }
        };
        if let Some(_think) = _think {
            *reasoning = Some(format!(
                "{}{}",
                reasoning.as_deref().unwrap_or(""),
                if self.native_reasoning {
                    format!("\n\n{}", _think)
                } else {
                    _think
                }
            ));
        }
        if !RE_PFX.is_match(&response_key) {
            *error = Some(crate::error::ResponseError::from(
                &super::Error::InvalidSelection(response_key.clone()),
            ));
            *finish_reason =
                Some(chat::completions::response::FinishReason::Error);
            return;
        }

        // get proportional confidence
        if let Some(chat::completions::response::Logprobs {
            content: Some(logprobs),
            ..
        }) = self.logprobs
            && logprobs.len() > 0
        {
            for chat::completions::response::Logprob {
                token,
                top_logprobs,
                ..
            } in logprobs
            {
                match token
                    .trim_matches('`')
                    .chars()
                    .next()
                    .map(SelectPfx::from_char)
                {
                    Some(Some(_)) => {}
                    _ => continue,
                };
                let mut pfx_probabilities =
                    HashMap::with_capacity(top_logprobs.len());
                let mut probabilities_sum = 0.0;
                for chat::completions::response::TopLogprob {
                    token,
                    logprob,
                    ..
                } in top_logprobs
                {
                    if logprob.is_none() {
                        continue;
                    }
                    let pfx = match token
                        .trim_matches('`')
                        .chars()
                        .next()
                        .map(SelectPfx::from_char)
                    {
                        Some(Some(pfx)) => pfx,
                        _ => continue,
                    };
                    let index = match self.pfx_tree.get(pfx) {
                        Some(SelectPfxTree::Leaf(key)) => key,
                        _ => continue,
                    };
                    let probability = logprob.unwrap().exp();
                    pfx_probabilities
                        .entry(pfx)
                        .and_modify(|(p, _)| *p += probability)
                        .or_insert((probability, index));
                    probabilities_sum += probability;
                }
                if pfx_probabilities.len() <= 1 {
                    continue;
                }
                let mut vote_ = vec![0.0; self.choice_len];
                for (probability, index) in pfx_probabilities.into_values() {
                    vote_[index] += probability / probabilities_sum;
                }
                *vote = Some(vote_);
                return;
            }
        }

        // fallback, or no logprobs
        let captures = RE_PFX_CAP.captures_iter(&response_key);
        for (i, cap) in captures.enumerate() {
            if i < self.pfx_tree_depth {
                continue;
            }
            let pfx = SelectPfx::from_char(
                cap.get(1).unwrap().as_str().chars().next().unwrap(),
            )
            .unwrap();
            self.pfx_tree = match self.pfx_tree.get(pfx) {
                Some(pfx_tree) => pfx_tree,
                None => break,
            };
            match self.pfx_tree {
                SelectPfxTree::Branch(_) => {}
                SelectPfxTree::Leaf(index) => {
                    let mut vote_ = vec![0.0; self.choice_len];
                    vote_[index] = 1.0;
                    *vote = Some(vote_);
                    return;
                }
            }
        }

        *error = Some(crate::error::ResponseError::from(
            &super::Error::InvalidSelection(response_key.clone()),
        ));
        *finish_reason = Some(chat::completions::response::FinishReason::Error);
    }

    fn push_to_key(&mut self, c: char) {
        match &mut self.key {
            Some(key) => {
                key.push(c);
            }
            None => {
                self.key = Some(c.to_string());
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectChunkerMode {
    InKey,
    BetweenKeyAndValue,
    InResponseKeyValue,
    InResponseKeyValueInTick,
    InOtherValue,
    Done,
}
