use super::weight;
use crate::{
    chat, completions_archive, score,
    util::{ChoiceIndexer, StreamOnce},
};
use futures::{Stream, StreamExt, TryFutureExt, TryStreamExt};
use indexmap::IndexMap;
use rand::{Rng, seq::SliceRandom};
use regex::Regex;
use rust_decimal::MathematicalOps;
use serde::ser::SerializeMap;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    sync::Arc,
    time,
};

pub fn response_id(created: u64) -> String {
    let uuid = uuid::Uuid::new_v4();
    format!("scrcpl-{}-{}", uuid.simple(), created)
}

pub struct Client<CTX, CCLIENT, FMODEL, FSTATIC, FTRAININGTABLE, FCOMPLETIONS> {
    pub chat_client: Arc<CCLIENT>,
    pub model_fetcher: Arc<FMODEL>,
    pub weight_fetchers:
        Arc<score::completions::weight::Fetchers<CTX, FSTATIC, FTRAININGTABLE>>,
    pub completions_archive_fetcher: Arc<FCOMPLETIONS>,
}

impl<CTX, CCLIENT, FMODEL, FSTATIC, FTRAININGTABLE, FCOMPLETIONS>
    Client<CTX, CCLIENT, FMODEL, FSTATIC, FTRAININGTABLE, FCOMPLETIONS>
{
    pub fn new(
        chat_client: Arc<CCLIENT>,
        model_fetcher: Arc<FMODEL>,
        weight_fetchers: Arc<
            score::completions::weight::Fetchers<CTX, FSTATIC, FTRAININGTABLE>,
        >,
        completions_archive_fetcher: Arc<FCOMPLETIONS>,
    ) -> Self {
        Self {
            chat_client,
            model_fetcher,
            weight_fetchers,
            completions_archive_fetcher,
        }
    }
}

impl<CTX, CCLIENT, FMODEL, FSTATIC, FTRAININGTABLE, FCOMPLETIONS>
    Client<CTX, CCLIENT, FMODEL, FSTATIC, FTRAININGTABLE, FCOMPLETIONS>
where
    CTX: Clone + Send + Sync + 'static,
    CCLIENT: chat::completions::Client<CTX> + Send + Sync + 'static,
    FMODEL: score::model::Fetcher<CTX> + Send + Sync + 'static,
    FSTATIC: score::completions::weight::Fetcher<CTX, weight::StaticData>
        + Send
        + Sync
        + 'static,
    FTRAININGTABLE: score::completions::weight::Fetcher<CTX, weight::TrainingTableData>
        + Send
        + Sync
        + 'static,
    FCOMPLETIONS: completions_archive::Fetcher<CTX> + Send + Sync + 'static,
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

        // fetch or validate the score model and fetch completions
        let (model, completions) = tokio::try_join!(
            fetch_or_validate_score_model(
                self.model_fetcher.clone(),
                ctx.clone(),
                request.model
            ),
            fetch_completion_futs_from_choices_and_messages(
                self.completions_archive_fetcher.clone(),
                ctx.clone(),
                &request.choices,
                &request.messages
            )
            .map_err(super::Error::CompletionsArchiveError)
        )?;

        // replace request model, choices, and messages
        request.model = super::request::Model::Id(model.id.clone());
        replace_completion_messages_and_completion_choices_with_assistant_messages_and_text_choices(
            completions,
            &mut request.choices,
            &mut request.messages,
        )?;

        // wrap finalized request in Arc
        let request = Arc::new(request);

        // fetch weights
        let (weights, weight_data) = self
            .weight_fetchers
            .fetch(ctx.clone(), request.clone(), model.clone())
            .await
            .map_err(|e| super::Error::FetchModelWeights(e))?;

        // track usage
        let mut usage: chat::completions::response::Usage = match &weight_data {
            super::weight::Data::TrainingTable(ttd) => {
                ttd.embeddings_response.usage.clone().unwrap_or_default()
            }
            super::weight::Data::Static(_) => {
                chat::completions::response::Usage::default()
            }
        };

        // create the first chunk, containing the provided choices
        let mut aggregate = super::response::streaming::ChatCompletionChunk {
            id: response_id.clone(),
            choices: {
                let mut choices = Vec::with_capacity(request.choices.len() + model.llms.len());
                for (i, choice_) in request.choices.iter().enumerate() {
                    choices.push(super::response::streaming::Choice {
                        delta: super::response::streaming::Delta {
                            inner: chat::completions::response::streaming::Delta {
                                content: Some(choice_.unwrap_text().to_owned()),
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
        let indexer =
            Arc::new(ChoiceIndexer::new(request.choices.len() as u64));
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
                // accumulate all chunks into aggregate
                aggregate.push(&chunk);
                // handle usage for each choice
                for super::response::streaming::Choice {
                    completion_metadata,
                    ..
                } in &mut chunk.choices
                {
                    if let Some(super::response::CompletionMetadata {
                        usage: llm_usage,
                        ..
                    }) = completion_metadata.as_mut() {
                        // accumulate usage
                        // include usage only in the last chunk
                        if let Some(llm_usage) = llm_usage.take() {
                            usage.push(&llm_usage);
                        }
                    }
                }
                yield Ok(chunk);
            }

            // tally all votes and check for all-error
            let mut choice_weight = vec![rust_decimal::Decimal::ZERO; request.choices.len()];
            let mut all_choices_error = true;
            let mut all_choices_error_code = None;
            for choice in aggregate.choices.iter().skip(request.choices.len()) {
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
                        choice_weight[i] += *v * choice.weight
                            .unwrap_or(rust_decimal::Decimal::ZERO);
                    }
                }
            }

            // yield final chunk with:
            // - weight data
            // - usage
            // - confidence for each choice
            let choice_weight_sum = choice_weight
                .iter()
                .sum::<rust_decimal::Decimal>();
            aggregate.weight_data = Some(weight_data);
            usage.with_total_cost();
            aggregate.usage = Some(usage);
            for choice in &mut aggregate.choices {
                if choice.index < request.choices.len() as u64 {
                    let confidence = if choice_weight_sum > rust_decimal::Decimal::ZERO {
                        choice_weight[choice.index as usize] / choice_weight_sum
                    } else {
                        rust_decimal::Decimal::ZERO
                    };
                    choice.confidence = Some(confidence);
                } else if let Some(vote) = choice.delta.vote.take() {
                    for (i, v) in vote.into_iter().enumerate() {
                        let vote_confidence = if choice_weight_sum > rust_decimal::Decimal::ZERO {
                            choice_weight[i] / choice_weight_sum
                        } else {
                            rust_decimal::Decimal::ZERO
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
        weight: rust_decimal::Decimal,
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
        let choices_len = choices.len();
        let mut messages = messages.clone();
        if let Some(mut prefix_messages) = llm.base.prefix_messages {
            prefix_messages.extend(messages);
            messages = prefix_messages;
        }
        if let Some(suffix_messages) = llm.base.suffix_messages {
            messages.extend(suffix_messages);
        }

        // create prefixes and get choices string
        let (pfx_tree, pfx_indices, choices_string) = {
            let mut rng = rand::rng();
            // create the prefixes
            let pfx_tree = SelectPfxTree::new(
                &mut rng,
                choices_len,
                match llm.base.top_logprobs {
                    Some(top_logprobs) => top_logprobs as usize,
                    None => 20,
                },
            );
            // map prefix to choice index
            let pfx_indices = pfx_tree.pfx_indices(&mut rng, choices_len);
            // serialize choices
            let choices_string = SelectPfxTree::json_serialize_select_choices(
                &choices,
                &pfx_indices,
            );
            (pfx_tree, pfx_indices, choices_string)
        };

        // all possible selectable response keys
        let choices_keys = pfx_indices
            .into_iter()
            .map(|(pfx, _)| pfx)
            .collect::<Vec<_>>();

        let (
            // regex capture pattern matching response keys as-is
            choices_key_pattern,
            // regex capture pattern matching response keys stripped of first and last tick
            choices_key_pattern_stripped,
        ) = pfx_tree.regex_patterns(&choices_keys);

        // add selection to prompt
        let content = match llm.base.output_mode {
            score::llm::OutputMode::Instruction => format!(
                "Select the response:\n\n{}\n\nOutput exactly one response key including backticks, nothing else:\n- {}",
                choices_string,
                choices_keys.join("\n- ")
            ),
            score::llm::OutputMode::JsonSchema
            | score::llm::OutputMode::ToolCall => {
                format!("Select the response:\n\n{}", choices_string)
            }
        };
        if let Some(chat::completions::request::Message::System(last_message)) =
            messages.last_mut()
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

        // potentially force assistant to output the prefix of a choice
        // maybe the compiler skips this when it goes unused (e.g. OutputMode::Instruction)
        let response_format = ResponseKey::response_format(
            choices_keys,
            llm.base.synthetic_reasoning.unwrap_or(false),
        );

        // if 'tool_response_format' use a required tool instead of response_format
        let (response_format, tools, tool_choice) = match (
            llm.base.output_mode,
            response_format,
            readonly_tools,
        ) {
            (
                score::llm::OutputMode::Instruction,
                _,
                Some(tools_param),
            ) if !tools_param.is_empty() => {
                (
                    None,
                    Some(tools_param.clone()),
                    Some(chat::completions::request::ToolChoice::None),
                )
            },
            (
                score::llm::OutputMode::Instruction,
                _,
                _,
            ) => (None, None, None),
            (
                score::llm::OutputMode::JsonSchema,
                response_format,
                Some(tools_param),
            ) if !tools_param.is_empty() => {
                (
                    Some(response_format),
                    Some(tools_param.clone()),
                    Some(chat::completions::request::ToolChoice::None),
                )
            },
            (
                score::llm::OutputMode::JsonSchema,
                response_format,
                _,
            ) => (Some(response_format), None, None),
            (
                score::llm::OutputMode::ToolCall,
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
                score::llm::OutputMode::ToolCall,
                _,
                _,
            ) => unreachable!(),
        };

        // stream
        let mut stream = match self
            .chat_client
            .clone()
            .create_streaming(
                ctx,
                chat::completions::request::ChatCompletionCreateParams {
                    messages,
                    model: llm.base.model,
                    frequency_penalty: llm.base.frequency_penalty,
                    logit_bias: llm.base.logit_bias,
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
                    stop: llm.base.stop,
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
                    provider: llm.base.provider,
                    reasoning: llm.base.reasoning,
                    repetition_penalty: llm.base.repetition_penalty,
                    top_a: llm.base.top_a,
                    top_k: llm.base.top_k,
                    usage: request.usage,
                    verbosity: llm.base.verbosity,
                    models: llm.base.models,
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
        let mut next_chat_chunk = match stream.try_next().await {
            Ok(Some(chunk)) => Some(chunk),
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
            Ok(None) => {
                // chat client will always yield at least 1 chunk or error
                unreachable!()
            }
        };

        // the final chunk
        let mut final_chunk: Option<
            super::response::streaming::ChatCompletionChunk,
        > = None;

        // the aggregate of all chunks
        let mut aggregate: Option<
            super::response::streaming::ChatCompletionChunk,
        > = None;

        async_stream::stream! {
            while let Some(chat_chunk) = next_chat_chunk.take() {
                // fetch the next chat chunk or error
                let error = match stream.next().await {
                    Some(Ok(ncc)) => {
                        // set next chat chunk
                        next_chat_chunk = Some(ncc);
                        None
                    }
                    Some(Err(e)) => {
                        // end the loop after this iteration
                        // add error to choices
                        Some(crate::error::ResponseError::from(&e))
                    }
                    None => {
                        // end the loop after this iteration
                        None
                    }
                };
                // construct the score completions chunk from the chat completions chunk
                let mut chunk = super::response::streaming::ChatCompletionChunk {
                    id: response_id.clone(),
                    choices: chat_chunk.choices.into_iter().map(|choice| {
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
                                id: chat_chunk.id.clone(),
                                created: chat_chunk.created,
                                model: chat_chunk.model.clone(),
                                service_tier: chat_chunk.service_tier,
                                system_fingerprint: chat_chunk.system_fingerprint.clone(),
                                usage: chat_chunk.usage.clone(),
                                provider: chat_chunk.provider.clone(),
                            }),
                        }
                    }).collect(),
                    created,
                    model: request.model.unwrap_id().to_owned(),
                    object: chat::completions::response::streaming::Object::ChatCompletionChunk,
                    usage: None,
                    weight_data: None,
                };
                // convert tool calls to content if needed
                if matches!(
                    llm.base.output_mode,
                    score::llm::OutputMode::ToolCall,
                ) {
                    chunk.tool_as_content();
                }
                // push the chunk into the aggregate
                match aggregate {
                    Some(ref mut aggregate) => {
                        aggregate.push(&chunk);
                    }
                    None => {
                        aggregate = Some(chunk.clone());
                    }
                }
                // split off finished choices, to be yielded later
                match (&mut final_chunk, split_off_finished_choices(&mut chunk)) {
                    (Some(final_chunk), Some(next_final_chunk)) => {
                        final_chunk.push(&next_final_chunk);
                    }
                    (None, Some(next_final_chunk)) => {
                        final_chunk = Some(next_final_chunk);
                    }
                    (_, None) => {}
                }
                // yield chunk if it contains any unfinished choices
                if chunk.choices.len() > 0 {
                    yield chunk;
                }
            }

            let aggregate = aggregate.unwrap();
            let mut final_chunk = final_chunk.unwrap();

            // final chunk including votes
            for choice in &mut final_chunk.choices {
                let aggregate_choice = aggregate.choices.iter().find(|c| c.index == choice.index).unwrap();
                match get_vote(
                    pfx_tree.clone(),
                    &choices_key_pattern,
                    &choices_key_pattern_stripped,
                    choices_len,
                    aggregate_choice,
                ) {
                    Ok(vote) => choice.delta.vote = Some(vote),
                    Err(e) => {
                        if choice.error.is_none() {
                            choice.error = Some(crate::error::ResponseError::from(&e));
                            choice.finish_reason = Some(chat::completions::response::FinishReason::Error);
                        }
                    }
                }
            }
            yield final_chunk;
        }.boxed()
    }
}

async fn fetch_or_validate_score_model<CTX>(
    model_fetcher: Arc<impl score::model::Fetcher<CTX> + Send + Sync + 'static>,
    ctx: CTX,
    model_param: super::request::Model,
) -> Result<score::model::Model, super::Error> {
    match model_param {
        super::request::Model::Id(id) => {
            // 22 character id, fetch model
            if id.len() == 22 {
                model_fetcher
                    .fetch(ctx, &id)
                    .await
                    .map_err(|e| super::Error::FetchModel(e))
            } else {
                match id.split("/").last() {
                    // 22 character id with author prefix, fetch model
                    Some(slug) if slug.len() == 22 => model_fetcher
                        .fetch(ctx, slug)
                        .await
                        .map_err(|e| super::Error::FetchModel(e)),
                    // JSON string model, parse and validate
                    _ => {
                        match serde_json::from_str::<score::model::ModelBase>(
                            &id,
                        ) {
                            Ok(provided) => provided
                                .into_model_validate()
                                .map_err(|e| super::Error::InvalidModel(e)),
                            Err(_) => Err(super::Error::InvalidModel(id)),
                        }
                    }
                }
            }
        }
        // JSON body model, parse and validate
        super::request::Model::Provided(provided) => provided
            .into_model_validate()
            .map_err(|e| super::Error::InvalidModel(e)),
    }
}

pub async fn fetch_completion_futs_from_choices_and_messages<CTX: Clone>(
    completions_archive_fetcher: Arc<
        impl completions_archive::Fetcher<CTX> + Send + Sync + 'static,
    >,
    ctx: CTX,
    choices: &[super::request::Choice],
    messages: &[chat::completions::request::Message],
) -> Result<Vec<completions_archive::Completion>, crate::error::ResponseError> {
    // first, create a future for each unique completion in choices and messages
    let mut completions_futs = Vec::new();
    let mut ids = HashSet::new();
    for choice in choices {
        match choice {
            super::request::Choice::ChatCompletion { id, .. } => {
                if !ids.insert(id.as_str()) {
                    continue;
                }
                completions_futs.push(futures::future::Either::Left(
                    completions_archive_fetcher
                        .fetch_chat_completion(ctx.clone(), id)
                        .map_ok(completions_archive::Completion::Chat),
                ));
            }
            super::request::Choice::ScoreCompletion { id, .. } => {
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
            super::request::Choice::MultichatCompletion { id, .. } => {
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
    for message in messages {
        match message {
            chat::completions::request::Message::ChatCompletion(
                chat::completions::request::ChatCompletionMessage {
                    id, ..
                },
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
            chat::completions::request::Message::ScoreCompletion(
                chat::completions::request::ScoreCompletionMessage {
                    id, ..
                },
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
            chat::completions::request::Message::MultichatCompletion(
                chat::completions::request::MultichatCompletionMessage {
                    id,
                    ..
                },
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
        Ok(Vec::new())
    } else {
        futures::future::try_join_all(completions_futs).await
    }
}

// long name but you know what it does
pub fn replace_completion_messages_and_completion_choices_with_assistant_messages_and_text_choices(
    completions: Vec<completions_archive::Completion>,
    choices: &mut Vec<super::request::Choice>,
    messages: &mut Vec<chat::completions::request::Message>,
) -> Result<(), super::Error> {
    if completions.len() == 0 {
        return Ok(());
    }

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

    // replace completion choices with text choices
    for choice in choices {
        let (id, choice_index) = match choice {
            super::request::Choice::ChatCompletion {
                id, choice_index, ..
            } => (id, *choice_index),
            super::request::Choice::ScoreCompletion {
                id,
                choice_index,
                ..
            } => (id, *choice_index),
            _ => continue,
        };
        // return error if the choice_index is invalid
        let completion_choice_message_content = match id_to_completion
            [id.as_str()]
        {
            completions_archive::Completion::Chat(ref completion) => completion
                .choices
                .iter()
                .find(|choice| choice.index == choice_index)
                .and_then(|choice| match choice.message.content {
                    Some(ref content) if !content.is_empty() => {
                        Some(content.clone())
                    }
                    _ => None,
                }),
            completions_archive::Completion::Score(ref completion) => {
                completion
                    .choices
                    .iter()
                    .find(|choice| choice.index == choice_index)
                    .and_then(|choice| match choice.message.inner.content {
                        Some(ref content) if !content.is_empty() => {
                            Some(content.clone())
                        }
                        _ => None,
                    })
            }
            completions_archive::Completion::Multichat(ref completion) => {
                completion
                    .choices
                    .iter()
                    .find(|choice| choice.index == choice_index)
                    .and_then(|choice| match choice.message.content {
                        Some(ref content) if !content.is_empty() => {
                            Some(content.clone())
                        }
                        _ => None,
                    })
            }
        }
        .ok_or(super::Error::InvalidCompletionChoiceIndex(
            id.clone(),
            choice_index,
        ))?;
        // replace the completion choice with a text choice
        *choice =
            super::request::Choice::Text(completion_choice_message_content);
    }

    // replace completion messages with assistant message
    for message in messages {
        let (id, choice_index, name) = match message {
            chat::completions::request::Message::ChatCompletion(
                chat::completions::request::ChatCompletionMessage {
                    id,
                    choice_index,
                    name,
                },
            ) => (id, *choice_index, name.clone()),
            chat::completions::request::Message::ScoreCompletion(
                chat::completions::request::ScoreCompletionMessage {
                    id,
                    choice_index,
                    name,
                },
            ) => (id, *choice_index, name.clone()),
            chat::completions::request::Message::MultichatCompletion(
                chat::completions::request::MultichatCompletionMessage {
                    id,
                    choice_index,
                    name,
                },
            ) => (id, *choice_index, name.clone()),
            _ => continue,
        };
        // return error if the choice_index is invalid
        let completion_choice_message = match id_to_completion[id.as_str()] {
            completions_archive::Completion::Chat(ref completion) => completion
                .choices
                .iter()
                .find(|choice| choice.index == choice_index)
                .map(|choice| choice.message.clone()),
            completions_archive::Completion::Score(ref completion) => {
                completion
                    .choices
                    .iter()
                    .find(|choice| choice.index == choice_index)
                    .map(|choice| choice.message.inner.clone())
            }
            completions_archive::Completion::Multichat(ref completion) => {
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
        *message = chat::completions::request::Message::Assistant(
            chat::completions::convert_completion_choice_message_to_assistant_message(
                completion_choice_message,
                name,
            ),
        );
    }

    Ok(())
}

#[derive(Debug, serde::Deserialize)]
struct ResponseKey {
    _think: Option<String>,
    #[allow(dead_code)]
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

    fn get(&self, pfx: SelectPfx) -> Option<SelectPfxTree> {
        match self {
            SelectPfxTree::Branch(branch) => branch.get(&pfx).cloned(),
            SelectPfxTree::Leaf(_) => None,
        }
    }

    // fn is_leaf_branch(&self) -> bool {
    //     match self {
    //         SelectPfxTree::Branch(branch) => branch
    //             .values()
    //             .any(|v| matches!(v, SelectPfxTree::Leaf { .. })),
    //         SelectPfxTree::Leaf(_) => false,
    //     }
    // }

    fn depth(&self) -> usize {
        match self {
            SelectPfxTree::Branch(branch) => {
                1 + branch
                    .values()
                    .next() // all sub-branches have the same depth
                    .map(|v| v.depth())
                    .unwrap_or(0)
            }
            SelectPfxTree::Leaf(_) => 0,
        }
    }

    // fn unwrap_branch(self) -> Arc<IndexMap<SelectPfx, SelectPfxTree>> {
    //     match self {
    //         SelectPfxTree::Branch(branch) => branch,
    //         SelectPfxTree::Leaf(_) => panic!("Called unwrap_branch on a Leaf"),
    //     }
    // }

    fn unwrap_leaf(&self) -> usize {
        match self {
            SelectPfxTree::Leaf(index) => *index,
            SelectPfxTree::Branch(_) => {
                panic!("Called unwrap_leaf on a Branch")
            }
        }
    }

    fn json_serialize_select_choices(
        choices: &[super::request::Choice], // guaranteed all text
        indices: &[(String, usize)],
    ) -> String {
        struct OrderedChoices<'a> {
            indices: &'a [(String, usize)],
            choices: &'a [super::request::Choice],
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

    fn regex_patterns(&self, keys: &[String]) -> (String, String) {
        let depth = self.depth();
        let mut with_ticks = String::with_capacity(
            (keys.len() - 1) // '|' characters
                + (keys.len() * depth * 3) // each key
                + keys.len() * 2, // parentheses
        );
        let mut without_ticks = String::with_capacity(
            (keys.len() - 1) // for '|' characters
                + keys.len() * (depth * 3 - 2) // each key stripped of ticks
                + keys.len() * 2, // parentheses
        );
        for key in keys {
            if with_ticks.len() > 0 {
                with_ticks.push('|');
                without_ticks.push('|');
            }
            with_ticks.push('(');
            without_ticks.push('(');
            with_ticks.push_str(key);
            without_ticks.push_str(&key[1..key.len() - 1]); // strip ticks
            with_ticks.push(')');
            without_ticks.push(')');
        }
        (with_ticks, without_ticks)
    }
}

fn split_off_finished_choices(
    chunk: &mut super::response::streaming::ChatCompletionChunk,
) -> Option<super::response::streaming::ChatCompletionChunk> {
    if !chunk
        .choices
        .iter()
        .any(super::response::streaming::Choice::has_finish_reason_or_usage)
    {
        return None;
    }
    // initialize the finished chunk as a clone with no choices
    // capacity could be optimized
    let mut finished_chunk = chunk.clone_without_choices(chunk.choices.len());
    // prepare to replace chunk.choices with only unfinished choices
    // capacity could be optimized
    let mut unfinished_choices = Vec::with_capacity(chunk.choices.len());
    // distribute choices
    for choice in chunk.choices.drain(..) {
        if choice.has_finish_reason_or_usage() {
            finished_chunk.choices.push(choice);
        } else {
            unfinished_choices.push(choice);
        }
    }
    chunk.choices = unfinished_choices;
    Some(finished_chunk)
}

fn get_vote(
    mut pfx_tree: SelectPfxTree,
    with_ticks_pattern: &str,
    without_ticks_pattern: &str,
    choices_len: usize,
    choice: &super::response::streaming::Choice,
) -> Result<Vec<rust_decimal::Decimal>, super::Error> {
    // extract content, return if empty
    let content = match choice.delta.inner.content.as_ref() {
        Some(content) => Ok(content.as_str()),
        None => Err(super::Error::InvalidContent),
    }?;

    // extract response key, return if not found
    let with_ticks_re = Regex::new(with_ticks_pattern).unwrap();
    let mut key_match = with_ticks_re.find_iter(content).last();
    let without_ticks_re = match key_match {
        Some(_) => None,
        None => Some(Regex::new(without_ticks_pattern).unwrap()),
    };
    if key_match.is_none() {
        key_match = without_ticks_re
            .as_ref()
            .and_then(|re| re.find_iter(content).last());
    }
    let key = key_match
        .map(|cap| cap.as_str())
        .ok_or(super::Error::InvalidContent)?;

    // get the final prefix
    let (final_pfx_char, final_pfx) = key
        .chars()
        .rev()
        .map(|c| (c, SelectPfx::from_char(c)))
        .filter(|(_, pfx)| pfx.is_some())
        .next()
        .unwrap();
    let final_pfx = final_pfx.unwrap();

    // get to the lowest pfx tree branch
    let mut i = pfx_tree.depth() - 1;
    if i > 0 {
        for c in key.chars() {
            if let Some(pfx) = SelectPfx::from_char(c) {
                pfx_tree = pfx_tree.get(pfx).unwrap();
                i -= 1;
                if i == 0 {
                    break;
                }
            }
        }
    }
    let pfx_tree = match pfx_tree {
        SelectPfxTree::Branch(branch) => branch,
        SelectPfxTree::Leaf(_) => unreachable!(),
    };

    // prepare vote
    let mut vote = vec![rust_decimal::Decimal::ZERO; choices_len];

    // try to get probabilities from logprobs
    if let Some(chat::completions::response::Logprobs {
        content: Some(logprobs),
        ..
    }) = choice.logprobs.as_ref()
    {
        // reverse key to check against
        let key_rev = key.chars().rev().collect::<String>();
        // slice as we go
        let mut key_rev_slice = key_rev.as_str();
        // keep the relevant logprob
        let mut key_logprob = None;
        let mut key_logprob_index = 0;
        // find the logprob segment that matches the key
        'outer: for logprob in logprobs.iter().rev() {
            let mut i = logprob.token.len();
            for c in logprob.token.chars().rev() {
                i -= c.len_utf8();
                if key_rev_slice.starts_with(c) {
                    // match
                    // remove the matched char from the slice
                    key_rev_slice = &key_rev_slice[c.len_utf8()..];
                    // keep the logprob that contains the final pfx
                    if key_logprob.is_none() && c == final_pfx_char {
                        key_logprob = Some(logprob);
                        key_logprob_index = i;
                    }
                    // stop when the full match is found
                    if key_rev_slice.is_empty() {
                        break 'outer;
                    }
                } else if key_rev_slice.len() != key_rev.len() {
                    // not match
                    // reset
                    key_rev_slice = key_rev.as_str();
                    key_logprob = None;
                    key_logprob_index = 0;
                } else {
                    // unknown
                }
            }
        }
        if key_rev_slice.is_empty() {
            // get the probabilities
            let mut probability_sum = rust_decimal::Decimal::ZERO;
            for chat::completions::response::TopLogprob {
                token,
                logprob,
                ..
            } in &key_logprob.as_ref().unwrap().top_logprobs
            {
                if key_logprob_index < token.len()
                    && let Some(logprob) = logprob
                    && let Some((_, c)) = token
                        .char_indices()
                        .find(|(i, _)| *i == key_logprob_index)
                    && let Some(pfx) = SelectPfx::from_char(c)
                    && let Some(leaf) = pfx_tree.get(&pfx)
                {
                    let probability = logprob.exp();
                    vote[leaf.unwrap_leaf()] += probability;
                    probability_sum += probability;
                }
            }
            // normalize and return
            if probability_sum == rust_decimal::Decimal::ZERO {
                unreachable!()
            }
            for v in &mut vote {
                *v /= probability_sum;
            }
            return Ok(vote);
        }
    }

    // fallback, set vote indexed to selected choice to 1.0
    vote[pfx_tree.get(&final_pfx).unwrap().unwrap_leaf()] =
        rust_decimal::Decimal::ONE;
    Ok(vote)
}
