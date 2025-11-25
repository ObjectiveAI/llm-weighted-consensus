use envconfig::Envconfig;

#[derive(Envconfig)]
struct Config {
    #[envconfig(from = "BACKOFF_CURRENT_INTERVAL_MILLIS", default = "100")]
    backoff_current_interval_millis: u64,
    #[envconfig(from = "BACKOFF_INITIAL_INTERVAL_MILLIS", default = "100")]
    backoff_initial_interval_millis: u64,
    #[envconfig(from = "BACKOFF_RANDOMIZATION_FACTOR", default = "0.5")]
    backoff_randomization_factor: f64,
    #[envconfig(from = "BACKOFF_MULTIPLIER", default = "1.5")]
    backoff_multiplier: f64,
    #[envconfig(from = "BACKOFF_MAX_INTERVAL_MILLIS", default = "1000")]
    backoff_max_interval_millis: u64,
    #[envconfig(from = "BACKOFF_MAX_ELAPSED_TIME_MILLIS", default = "40000")]
    backoff_max_elapsed_time_millis: u64,
    #[envconfig(from = "FIRST_CHUNK_TIMEOUT_MILLIS", default = "10000")]
    first_chunk_timeout_millis: u64,
    #[envconfig(from = "OTHER_CHUNK_TIMEOUT_MILLIS", default = "60000")]
    other_chunk_timeout_millis: u64,
    #[envconfig(from = "OPENAI_API_BASE")]
    openai_api_base: Option<String>,
    #[envconfig(from = "OPENAI_API_KEY")]
    openai_api_key: Option<String>,
    #[envconfig(from = "OPENAI_APIS")]
    openai_apis: Option<String>,
    #[envconfig(from = "OPENAI_USER_AGENT")]
    openai_user_agent: Option<String>,
    #[envconfig(from = "OPENAI_X_TITLE")]
    openai_x_title: Option<String>,
    #[envconfig(from = "OPENAI_REFERER")]
    openai_referer: Option<String>,
    #[envconfig(from = "ADDRESS", default = "0.0.0.0")]
    address: String,
    #[envconfig(from = "PORT", default = "5000")]
    port: u16,
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::main]
async fn main() {
    use axum::{
        Json,
        response::{IntoResponse, Sse, sse::Event},
    };
    use llm_weighted_consensus::{
        chat::completions::Client, error::StatusError, *,
    };
    use std::{convert::Infallible, sync::Arc};
    use tokio_stream::StreamExt;

    // load .env file if present
    let _ = dotenv::dotenv();

    // initial config from environment
    let Config {
        backoff_current_interval_millis,
        backoff_initial_interval_millis,
        backoff_randomization_factor,
        backoff_multiplier,
        backoff_max_interval_millis,
        backoff_max_elapsed_time_millis,
        first_chunk_timeout_millis,
        other_chunk_timeout_millis,
        openai_api_base,
        openai_api_key,
        openai_apis,
        openai_user_agent,
        openai_x_title,
        openai_referer,
        address,
        port,
    } = Config::init_from_env().unwrap();

    // parse openai_apis if present
    let openai_apis = openai_apis
        .map(|s| serde_json::from_str::<Vec<chat::completions::ApiBase>>(&s))
        .transpose()
        .unwrap();

    // use static openai_api_base and openai_api_key if openai_apis not present
    let openai_apis = openai_apis.unwrap_or_else(|| {
        let (openai_api_base, openai_api_key) = match (
            openai_api_base,
            openai_api_key,
        ) {
            (Some(base), Some(key)) => (base, key),
            _ => panic!(
                "Either OPENAI_APIS or both OPENAI_API_BASE and OPENAI_API_KEY must be set",
            ),
        };
        vec![chat::completions::ApiBase {
            api_base: openai_api_base,
            api_key: openai_api_key,
        }]
    });

    let completions_archive_fetcher =
        Arc::new(completions_archive::UnimplementedFetcher);

    let chat_completions_client =
        Arc::new(chat::completions::DefaultClient::new(
            reqwest::Client::new(),
            backoff::ExponentialBackoff {
                current_interval: std::time::Duration::from_millis(
                    backoff_current_interval_millis,
                ),
                initial_interval: std::time::Duration::from_millis(
                    backoff_initial_interval_millis,
                ),
                randomization_factor: backoff_randomization_factor,
                multiplier: backoff_multiplier,
                max_interval: std::time::Duration::from_millis(
                    backoff_max_interval_millis,
                ),
                start_time: std::time::Instant::now(),
                max_elapsed_time: Some(std::time::Duration::from_millis(
                    backoff_max_elapsed_time_millis,
                )),
                clock: backoff::SystemClock::default(),
            },
            openai_apis,
            openai_user_agent,
            openai_x_title,
            openai_referer,
            std::time::Duration::from_millis(first_chunk_timeout_millis),
            std::time::Duration::from_millis(other_chunk_timeout_millis),
            Arc::new(chat::completions::NoOpCtxHandler::new()),
            completions_archive_fetcher.clone(),
        ));

    let score_completions_client = Arc::new(score::completions::Client::new(
        chat_completions_client.clone(),
        Arc::new(score::model::UnimplementedFetcher),
        Arc::new(score::completions::weight::Fetchers::new(
            score::completions::weight::StaticFetcher,
            score::completions::weight::UnimplementedTrainingTableFetcher,
        )),
        completions_archive_fetcher,
    ));

    let app = axum::Router::new()
        // chat completions
        .route(
            "/chat/completions",
            axum::routing::post({
                let chat_completions_client = chat_completions_client.clone();
                move |Json(request): Json<chat::completions::request::ChatCompletionCreateParams>| {
                    let chat_completions_client = chat_completions_client.clone();
                    async move {
                        if request.stream.unwrap_or(false) {
                            match chat_completions_client.create_streaming((), request).await {
                                Ok(stream) => Sse::new(
                                    stream.map(|result| {
                                        Ok::<Event, Infallible>(
                                            Event::default().data(
                                                match result {
                                                    Ok(chunk) => serde_json::to_string(&chunk),
                                                    Err(e) => serde_json::to_string(&error::ResponseError::from(&e)),
                                                }
                                                    .unwrap_or_default(),
                                            ),
                                        )
                                    })
                                    .chain(util::StreamOnce::new(Ok(Event::default().data("[DONE]"))))
                                )
                                    .into_response(),
                                Err(e) => (
                                    axum::http::StatusCode::from_u16(e.status()).unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
                                    serde_json::to_string(&e.message()).unwrap_or_default(),
                                )
                                    .into_response(),
                            }
                        } else {
                            match chat_completions_client.create_unary((), request).await {
                                Ok(response) => Json(response).into_response(),
                                Err(e) => (
                                    axum::http::StatusCode::from_u16(e.status()).unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
                                    serde_json::to_string(&e.message()).unwrap_or_default(),
                                )
                                    .into_response()
                            }
                        }
                    }
                }
            })
        )
        // score completions
        .route(
            "/score/completions",
            axum::routing::post({
                let score_completions_client = score_completions_client.clone();
                move |Json(request): Json<score::completions::request::ChatCompletionCreateParams>| {
                    let score_completions_client = score_completions_client.clone();
                    async move {
                        if request.stream.unwrap_or(false) {
                            match score_completions_client.create_streaming((), request).await {
                                Ok(stream) => Sse::new(
                                    stream.map(|result| {
                                        Ok::<Event, Infallible>(
                                            Event::default().data(
                                                match result {
                                                    Ok(chunk) => serde_json::to_string(&chunk),
                                                    Err(e) => serde_json::to_string(&error::ResponseError::from(&e)),
                                                }
                                                    .unwrap_or_default(),
                                            ),
                                        )
                                    })
                                    .chain(util::StreamOnce::new(Ok(Event::default().data("[DONE]"))))
                                )
                                    .into_response(),
                                Err(e) => (
                                    axum::http::StatusCode::from_u16(e.status()).unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
                                    serde_json::to_string(&e.message()).unwrap_or_default(),
                                )
                                    .into_response(),
                            }
                        } else {
                            match score_completions_client.create_unary((), request).await {
                                Ok(response) => Json(response).into_response(),
                                Err(e) => (
                                    axum::http::StatusCode::from_u16(e.status()).unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
                                    serde_json::to_string(&e.message()).unwrap_or_default(),
                                )
                                    .into_response()
                            }
                        }
                    }
                }
            })
        );

    let listener =
        tokio::net::TcpListener::bind(format!("{}:{}", address, port))
            .await
            .unwrap();

    axum::serve(listener, app).await.unwrap();
}

#[cfg(target_arch = "wasm32")]
fn main() {}
