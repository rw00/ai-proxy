use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use serde_json::json;
use url::Url;

use crate::config::AppConfig;
use crate::pool::{KeyPool, SelectError};
use crate::provider::{
    build_adapter, filter_request_headers, filter_response_headers, ProviderAdapter,
    UpstreamRequest,
};
use crate::quota::{classify_response, ClassifiedResponse};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub pool: Arc<KeyPool>,
    pub adapters: Arc<HashMap<String, Arc<dyn ProviderAdapter>>>,
    pub http_client: reqwest::Client,
    pub enforce_loopback_origin: bool,
}

impl AppState {
    pub fn new(config: AppConfig, pool: KeyPool) -> Self {
        let mut adapters = HashMap::new();
        for (name, pcfg) in &config.providers {
            adapters.insert(name.clone(), build_adapter(pcfg));
        }
        let enforce_loopback_origin = config.is_loopback_bind();
        let http_client = reqwest::Client::builder()
            .pool_max_idle_per_host(16)
            .build()
            .expect("failed to build reqwest client");

        Self {
            config: Arc::new(config),
            pool: Arc::new(pool),
            adapters: Arc::new(adapters),
            http_client,
            enforce_loopback_origin,
        }
    }
}

pub struct ForwardParams<'a> {
    pub provider_name: &'a str,
    pub upstream_url: Url,
    pub method: Method,
    pub incoming_headers: &'a HeaderMap,
    pub body: Bytes,
    pub model: Option<String>,
    pub is_openai_compat: bool,
}

pub async fn forward_request(state: &AppState, params: ForwardParams<'_>) -> Response {
    let ForwardParams {
        provider_name,
        upstream_url,
        method,
        incoming_headers,
        body,
        model,
        is_openai_compat,
    } = params;
    let Some(pcfg) = state.pool.provider(provider_name) else {
        return json_error(
            StatusCode::NOT_FOUND,
            "unknown_provider",
            &format!("Provider '{provider_name}' is not configured"),
        );
    };
    let Some(adapter) = state.adapters.get(provider_name) else {
        return json_error(
            StatusCode::NOT_FOUND,
            "unknown_provider",
            &format!("Adapter for provider '{provider_name}' not found"),
        );
    };

    let filtered_headers = filter_request_headers(incoming_headers);
    let max_attempts = pcfg.keys.len().max(1);
    let mut tried_keys: HashSet<String> = HashSet::new();

    for _ in 0..=max_attempts {
        let now = Utc::now();
        let selected = match state
            .pool
            .select_key(provider_name, model.as_deref(), &tried_keys, now)
        {
            Ok(k) => k,
            Err(SelectError::NoKeysConfigured) => {
                return json_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "no_keys_configured",
                    &format!("No API keys configured for provider '{provider_name}'"),
                );
            }
            Err(SelectError::UnknownProvider) => {
                return json_error(
                    StatusCode::NOT_FOUND,
                    "unknown_provider",
                    &format!("Provider '{provider_name}' is not configured"),
                );
            }
            Err(SelectError::AllExhausted { next_reset_at }) => {
                return exhausted_response(provider_name, model.as_deref(), next_reset_at, now);
            }
        };

        let mut upstream_req = UpstreamRequest {
            url: upstream_url.clone(),
            headers: filtered_headers.clone(),
        };
        adapter.apply_auth(&mut upstream_req, selected.secret.expose(), is_openai_compat);

        let send_res = state
            .http_client
            .request(method.clone(), upstream_req.url)
            .headers(upstream_req.headers)
            .body(body.clone())
            .send()
            .await;

        let resp = match send_res {
            Ok(r) => r,
            Err(err) => {
                state.pool.record_error(provider_name, &selected.key_id);
                tracing::warn!(
                    provider = provider_name,
                    key_id = %selected.key_id,
                    error = %err,
                    "upstream transport error"
                );
                return json_error(
                    StatusCode::BAD_GATEWAY,
                    "upstream_error",
                    &format!("Upstream request failed: {err}"),
                );
            }
        };

        let status = resp.status();
        let resp_headers = filter_response_headers(resp.headers());

        // Stream 2xx and non-retryable statuses directly without buffering the response body.
        if !matches!(
            status,
            StatusCode::UNAUTHORIZED
                | StatusCode::FORBIDDEN
                | StatusCode::TOO_MANY_REQUESTS
                | StatusCode::SERVICE_UNAVAILABLE
        ) {
            if status.is_success() {
                state
                    .pool
                    .record_success(provider_name, &selected.key_id, model.as_deref())
                    .await;
            } else {
                state.pool.record_error(provider_name, &selected.key_id);
            }
            let stream = resp.bytes_stream();
            let mut builder = Response::builder().status(status);
            for (k, v) in resp_headers {
                if let Some(name) = k {
                    builder = builder.header(name, v);
                }
            }
            return builder
                .body(Body::from_stream(stream))
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response());
        }

        // Buffer only potential quota/auth error bodies so we can classify and rotate.
        let err_body = match resp.bytes().await {
            Ok(b) => b,
            Err(err) => {
                state.pool.record_error(provider_name, &selected.key_id);
                return json_error(
                    StatusCode::BAD_GATEWAY,
                    "upstream_read_error",
                    &format!("Failed to read upstream error body: {err}"),
                );
            }
        };

        let classify_now = Utc::now();
        // Peek prior consecutive hits for backoff calculation.
        let prior_hits = state.pool.previous_consecutive_hits(
            provider_name,
            &selected.key_id,
            model.as_deref(),
            crate::provider::QuotaScope::Model,
        );

        match classify_response(
            adapter.as_ref(),
            pcfg,
            status,
            &resp_headers,
            &err_body,
            classify_now,
            prior_hits,
        ) {
            ClassifiedResponse::Quota {
                scope,
                reset_at,
                reason,
            } => {
                tracing::warn!(
                    provider = provider_name,
                    key_id = %selected.key_id,
                    model = ?model,
                    status = status.as_u16(),
                    reset_at = %reset_at.to_rfc3339(),
                    reason = %reason,
                    "key quota/rate limited; rotating"
                );
                state
                    .pool
                    .record_quota_hit(
                        provider_name,
                        &selected.key_id,
                        model.as_deref(),
                        scope,
                        reset_at,
                        reason,
                    )
                    .await;
                tried_keys.insert(selected.key_id);
                continue;
            }
            ClassifiedResponse::InvalidKey { reason } => {
                tracing::warn!(
                    provider = provider_name,
                    key_id = %selected.key_id,
                    status = status.as_u16(),
                    reason = %reason,
                    "marking key invalid; rotating"
                );
                state
                    .pool
                    .record_invalid_key(provider_name, &selected.key_id, reason)
                    .await;
                tried_keys.insert(selected.key_id);
                continue;
            }
            ClassifiedResponse::Passthrough => {
                state.pool.record_error(provider_name, &selected.key_id);
                let mut builder = Response::builder().status(status);
                for (k, v) in resp_headers {
                    if let Some(name) = k {
                        builder = builder.header(name, v);
                    }
                }
                return builder
                    .body(Body::from(err_body))
                    .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response());
            }
        }
    }

    exhausted_response(provider_name, model.as_deref(), None, Utc::now())
}

fn exhausted_response(
    provider: &str,
    model: Option<&str>,
    next_reset_at: Option<DateTime<Utc>>,
    now: DateTime<Utc>,
) -> Response {
    let retry_after_secs = next_reset_at
        .map(|dt| (dt - now).num_seconds().max(1))
        .unwrap_or(60);

    let payload = json!({
        "error": {
            "type": "all_keys_exhausted",
            "message": format!("All API keys for provider '{provider}' are exhausted or invalid"),
            "provider": provider,
            "model": model,
            "next_reset_at": next_reset_at.map(|dt| dt.to_rfc3339()),
            "retry_after_seconds": retry_after_secs,
        }
    });

    let mut resp = (StatusCode::TOO_MANY_REQUESTS, Json(payload)).into_response();
    if let Ok(val) = HeaderValue::from_str(&retry_after_secs.to_string()) {
        resp.headers_mut().insert(http::header::RETRY_AFTER, val);
    }
    resp
}

pub fn json_error(status: StatusCode, err_type: &str, message: &str) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "type": err_type,
                "message": message,
            }
        })),
    )
        .into_response()
}
