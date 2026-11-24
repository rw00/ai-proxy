use axum::body::Bytes;
use axum::extract::{Path, Query, Request, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Json, Router};
use chrono::Utc;
use serde::Deserialize;
use serde_json::{json, Value};
use url::Url;

use crate::proxy::{forward_request, json_error, AppState, ForwardParams};

pub fn build_router(state: AppState) -> Router {
    let protected = Router::new()
        .route("/status", get(status_handler))
        .route("/status/reset", post(status_reset_handler))
        .route("/p/{provider}/{*path}", any(passthrough_handler))
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/embeddings", post(openai_embeddings))
        .route("/v1/models", get(openai_models))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            api_token_middleware,
        ));

    let app = Router::new()
        .route("/health", get(health_handler))
        .merge(protected)
        .with_state(state.clone());

    if state.enforce_loopback_origin {
        app.layer(middleware::from_fn(loopback_security_middleware))
    } else {
        app
    }
}

async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({ "status": "ok" })))
}

async fn status_handler(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.pool.status_snapshot(Utc::now()))
}

#[derive(Debug, Default, Deserialize)]
struct ResetRequest {
    provider: Option<String>,
    key_id: Option<String>,
}

async fn status_reset_handler(
    State(state): State<AppState>,
    Query(query): Query<ResetRequest>,
    body: Bytes,
) -> Response {
    let body_req: ResetRequest = if body.is_empty() {
        ResetRequest::default()
    } else {
        match serde_json::from_slice(&body) {
            Ok(r) => r,
            Err(err) => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    &format!("Invalid JSON body: {err}"),
                );
            }
        }
    };

    let provider = body_req.provider.or(query.provider);
    let key_id = body_req.key_id.or(query.key_id);

    match state
        .pool
        .reset(provider.as_deref(), key_id.as_deref())
        .await
    {
        Ok(cleared) => (
            StatusCode::OK,
            Json(json!({
                "status": "ok",
                "cleared": cleared,
                "provider": provider,
                "key_id": key_id,
            })),
        )
            .into_response(),
        Err(err) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "store_error",
            &format!("Failed to reset cooldowns: {err}"),
        ),
    }
}

async fn passthrough_handler(
    State(state): State<AppState>,
    Path((provider, path)): Path<(String, String)>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Some(pcfg) = state.pool.provider(&provider) else {
        return json_error(
            StatusCode::NOT_FOUND,
            "unknown_provider",
            &format!("Provider '{provider}' is not configured"),
        );
    };
    let Some(adapter) = state.adapters.get(&provider) else {
        return json_error(
            StatusCode::NOT_FOUND,
            "unknown_provider",
            &format!("Adapter for provider '{provider}' not found"),
        );
    };

    let rel_path = path.trim_start_matches('/');
    let mut url_str = format!("{}/{}", pcfg.base_url.trim_end_matches('/'), rel_path);
    if let Some(q) = uri.query() {
        url_str.push('?');
        url_str.push_str(q);
    }
    let upstream_url = match Url::parse(&url_str) {
        Ok(u) => u,
        Err(err) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_url",
                &format!("Invalid upstream URL: {err}"),
            );
        }
    };

    let model = adapter.extract_model(rel_path, &body);
    forward_request(
        &state,
        ForwardParams {
            provider_name: &provider,
            upstream_url,
            method,
            incoming_headers: &headers,
            body,
            model,
            is_openai_compat: false,
        },
    )
    .await
}

async fn openai_chat_completions(
    State(state): State<AppState>,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    openai_post_route(&state, "chat/completions", uri, headers, body).await
}

async fn openai_embeddings(
    State(state): State<AppState>,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    openai_post_route(&state, "embeddings", uri, headers, body).await
}

#[derive(Debug, Default, Deserialize)]
struct ModelsQuery {
    provider: Option<String>,
}

async fn openai_models(
    State(state): State<AppState>,
    Query(q): Query<ModelsQuery>,
    headers: HeaderMap,
) -> Response {
    let provider = match q
        .provider
        .or_else(|| resolve_default_provider(&state))
    {
        Some(p) => p,
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "provider_unresolved",
                "Specify ?provider=<name> or configure server.default_provider",
            );
        }
    };

    let Some(pcfg) = state.pool.provider(&provider) else {
        return json_error(
            StatusCode::NOT_FOUND,
            "unknown_provider",
            &format!("Provider '{provider}' is not configured"),
        );
    };
    let Some(adapter) = state.adapters.get(&provider) else {
        return json_error(
            StatusCode::NOT_FOUND,
            "unknown_provider",
            &format!("Adapter for provider '{provider}' not found"),
        );
    };

    let url_str = adapter.openai_compat_url(&pcfg.base_url, &pcfg.openai_compat_path, "models");
    let upstream_url = match Url::parse(&url_str) {
        Ok(u) => u,
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "invalid_url",
                &format!("Invalid upstream URL: {err}"),
            );
        }
    };

    forward_request(
        &state,
        ForwardParams {
            provider_name: &provider,
            upstream_url,
            method: Method::GET,
            incoming_headers: &headers,
            body: Bytes::new(),
            model: None,
            is_openai_compat: true,
        },
    )
    .await
}

async fn openai_post_route(
    state: &AppState,
    endpoint: &str,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let mut parsed_json: Option<Value> = serde_json::from_slice(&body).ok();
    let raw_model = parsed_json
        .as_ref()
        .and_then(|v| v.get("model"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    let (provider, resolved_model, rewritten_body) = match resolve_provider_and_model(
        state,
        raw_model.as_deref(),
        &mut parsed_json,
        body,
    ) {
        Ok(v) => v,
        Err(resp) => return *resp,
    };

    let Some(pcfg) = state.pool.provider(&provider) else {
        return json_error(
            StatusCode::NOT_FOUND,
            "unknown_provider",
            &format!("Provider '{provider}' is not configured"),
        );
    };
    let Some(adapter) = state.adapters.get(&provider) else {
        return json_error(
            StatusCode::NOT_FOUND,
            "unknown_provider",
            &format!("Adapter for provider '{provider}' not found"),
        );
    };

    let mut url_str =
        adapter.openai_compat_url(&pcfg.base_url, &pcfg.openai_compat_path, endpoint);
    if let Some(q) = uri.query() {
        url_str.push('?');
        url_str.push_str(q);
    }
    let upstream_url = match Url::parse(&url_str) {
        Ok(u) => u,
        Err(err) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "invalid_url",
                &format!("Invalid upstream URL: {err}"),
            );
        }
    };

    forward_request(
        state,
        ForwardParams {
            provider_name: &provider,
            upstream_url,
            method: Method::POST,
            incoming_headers: &headers,
            body: rewritten_body,
            model: resolved_model,
            is_openai_compat: true,
        },
    )
    .await
}

fn resolve_provider_and_model(
    state: &AppState,
    raw_model: Option<&str>,
    parsed_json: &mut Option<Value>,
    original_body: Bytes,
) -> Result<(String, Option<String>, Bytes), Box<Response>> {
    if let Some(model) = raw_model {
        // 1. Explicit `provider/model` prefix
        if let Some((prefix, rest)) = model.split_once('/') {
            if !rest.is_empty() && state.config.providers.contains_key(prefix) {
                let mut out_body = original_body;
                if let Some(Value::Object(map)) = parsed_json {
                    map.insert("model".to_string(), Value::String(rest.to_string()));
                    if let Ok(bytes) = serde_json::to_vec(map) {
                        out_body = Bytes::from(bytes);
                    }
                }
                return Ok((prefix.to_string(), Some(rest.to_string()), out_body));
            }
        }

        // 2. Explicit routing.model_map
        if let Some(mapped_provider) = state.config.routing.model_map.get(model) {
            return Ok((
                mapped_provider.clone(),
                Some(model.to_string()),
                original_body,
            ));
        }
    }

    // 3. Configured default_provider or sole configured provider
    if let Some(default_provider) = resolve_default_provider(state) {
        return Ok((
            default_provider,
            raw_model.map(str::to_string),
            original_body,
        ));
    }

    Err(Box::new(json_error(
        StatusCode::BAD_REQUEST,
        "provider_unresolved",
        "Could not resolve target provider: use 'provider/model' prefix, routing.model_map, or server.default_provider",
    )))
}

fn resolve_default_provider(state: &AppState) -> Option<String> {
    if let Some(def) = &state.config.server.default_provider {
        if !def.is_empty() {
            return Some(def.clone());
        }
    }
    if state.config.providers.len() == 1 {
        return state.config.providers.keys().next().cloned();
    }
    None
}

async fn api_token_middleware(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    let Some(expected) = state.config.server.api_token.as_deref() else {
        return next.run(req).await;
    };

    let provided = req
        .headers()
        .get(http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer ").or_else(|| s.strip_prefix("bearer ")));

    match provided {
        Some(token) if constant_time_eq(token.as_bytes(), expected.as_bytes()) => {
            next.run(req).await
        }
        _ => json_error(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "Missing or invalid Bearer API token",
        ),
    }
}

async fn loopback_security_middleware(req: Request, next: Next) -> Response {
    if let Some(host_val) = req
        .headers()
        .get(http::header::HOST)
        .and_then(|v| v.to_str().ok())
    {
        if !is_loopback_host(host_val) {
            return json_error(
                StatusCode::FORBIDDEN,
                "forbidden_host",
                "Non-loopback Host header rejected",
            );
        }
    }

    if let Some(origin_val) = req
        .headers()
        .get(http::header::ORIGIN)
        .and_then(|v| v.to_str().ok())
    {
        let allowed = Url::parse(origin_val)
            .ok()
            .and_then(|u| u.host_str().map(str::to_string))
            .is_some_and(|h| is_loopback_hostname(&h));
        if !allowed {
            return json_error(
                StatusCode::FORBIDDEN,
                "forbidden_origin",
                "Non-loopback Origin header rejected",
            );
        }
    }

    next.run(req).await
}

fn is_loopback_host(host_header: &str) -> bool {
    let trimmed = host_header.trim();
    if let Some(bracketed) = trimmed.strip_prefix('[') {
        if let Some((ip, _)) = bracketed.split_once(']') {
            return is_loopback_hostname(ip);
        }
        return false;
    }
    let hostname = trimmed
        .rsplit_once(':')
        .map(|(h, _)| h)
        .unwrap_or(trimmed);
    is_loopback_hostname(hostname)
}

fn is_loopback_hostname(host: &str) -> bool {
    let h = host.trim().trim_start_matches('[').trim_end_matches(']');
    h.eq_ignore_ascii_case("localhost") || h == "127.0.0.1" || h == "::1"
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}
