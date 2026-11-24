use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, Method, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::any;
use axum::Router;
use futures_util::stream;
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tower::ServiceExt;

use ai_proxy::config::{
    AuthMode, ProviderConfig, ProviderPreset, RawConfig, RoutingConfig, SelectionStrategy,
    ServerConfig,
};
use ai_proxy::{build_router, AppConfig, AppState, KeyPool, Store};

type CallRecord = (String, String, Option<String>);

#[derive(Clone, Default)]
struct MockCounters {
    calls: Arc<Mutex<Vec<CallRecord>>>,
}

impl MockCounters {
    fn record(&self, path: String, key: String, model: Option<String>) {
        self.calls.lock().unwrap().push((path, key, model));
    }

    fn snapshot(&self) -> Vec<CallRecord> {
        self.calls.lock().unwrap().clone()
    }
}

async fn spawn_mock_upstream(router: Router) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

fn build_app(raw: RawConfig, db_path: &Path) -> Router {
    let cfg = AppConfig::from_raw(raw, &[]).unwrap();
    let store = Store::open(db_path).unwrap();
    let pool = KeyPool::new(cfg.providers.clone(), store).unwrap();
    let state = AppState::new(cfg, pool);
    build_router(state)
}

async fn read_body_bytes(resp: Response) -> Bytes {
    axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap()
}

fn extract_upstream_key(headers: &HeaderMap) -> String {
    if let Some(v) = headers.get("x-goog-api-key").and_then(|v| v.to_str().ok()) {
        return v.to_string();
    }
    if let Some(v) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
        return v.strip_prefix("Bearer ").unwrap_or(v).to_string();
    }
    if let Some(v) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        return v.to_string();
    }
    String::new()
}

#[tokio::test]
async fn multi_key_failover_unary_and_sse_streaming() {
    let counters = MockCounters::default();
    let mock_router = Router::new()
        .fallback(any(
            |State(c): State<MockCounters>,
             uri: axum::http::Uri,
             headers: HeaderMap,
             body: Bytes| async move {
                let key = extract_upstream_key(&headers);
                let path = uri.path().to_string();
                let is_sse = uri.query().unwrap_or("").contains("alt=sse");
                // Ensure dummy caller key was stripped
                assert_ne!(key, "dummy-caller-key");
                c.record(path.clone(), key.clone(), None);

                if key == "gem-key-1" {
                    let mut resp = (
                        StatusCode::TOO_MANY_REQUESTS,
                        axum::Json(json!({
                            "error": {
                                "code": 429,
                                "status": "RESOURCE_EXHAUSTED",
                                "message": "Rate limit exceeded",
                                "details": [
                                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "45s"}
                                ]
                            }
                        })),
                    )
                        .into_response();
                    resp.headers_mut()
                        .insert("x-test-hit", HeaderValue::from_static("1"));
                    return resp;
                }

                if is_sse {
                    let chunks: Vec<Result<Bytes, std::io::Error>> = vec![
                        Ok(Bytes::from("data: {\"chunk\":1}\n\n")),
                        Ok(Bytes::from("data: {\"chunk\":2}\n\n")),
                    ];
                    return Response::builder()
                        .status(StatusCode::OK)
                        .header("content-type", "text/event-stream")
                        .body(Body::from_stream(stream::iter(chunks)))
                        .unwrap();
                }

                // Echo request body back to prove lossless request retry
                (
                    StatusCode::OK,
                    axum::Json(json!({
                        "used_key": key,
                        "echo": String::from_utf8_lossy(&body),
                    })),
                )
                    .into_response()
            },
        ))
        .with_state(counters.clone());

    let base_url = spawn_mock_upstream(mock_router).await;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("proxy.db");

    let mut raw = RawConfig::default();
    raw.providers.insert(
        "gemini".into(),
        ProviderConfig {
            preset: ProviderPreset::Gemini,
            base_url: Some(base_url),
            strategy: SelectionStrategy::Sticky,
            keys: vec!["gem-key-1".into(), "gem-key-2".into()],
            ..Default::default()
        },
    );
    let app = build_app(raw, &db_path);

    // 1. Unary request fails over from gem-key-1 (429) -> gem-key-2 (200) losslessly
    let req = Request::builder()
        .method(Method::POST)
        .uri("/p/gemini/v1beta/models/gemini-2.5-pro:generateContent?key=dummy-caller-key")
        .header("host", "127.0.0.1:8080")
        .header("x-goog-api-key", "dummy-caller-key")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"contents":[{"text":"hello"}]}"#))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json_body: Value = serde_json::from_slice(&read_body_bytes(resp).await).unwrap();
    assert_eq!(json_body["used_key"], "gem-key-2");
    assert_eq!(json_body["echo"], r#"{"contents":[{"text":"hello"}]}"#);

    // 2. SSE streaming request on a fresh model triggers failover on gem-key-1 and streams SSE from gem-key-2
    let sse_req = Request::builder()
        .method(Method::POST)
        .uri("/p/gemini/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse")
        .header("host", "127.0.0.1:8080")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"contents":[{"text":"stream"}]}"#))
        .unwrap();
    let sse_resp = app.clone().oneshot(sse_req).await.unwrap();
    assert_eq!(sse_resp.status(), StatusCode::OK);
    assert_eq!(
        sse_resp.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    let sse_text = String::from_utf8(read_body_bytes(sse_resp).await.to_vec()).unwrap();
    assert!(sse_text.contains("data: {\"chunk\":1}"));
    assert!(sse_text.contains("data: {\"chunk\":2}"));
}

#[tokio::test]
async fn per_model_cooldown_isolation_on_same_key() {
    let mock_router = Router::new().fallback(any(|headers: HeaderMap, body: Bytes| async move {
        let key = extract_upstream_key(&headers);
        let parsed: Value = serde_json::from_slice(&body).unwrap_or(json!({}));
        let model = parsed["model"].as_str().unwrap_or("");
        if key == "oa-key-1" && model == "model-a" {
            let mut resp = (
                StatusCode::TOO_MANY_REQUESTS,
                axum::Json(json!({"error": {"message": "limit on model-a"}})),
            )
                .into_response();
            resp.headers_mut()
                .insert("retry-after", HeaderValue::from_static("120"));
            return resp;
        }
        (
            StatusCode::OK,
            axum::Json(json!({"key": key, "model": model})),
        )
            .into_response()
    }));

    let base_url = spawn_mock_upstream(mock_router).await;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("iso.db");

    let mut raw = RawConfig::default();
    raw.server.default_provider = Some("openai".into());
    raw.providers.insert(
        "openai".into(),
        ProviderConfig {
            preset: ProviderPreset::OpenAiCompat,
            base_url: Some(base_url),
            auth: Some(AuthMode::Bearer),
            strategy: SelectionStrategy::Sticky,
            keys: vec!["oa-key-1".into(), "oa-key-2".into()],
            ..Default::default()
        },
    );
    let app = build_app(raw, &db_path);

    // model-a hits 429 on oa-key-1 -> fails over to oa-key-2
    let req_a = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("host", "127.0.0.1:8080")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"model-a","messages":[]}"#))
        .unwrap();
    let resp_a = app.clone().oneshot(req_a).await.unwrap();
    assert_eq!(resp_a.status(), StatusCode::OK);
    let body_a: Value = serde_json::from_slice(&read_body_bytes(resp_a).await).unwrap();
    assert_eq!(body_a["key"], "oa-key-2");

    // model-b still uses oa-key-1 because cooldown is isolated per model
    let req_b = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("host", "127.0.0.1:8080")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"model-b","messages":[]}"#))
        .unwrap();
    let resp_b = app.clone().oneshot(req_b).await.unwrap();
    assert_eq!(resp_b.status(), StatusCode::OK);
    let body_b: Value = serde_json::from_slice(&read_body_bytes(resp_b).await).unwrap();
    assert_eq!(body_b["key"], "oa-key-1");
}

#[tokio::test]
async fn invalid_key_401_fails_over_and_never_retries_bad_key() {
    let counters = MockCounters::default();
    let mock_router = Router::new()
        .fallback(any(
            |State(c): State<MockCounters>, headers: HeaderMap| async move {
                let key = extract_upstream_key(&headers);
                c.record("/v1/chat/completions".into(), key.clone(), None);
                if key == "bad-key-1" {
                    return (
                        StatusCode::UNAUTHORIZED,
                        axum::Json(json!({"error": {"message": "Invalid API key"}})),
                    )
                        .into_response();
                }
                (StatusCode::OK, axum::Json(json!({"key": key}))).into_response()
            },
        ))
        .with_state(counters.clone());

    let base_url = spawn_mock_upstream(mock_router).await;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("inv.db");

    let mut raw = RawConfig::default();
    raw.server.default_provider = Some("openai".into());
    raw.providers.insert(
        "openai".into(),
        ProviderConfig {
            preset: ProviderPreset::OpenAiCompat,
            base_url: Some(base_url),
            strategy: SelectionStrategy::RoundRobin,
            keys: vec!["bad-key-1".into(), "good-key-2".into()],
            ..Default::default()
        },
    );
    let app = build_app(raw, &db_path);

    for _ in 0..3 {
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("host", "127.0.0.1:8080")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"gpt-4o","messages":[]}"#))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: Value = serde_json::from_slice(&read_body_bytes(resp).await).unwrap();
        assert_eq!(v["key"], "good-key-2");
    }

    let bad_hits = counters
        .snapshot()
        .iter()
        .filter(|(_, k, _)| k == "bad-key-1")
        .count();
    assert_eq!(bad_hits, 1, "invalid key must not be retried after 401");
}

#[tokio::test]
async fn non_quota_400_passes_through_without_rotation() {
    let counters = MockCounters::default();
    let mock_router = Router::new()
        .fallback(any(
            |State(c): State<MockCounters>, headers: HeaderMap| async move {
                let key = extract_upstream_key(&headers);
                c.record("/v1/chat/completions".into(), key, None);
                (
                    StatusCode::BAD_REQUEST,
                    axum::Json(json!({"error": {"message": "invalid prompt parameter"}})),
                )
                    .into_response()
            },
        ))
        .with_state(counters.clone());

    let base_url = spawn_mock_upstream(mock_router).await;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("badreq.db");

    let mut raw = RawConfig::default();
    raw.server.default_provider = Some("openai".into());
    raw.providers.insert(
        "openai".into(),
        ProviderConfig {
            preset: ProviderPreset::OpenAiCompat,
            base_url: Some(base_url),
            keys: vec!["k1".into(), "k2".into()],
            ..Default::default()
        },
    );
    let app = build_app(raw, &db_path);

    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("host", "127.0.0.1:8080")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"gpt-4o"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: Value = serde_json::from_slice(&read_body_bytes(resp).await).unwrap();
    assert_eq!(v["error"]["message"], "invalid prompt parameter");
    assert_eq!(
        counters.snapshot().len(),
        1,
        "400 must not trigger rotation to second key"
    );
}

#[tokio::test]
async fn all_keys_exhausted_returns_429_with_earliest_retry_after() {
    let mock_router = Router::new().fallback(any(|headers: HeaderMap| async move {
        let key = extract_upstream_key(&headers);
        let retry_secs = if key == "k-slow" { "120" } else { "30" };
        let mut resp = (
            StatusCode::TOO_MANY_REQUESTS,
            axum::Json(json!({"error": {"message": "rate limited"}})),
        )
            .into_response();
        resp.headers_mut()
            .insert("retry-after", HeaderValue::from_str(retry_secs).unwrap());
        resp
    }));

    let base_url = spawn_mock_upstream(mock_router).await;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("exhausted.db");

    let mut raw = RawConfig::default();
    raw.server.default_provider = Some("openai".into());
    raw.providers.insert(
        "openai".into(),
        ProviderConfig {
            preset: ProviderPreset::OpenAiCompat,
            base_url: Some(base_url),
            keys: vec!["k-slow".into(), "k-fast".into()],
            ..Default::default()
        },
    );
    let app = build_app(raw, &db_path);

    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("host", "127.0.0.1:8080")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"gpt-4o","messages":[]}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

    let retry_after: i64 = resp
        .headers()
        .get("retry-after")
        .unwrap()
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        (25..=31).contains(&retry_after),
        "expected Retry-After near earliest reset (30s), got {retry_after}"
    );

    let v: Value = serde_json::from_slice(&read_body_bytes(resp).await).unwrap();
    assert_eq!(v["error"]["type"], "all_keys_exhausted");
    assert_eq!(v["error"]["provider"], "openai");
    assert_eq!(v["error"]["model"], "gpt-4o");
    assert!(v["error"]["next_reset_at"].as_str().is_some());
}

#[tokio::test]
async fn cooldown_persists_across_sqlite_reload_and_never_stores_raw_keys() {
    let counters = MockCounters::default();
    let mock_router = Router::new()
        .fallback(any(
            |State(c): State<MockCounters>, headers: HeaderMap| async move {
                let key = extract_upstream_key(&headers);
                c.record("/v1/chat/completions".into(), key.clone(), None);
                if key == "super-secret-key-1" {
                    let mut resp = (
                        StatusCode::TOO_MANY_REQUESTS,
                        axum::Json(json!({"error": {"message": "limited"}})),
                    )
                        .into_response();
                    resp.headers_mut()
                        .insert("retry-after", HeaderValue::from_static("600"));
                    return resp;
                }
                (StatusCode::OK, axum::Json(json!({"key": key}))).into_response()
            },
        ))
        .with_state(counters.clone());

    let base_url = spawn_mock_upstream(mock_router).await;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("persist.db");

    let make_raw = || {
        let mut raw = RawConfig::default();
        raw.server.default_provider = Some("openai".into());
        raw.providers.insert(
            "openai".into(),
            ProviderConfig {
                preset: ProviderPreset::OpenAiCompat,
                base_url: Some(base_url.clone()),
                keys: vec!["super-secret-key-1".into(), "super-secret-key-2".into()],
                ..Default::default()
            },
        );
        raw
    };

    // First instance: key-1 gets 429 and rotates to key-2
    let app1 = build_app(make_raw(), &db_path);
    let req1 = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("host", "127.0.0.1:8080")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"gpt-4o","messages":[]}"#))
        .unwrap();
    let resp1 = app1.oneshot(req1).await.unwrap();
    assert_eq!(resp1.status(), StatusCode::OK);

    // Ensure raw key strings never appear in SQLite file bytes
    let db_bytes = std::fs::read(&db_path).unwrap();
    assert!(
        !db_bytes
            .windows("super-secret-key-1".len())
            .any(|w| w == b"super-secret-key-1"),
        "raw API key must never be stored in SQLite"
    );

    // Second instance (simulating process restart) reloads from the same SQLite DB
    let app2 = build_app(make_raw(), &db_path);

    // Verify /status shows cooling_down for gpt-4o on key-1
    let status_req = Request::builder()
        .uri("/status")
        .header("host", "127.0.0.1:8080")
        .body(Body::empty())
        .unwrap();
    let status_resp = app2.clone().oneshot(status_req).await.unwrap();
    assert_eq!(status_resp.status(), StatusCode::OK);
    let status_json: Value = serde_json::from_slice(&read_body_bytes(status_resp).await).unwrap();
    assert_eq!(
        status_json["providers"]["openai"]["keys"][0]["models"]["gpt-4o"]["state"],
        "cooling_down"
    );

    // Request on reloaded instance skips key-1 immediately and goes straight to key-2
    let before = counters.snapshot().len();
    let req2 = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("host", "127.0.0.1:8080")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"gpt-4o","messages":[]}"#))
        .unwrap();
    let resp2 = app2.clone().oneshot(req2).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
    let after = counters.snapshot();
    assert_eq!(after.len(), before + 1);
    assert_eq!(after.last().unwrap().1, "super-secret-key-2");

    // /status/reset clears the cooldown
    let reset_req = Request::builder()
        .method(Method::POST)
        .uri("/status/reset")
        .header("host", "127.0.0.1:8080")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"provider":"openai"}"#))
        .unwrap();
    let reset_resp = app2.clone().oneshot(reset_req).await.unwrap();
    assert_eq!(reset_resp.status(), StatusCode::OK);
    let reset_json: Value = serde_json::from_slice(&read_body_bytes(reset_resp).await).unwrap();
    assert!(reset_json["cleared"].as_u64().unwrap() >= 1);
}

#[tokio::test]
async fn unified_openai_routing_and_security_middlewares() {
    let mock_router = Router::new().fallback(any(
        |uri: axum::http::Uri, headers: HeaderMap, body: Bytes| async move {
            let key = extract_upstream_key(&headers);
            let parsed: Value = serde_json::from_slice(&body).unwrap_or(json!({}));
            (
                StatusCode::OK,
                axum::Json(json!({
                    "path": uri.path(),
                    "key": key,
                    "model": parsed.get("model"),
                })),
            )
        },
    ));
    let base_url = spawn_mock_upstream(mock_router).await;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("sec.db");

    let mut model_map = HashMap::new();
    model_map.insert("mapped-model".to_string(), "custom".to_string());

    let mut raw = RawConfig {
        server: ServerConfig {
            bind: "127.0.0.1:8080".into(),
            db_path: db_path.to_string_lossy().into(),
            api_token: Some("proxy-secret-token".into()),
            default_provider: Some("gemini".into()),
        },
        routing: RoutingConfig { model_map },
        providers: Default::default(),
    };
    raw.providers.insert(
        "gemini".into(),
        ProviderConfig {
            preset: ProviderPreset::Gemini,
            base_url: Some(base_url.clone()),
            keys: vec!["gem-k1".into()],
            ..Default::default()
        },
    );
    raw.providers.insert(
        "custom".into(),
        ProviderConfig {
            preset: ProviderPreset::OpenAiCompat,
            base_url: Some(base_url),
            auth: Some(AuthMode::Header("x-api-key".into())),
            keys: vec!["cust-k1".into()],
            ..Default::default()
        },
    );
    let app = build_app(raw, &db_path);

    // /health works without API_TOKEN
    let health = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("host", "127.0.0.1:8080")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);

    // /status without token -> 401
    let unauth = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/status")
                .header("host", "127.0.0.1:8080")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

    // Non-loopback Host -> 403
    let bad_host = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("host", "evil.example.com")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(bad_host.status(), StatusCode::FORBIDDEN);

    // Non-loopback Origin -> 403
    let bad_origin = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("host", "localhost:8080")
                .header("origin", "https://evil.example.com")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(bad_origin.status(), StatusCode::FORBIDDEN);

    // provider/model prefix routing strips prefix and forwards to /v1beta/openai/chat/completions
    let prefixed = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("host", "127.0.0.1:8080")
                .header("authorization", "Bearer proxy-secret-token")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"gemini/gemini-2.5-pro","messages":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(prefixed.status(), StatusCode::OK);
    let pv: Value = serde_json::from_slice(&read_body_bytes(prefixed).await).unwrap();
    assert_eq!(pv["path"], "/v1beta/openai/chat/completions");
    assert_eq!(pv["key"], "gem-k1");
    assert_eq!(pv["model"], "gemini-2.5-pro");

    // model_map routing forwards to custom provider with header:x-api-key
    let mapped = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("host", "127.0.0.1:8080")
                .header("authorization", "Bearer proxy-secret-token")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"mapped-model","messages":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(mapped.status(), StatusCode::OK);
    let mv: Value = serde_json::from_slice(&read_body_bytes(mapped).await).unwrap();
    assert_eq!(mv["path"], "/v1/chat/completions");
    assert_eq!(mv["key"], "cust-k1");
    assert_eq!(mv["model"], "mapped-model");
}
