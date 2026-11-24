pub mod gemini;
pub mod openai_compat;

use std::sync::Arc;

use chrono::{DateTime, Utc};
use http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use url::Url;

use crate::config::{AuthMode, ProviderPreset, ResolvedProvider};

#[derive(Debug, Clone)]
pub struct UpstreamRequest {
    pub url: Url,
    pub headers: HeaderMap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaScope {
    Model,
    Key,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorAction {
    Quota {
        scope: QuotaScope,
        reset_at: Option<DateTime<Utc>>,
        reason: String,
    },
    InvalidKey {
        reason: String,
    },
    Passthrough,
}

pub trait ProviderAdapter: Send + Sync {
    fn kind(&self) -> &'static str;
    fn extract_model(&self, path: &str, body: &[u8]) -> Option<String>;
    fn apply_auth(&self, req: &mut UpstreamRequest, raw_key: &str, is_openai_compat: bool);
    fn openai_compat_url(&self, base_url: &str, compat_path: &str, endpoint: &str) -> String {
        let base = base_url.trim_end_matches('/');
        let prefix = compat_path.trim_end_matches('/');
        let ep = endpoint.trim_start_matches('/');
        if prefix.is_empty() {
            format!("{base}/{ep}")
        } else {
            format!("{base}{prefix}/{ep}")
        }
    }
    fn classify_body(
        &self,
        status: StatusCode,
        body: &[u8],
        now: DateTime<Utc>,
    ) -> Option<ErrorAction>;
}

pub fn build_adapter(cfg: &ResolvedProvider) -> Arc<dyn ProviderAdapter> {
    match cfg.preset {
        ProviderPreset::Gemini => Arc::new(gemini::GeminiAdapter::new(cfg.auth.clone())),
        ProviderPreset::OpenAiCompat => Arc::new(openai_compat::OpenAiCompatAdapter::new(
            cfg.auth.clone().unwrap_or(AuthMode::Bearer),
        )),
    }
}

/// Strips caller-supplied dummy API credentials before injecting the rotated upstream key.
pub fn strip_caller_auth(req: &mut UpstreamRequest) {
    for name in ["authorization", "x-api-key", "x-goog-api-key", "api-key"] {
        req.headers.remove(name);
    }

    let kept_pairs: Vec<(String, String)> = req
        .url
        .query_pairs()
        .filter(|(k, _)| !k.eq_ignore_ascii_case("key"))
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect();

    if kept_pairs.is_empty() {
        req.url.set_query(None);
    } else {
        let mut ser = req.url.query_pairs_mut();
        ser.clear();
        for (k, v) in kept_pairs {
            ser.append_pair(&k, &v);
        }
    }
}

pub fn apply_auth_mode(req: &mut UpstreamRequest, mode: &AuthMode, raw_key: &str) {
    match mode {
        AuthMode::Bearer => {
            if let Ok(val) = HeaderValue::from_str(&format!("Bearer {raw_key}")) {
                req.headers.insert(http::header::AUTHORIZATION, val);
            }
        }
        AuthMode::Header(name) => {
            if let (Ok(hdr_name), Ok(val)) = (
                HeaderName::from_bytes(name.as_bytes()),
                HeaderValue::from_str(raw_key),
            ) {
                req.headers.insert(hdr_name, val);
            }
        }
        AuthMode::Query(param) => {
            let kept_pairs: Vec<(String, String)> = req
                .url
                .query_pairs()
                .filter(|(k, _)| k != param)
                .map(|(k, v)| (k.into_owned(), v.into_owned()))
                .collect();
            let mut ser = req.url.query_pairs_mut();
            ser.clear();
            for (k, v) in kept_pairs {
                ser.append_pair(&k, &v);
            }
            ser.append_pair(param, raw_key);
        }
    }
}

/// Filters hop-by-hop headers (RFC 7230) and host/content-length before forwarding.
pub fn filter_request_headers(headers: &HeaderMap) -> HeaderMap {
    let mut connection_tokens: Vec<String> = Vec::new();
    for val in headers.get_all(http::header::CONNECTION) {
        if let Ok(s) = val.to_str() {
            for token in s.split(',') {
                let t = token.trim().to_ascii_lowercase();
                if !t.is_empty() {
                    connection_tokens.push(t);
                }
            }
        }
    }

    let mut out = HeaderMap::new();
    for (name, val) in headers {
        let lower = name.as_str().to_ascii_lowercase();
        if matches!(
            lower.as_str(),
            "host"
                | "connection"
                | "keep-alive"
                | "proxy-authenticate"
                | "proxy-authorization"
                | "te"
                | "trailers"
                | "transfer-encoding"
                | "upgrade"
                | "content-length"
        ) || connection_tokens.iter().any(|t| t == &lower)
        {
            continue;
        }
        out.append(name.clone(), val.clone());
    }
    out
}

/// Filters hop-by-hop response headers before returning to the downstream caller.
pub fn filter_response_headers(headers: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (name, val) in headers {
        let lower = name.as_str().to_ascii_lowercase();
        if matches!(
            lower.as_str(),
            "connection"
                | "keep-alive"
                | "proxy-authenticate"
                | "proxy-authorization"
                | "te"
                | "trailers"
                | "transfer-encoding"
                | "upgrade"
        ) {
            continue;
        }
        out.append(name.clone(), val.clone());
    }
    out
}

pub fn extract_json_model(body: &[u8]) -> Option<String> {
    if body.is_empty() {
        return None;
    }
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    v.get("model")
        .and_then(|m| m.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}
