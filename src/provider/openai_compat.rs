use chrono::{DateTime, Utc};
use http::StatusCode;
use serde_json::Value;

use crate::config::AuthMode;
use crate::provider::{
    apply_auth_mode, extract_json_model, strip_caller_auth, ErrorAction, ProviderAdapter,
    QuotaScope, UpstreamRequest,
};
use crate::quota::extract_duration_hint_from_text;

pub struct OpenAiCompatAdapter {
    auth: AuthMode,
}

impl OpenAiCompatAdapter {
    pub fn new(auth: AuthMode) -> Self {
        Self { auth }
    }
}

impl ProviderAdapter for OpenAiCompatAdapter {
    fn kind(&self) -> &'static str {
        "openai_compat"
    }

    fn extract_model(&self, _path: &str, body: &[u8]) -> Option<String> {
        extract_json_model(body)
    }

    fn apply_auth(&self, req: &mut UpstreamRequest, raw_key: &str, _is_openai_compat: bool) {
        strip_caller_auth(req);
        apply_auth_mode(req, &self.auth, raw_key);
    }

    fn classify_body(
        &self,
        status: StatusCode,
        body: &[u8],
        now: DateTime<Utc>,
    ) -> Option<ErrorAction> {
        let parsed: Option<Value> = serde_json::from_slice(body).ok();
        let err_obj = parsed.as_ref().and_then(|v| v.get("error"));
        let err_type = err_obj
            .and_then(|e| e.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let err_code = err_obj
            .and_then(|e| e.get("code"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let err_msg = err_obj
            .and_then(|e| e.get("message"))
            .and_then(Value::as_str)
            .unwrap_or("");

        let lower_type = err_type.to_ascii_lowercase();
        let lower_code = err_code.to_ascii_lowercase();
        let lower_msg = err_msg.to_ascii_lowercase();

        if status == StatusCode::UNAUTHORIZED || lower_type == "authentication_error" {
            return Some(ErrorAction::InvalidKey {
                reason: non_empty_reason(err_msg, "HTTP 401 unauthorized"),
            });
        }

        let is_quota = status == StatusCode::TOO_MANY_REQUESTS
            || lower_type == "rate_limit_error"
            || lower_code == "rate_limit_exceeded"
            || lower_code == "insufficient_quota"
            || lower_type == "insufficient_quota"
            || (status == StatusCode::FORBIDDEN
                && (lower_msg.contains("quota") || lower_msg.contains("rate limit")));

        if is_quota {
            let scope = if lower_code == "insufficient_quota"
                || lower_type == "insufficient_quota"
                || lower_code == "billing_hard_limit_reached"
            {
                QuotaScope::Key
            } else {
                QuotaScope::Model
            };
            let reset_at = extract_duration_hint_from_text(err_msg).map(|d| now + d);
            return Some(ErrorAction::Quota {
                scope,
                reset_at,
                reason: non_empty_reason(err_msg, "rate limit or quota exceeded"),
            });
        }

        if status == StatusCode::FORBIDDEN {
            return Some(ErrorAction::InvalidKey {
                reason: non_empty_reason(err_msg, "HTTP 403 forbidden / invalid API key"),
            });
        }

        None
    }
}

fn non_empty_reason(msg: &str, fallback: &str) -> String {
    let t = msg.trim();
    if t.is_empty() {
        fallback.to_string()
    } else {
        t.chars().take(200).collect()
    }
}
