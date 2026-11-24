use chrono::{DateTime, Duration, TimeZone, Utc};
use chrono_tz::America::Los_Angeles;
use http::StatusCode;
use serde_json::Value;

use crate::config::AuthMode;
use crate::provider::{
    apply_auth_mode, extract_json_model, strip_caller_auth, ErrorAction, ProviderAdapter,
    QuotaScope, UpstreamRequest,
};
use crate::quota::{extract_duration_hint_from_text, parse_duration_str};

pub struct GeminiAdapter {
    auth_override: Option<AuthMode>,
}

impl GeminiAdapter {
    pub fn new(auth_override: Option<AuthMode>) -> Self {
        Self { auth_override }
    }
}

impl ProviderAdapter for GeminiAdapter {
    fn kind(&self) -> &'static str {
        "gemini"
    }

    fn extract_model(&self, path: &str, body: &[u8]) -> Option<String> {
        // Native Gemini URLs look like `/v1beta/models/gemini-2.5-pro:streamGenerateContent`
        if let Some(after_models) = path.split("models/").nth(1) {
            let model = after_models
                .split([':', '/', '?'])
                .next()
                .map(str::trim)
                .filter(|s| !s.is_empty());
            if let Some(m) = model {
                return Some(m.to_string());
            }
        }
        extract_json_model(body)
    }

    fn apply_auth(&self, req: &mut UpstreamRequest, raw_key: &str, is_openai_compat: bool) {
        strip_caller_auth(req);
        if let Some(mode) = &self.auth_override {
            apply_auth_mode(req, mode, raw_key);
            return;
        }
        if is_openai_compat || req.url.path().contains("/openai/") {
            apply_auth_mode(req, &AuthMode::Bearer, raw_key);
        } else {
            apply_auth_mode(
                req,
                &AuthMode::Header("x-goog-api-key".to_string()),
                raw_key,
            );
        }
    }

    fn classify_body(
        &self,
        status: StatusCode,
        body: &[u8],
        now: DateTime<Utc>,
    ) -> Option<ErrorAction> {
        let error_obj = parse_gemini_error_object(body);
        let err_status = error_obj
            .as_ref()
            .and_then(|e| e.get("status"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let err_message = error_obj
            .as_ref()
            .and_then(|e| e.get("message"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let lower_msg = err_message.to_ascii_lowercase();

        if status == StatusCode::UNAUTHORIZED {
            return Some(ErrorAction::InvalidKey {
                reason: non_empty_reason(err_message, "HTTP 401 unauthorized"),
            });
        }

        let is_quota = status == StatusCode::TOO_MANY_REQUESTS
            || err_status.eq_ignore_ascii_case("RESOURCE_EXHAUSTED")
            || (status == StatusCode::FORBIDDEN
                && (lower_msg.contains("quota")
                    || lower_msg.contains("rate limit")
                    || lower_msg.contains("resource_exhausted")
                    || has_quota_failure_detail(error_obj.as_ref())));

        if is_quota {
            let reset_at = error_obj
                .as_ref()
                .and_then(|e| parse_gemini_reset_at(e, now));
            return Some(ErrorAction::Quota {
                scope: QuotaScope::Model,
                reset_at,
                reason: non_empty_reason(err_message, "Gemini quota exceeded"),
            });
        }

        if status == StatusCode::FORBIDDEN {
            return Some(ErrorAction::InvalidKey {
                reason: non_empty_reason(err_message, "HTTP 403 forbidden / invalid API key"),
            });
        }

        None
    }
}

fn parse_gemini_error_object(body: &[u8]) -> Option<Value> {
    let val: Value = serde_json::from_slice(body).ok()?;
    if let Some(err) = val.get("error") {
        return Some(err.clone());
    }
    if let Some(arr) = val.as_array() {
        for item in arr {
            if let Some(err) = item.get("error") {
                return Some(err.clone());
            }
        }
    }
    None
}

fn has_quota_failure_detail(error_obj: Option<&Value>) -> bool {
    let Some(details) = error_obj.and_then(|e| e.get("details")).and_then(Value::as_array) else {
        return false;
    };
    details.iter().any(|d| {
        d.get("@type")
            .and_then(Value::as_str)
            .is_some_and(|t| t.ends_with("QuotaFailure"))
    })
}

fn parse_gemini_reset_at(error_obj: &Value, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let details = error_obj.get("details").and_then(Value::as_array);

    // 1. google.rpc.RetryInfo.retryDelay
    if let Some(details) = details {
        for detail in details {
            let kind = detail.get("@type").and_then(Value::as_str).unwrap_or("");
            if kind.ends_with("RetryInfo") {
                if let Some(delay_str) = detail.get("retryDelay").and_then(Value::as_str) {
                    if let Some(dur) = parse_duration_str(delay_str) {
                        return Some(now + dur);
                    }
                }
            }
        }

        // 2. google.rpc.QuotaFailure violations
        let mut saw_rpd = false;
        let mut saw_rpm_or_tpm = false;
        for detail in details {
            let kind = detail.get("@type").and_then(Value::as_str).unwrap_or("");
            if kind.ends_with("QuotaFailure") {
                if let Some(violations) = detail.get("violations").and_then(Value::as_array) {
                    for v in violations {
                        let combined = format!(
                            "{} {} {}",
                            v.get("quotaId").and_then(Value::as_str).unwrap_or(""),
                            v.get("quotaMetric").and_then(Value::as_str).unwrap_or(""),
                            v.get("description").and_then(Value::as_str).unwrap_or(""),
                        )
                        .to_ascii_lowercase();

                        if combined.contains("perday")
                            || combined.contains("per_day")
                            || combined.contains("per day")
                            || combined.contains("daily")
                            || combined.contains("rpd")
                        {
                            saw_rpd = true;
                        } else if combined.contains("perminute")
                            || combined.contains("per_minute")
                            || combined.contains("per minute")
                            || combined.contains("rpm")
                            || combined.contains("tpm")
                        {
                            saw_rpm_or_tpm = true;
                        }
                    }
                }
            }
        }
        if saw_rpd {
            return Some(next_midnight_pt(now));
        }
        if saw_rpm_or_tpm {
            return Some(now + Duration::seconds(60));
        }
    }

    // 3. Message-level hints
    if let Some(msg) = error_obj.get("message").and_then(Value::as_str) {
        if let Some(dur) = extract_duration_hint_from_text(msg) {
            return Some(now + dur);
        }
        let lower = msg.to_ascii_lowercase();
        if lower.contains("per day") || lower.contains("daily") || lower.contains("rpd") {
            return Some(next_midnight_pt(now));
        }
        if lower.contains("per minute") || lower.contains("rpm") || lower.contains("tpm") {
            return Some(now + Duration::seconds(60));
        }
    }

    None
}

/// Computes the next `00:00:00 America/Los_Angeles` after `now` (DST-safe).
pub fn next_midnight_pt(now: DateTime<Utc>) -> DateTime<Utc> {
    let pt_now = now.with_timezone(&Los_Angeles);
    let next_date = pt_now
        .date_naive()
        .succ_opt()
        .expect("date overflow computing next midnight PT");
    let naive_midnight = next_date
        .and_hms_opt(0, 0, 0)
        .expect("valid midnight naive time");
    match Los_Angeles.from_local_datetime(&naive_midnight) {
        chrono::LocalResult::Single(dt) => dt.with_timezone(&Utc),
        chrono::LocalResult::Ambiguous(earliest, _) => earliest.with_timezone(&Utc),
        chrono::LocalResult::None => now + Duration::hours(24),
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn extracts_model_from_native_path_and_openai_body() {
        let adapter = GeminiAdapter::new(None);
        assert_eq!(
            adapter.extract_model("/v1beta/models/gemini-2.5-pro:streamGenerateContent", b"{}"),
            Some("gemini-2.5-pro".to_string())
        );
        assert_eq!(
            adapter.extract_model(
                "/v1beta/openai/chat/completions",
                br#"{"model":"gemini-2.5-flash"}"#
            ),
            Some("gemini-2.5-flash".to_string())
        );
    }

    #[test]
    fn parses_retry_info_and_quota_failure_rpd_and_rpm() {
        let adapter = GeminiAdapter::new(None);
        // 2026-06-15 20:00:00 UTC = 13:00:00 PDT (UTC-7), next midnight PT = 2026-06-16 07:00:00 UTC
        let summer = Utc.with_ymd_and_hms(2026, 6, 15, 20, 0, 0).unwrap();

        let retry_body = br#"{
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "Quota exceeded",
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "42s"}
                ]
            }
        }"#;
        let action = adapter
            .classify_body(StatusCode::TOO_MANY_REQUESTS, retry_body, summer)
            .unwrap();
        assert_eq!(
            action,
            ErrorAction::Quota {
                scope: QuotaScope::Model,
                reset_at: Some(summer + Duration::seconds(42)),
                reason: "Quota exceeded".to_string(),
            }
        );

        let rpd_body = br#"{
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "Daily limit reached",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]
                    }
                ]
            }
        }"#;
        let rpd_action = adapter
            .classify_body(StatusCode::TOO_MANY_REQUESTS, rpd_body, summer)
            .unwrap();
        assert_eq!(
            rpd_action,
            ErrorAction::Quota {
                scope: QuotaScope::Model,
                reset_at: Some(Utc.with_ymd_and_hms(2026, 6, 16, 7, 0, 0).unwrap()),
                reason: "Daily limit reached".to_string(),
            }
        );

        // Winter (PST = UTC-8): 2026-01-15 20:00:00 UTC -> next midnight PT = 2026-01-16 08:00:00 UTC
        let winter = Utc.with_ymd_and_hms(2026, 1, 15, 20, 0, 0).unwrap();
        assert_eq!(
            next_midnight_pt(winter),
            Utc.with_ymd_and_hms(2026, 1, 16, 8, 0, 0).unwrap()
        );

        let rpm_body = br#"{
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "Minute limit reached",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}]
                    }
                ]
            }
        }"#;
        let rpm_action = adapter
            .classify_body(StatusCode::TOO_MANY_REQUESTS, rpm_body, summer)
            .unwrap();
        assert_eq!(
            rpm_action,
            ErrorAction::Quota {
                scope: QuotaScope::Model,
                reset_at: Some(summer + Duration::seconds(60)),
                reason: "Minute limit reached".to_string(),
            }
        );
    }
}
