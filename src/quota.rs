use chrono::{DateTime, Duration, Utc};
use http::{HeaderMap, StatusCode};

use crate::config::ResolvedProvider;
use crate::provider::{ErrorAction, ProviderAdapter, QuotaScope};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassifiedResponse {
    Quota {
        scope: QuotaScope,
        reset_at: DateTime<Utc>,
        reason: String,
    },
    InvalidKey {
        reason: String,
    },
    Passthrough,
}

/// Classifies an upstream non-2xx response and computes `reset_at` in priority order:
/// 1. Standard headers (`Retry-After`, `x-ratelimit-reset-*`, `anthropic-ratelimit-*-reset`)
/// 2. Provider-specific body parsing
/// 3. Configurable fallback cooldown with optional exponential backoff
pub fn classify_response(
    adapter: &dyn ProviderAdapter,
    cfg: &ResolvedProvider,
    status: StatusCode,
    headers: &HeaderMap,
    body: &[u8],
    now: DateTime<Utc>,
    previous_consecutive_hits: u32,
) -> ClassifiedResponse {
    if status == StatusCode::SERVICE_UNAVAILABLE {
        if !cfg.retry_on_503 {
            return ClassifiedResponse::Passthrough;
        }
        let header_reset = parse_rate_limit_headers(headers, now);
        let reset_at = clamp_future(
            header_reset.unwrap_or_else(|| compute_fallback_reset(cfg, now, previous_consecutive_hits)),
            now,
        );
        return ClassifiedResponse::Quota {
            scope: QuotaScope::Model,
            reset_at,
            reason: "HTTP 503 service unavailable".to_string(),
        };
    }

    let action = adapter
        .classify_body(status, body, now)
        .unwrap_or_else(|| match status {
            StatusCode::UNAUTHORIZED => ErrorAction::InvalidKey {
                reason: "HTTP 401 unauthorized".to_string(),
            },
            StatusCode::TOO_MANY_REQUESTS => ErrorAction::Quota {
                scope: QuotaScope::Model,
                reset_at: None,
                reason: "HTTP 429 too many requests".to_string(),
            },
            _ => ErrorAction::Passthrough,
        });

    match action {
        ErrorAction::InvalidKey { reason } => ClassifiedResponse::InvalidKey { reason },
        ErrorAction::Passthrough => ClassifiedResponse::Passthrough,
        ErrorAction::Quota {
            scope,
            reset_at: body_reset,
            reason,
        } => {
            let header_reset = parse_rate_limit_headers(headers, now);
            let chosen = header_reset
                .or(body_reset)
                .unwrap_or_else(|| compute_fallback_reset(cfg, now, previous_consecutive_hits));
            ClassifiedResponse::Quota {
                scope,
                reset_at: clamp_future(chosen, now),
                reason,
            }
        }
    }
}

fn clamp_future(dt: DateTime<Utc>, now: DateTime<Utc>) -> DateTime<Utc> {
    if dt <= now {
        now + Duration::seconds(1)
    } else {
        dt
    }
}

pub fn compute_fallback_reset(
    cfg: &ResolvedProvider,
    now: DateTime<Utc>,
    previous_consecutive_hits: u32,
) -> DateTime<Utc> {
    let base = cfg.fallback_cooldown_secs as f64;
    let mult = cfg.backoff_multiplier.powi(previous_consecutive_hits.min(16) as i32);
    let secs = (base * mult).round() as u64;
    let clamped = secs.clamp(1, cfg.max_backoff_secs);
    now + Duration::seconds(clamped as i64)
}

/// Parses rate-limit reset headers in priority order:
/// 1. `Retry-After` (delta seconds or HTTP-date)
/// 2. `x-ratelimit-reset-requests`, `x-ratelimit-reset-tokens`
/// 3. `anthropic-ratelimit-*-reset`
pub fn parse_rate_limit_headers(headers: &HeaderMap, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
    if let Some(val) = headers.get(http::header::RETRY_AFTER).and_then(|v| v.to_str().ok()) {
        if let Some(dt) = parse_retry_after_value(val, now) {
            return Some(dt);
        }
    }

    let openai_headers = ["x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"];
    if let Some(dt) = max_parsed_header(headers, &openai_headers, now) {
        return Some(dt);
    }

    let anthropic_headers = [
        "anthropic-ratelimit-requests-reset",
        "anthropic-ratelimit-tokens-reset",
        "anthropic-ratelimit-input-tokens-reset",
        "anthropic-ratelimit-output-tokens-reset",
    ];
    if let Some(dt) = max_parsed_header(headers, &anthropic_headers, now) {
        return Some(dt);
    }

    // Catch any other `anthropic-ratelimit-*-reset` header dynamically.
    let mut best: Option<DateTime<Utc>> = None;
    for (name, val) in headers {
        let lower = name.as_str().to_ascii_lowercase();
        if lower.starts_with("anthropic-ratelimit-") && lower.ends_with("-reset") {
            if let Ok(s) = val.to_str() {
                if let Some(dt) = parse_reset_value(s, now) {
                    best = Some(match best {
                        Some(prev) => prev.max(dt),
                        None => dt,
                    });
                }
            }
        }
    }
    best
}

fn max_parsed_header(
    headers: &HeaderMap,
    names: &[&str],
    now: DateTime<Utc>,
) -> Option<DateTime<Utc>> {
    let mut best: Option<DateTime<Utc>> = None;
    for name in names {
        if let Some(val) = headers.get(*name).and_then(|v| v.to_str().ok()) {
            if let Some(dt) = parse_reset_value(val, now) {
                best = Some(match best {
                    Some(prev) => prev.max(dt),
                    None => dt,
                });
            }
        }
    }
    best
}

pub fn parse_retry_after_value(raw: &str, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(secs) = trimmed.parse::<f64>() {
        if secs.is_finite() && secs >= 0.0 {
            let millis = (secs * 1000.0).ceil() as i64;
            return Some(now + Duration::milliseconds(millis));
        }
    }
    if let Ok(sys_time) = httpdate::parse_http_date(trimmed) {
        let dt: DateTime<Utc> = sys_time.into();
        return Some(dt);
    }
    None
}

fn parse_reset_value(raw: &str, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(dt) = DateTime::parse_from_rfc3339(trimmed) {
        return Some(dt.with_timezone(&Utc));
    }
    if let Ok(sys_time) = httpdate::parse_http_date(trimmed) {
        let dt: DateTime<Utc> = sys_time.into();
        return Some(dt);
    }
    // Unix epoch seconds (heuristic: > 1_000_000_000)
    if let Ok(epoch) = trimmed.parse::<i64>() {
        if epoch > 1_000_000_000 {
            return DateTime::from_timestamp(epoch, 0);
        }
        if epoch >= 0 {
            return Some(now + Duration::seconds(epoch));
        }
    }
    if let Some(dur) = parse_duration_str(trimmed) {
        return Some(now + dur);
    }
    None
}

/// Parses durations such as `"42s"`, `"34.5s"`, `"500ms"`, `"1m30s"`, `"2h5m10s"`, `"15"`.
pub fn parse_duration_str(raw: &str) -> Option<Duration> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(secs) = s.parse::<f64>() {
        if secs.is_finite() && secs >= 0.0 {
            return Some(Duration::milliseconds((secs * 1000.0).ceil() as i64));
        }
        return None;
    }

    let bytes = s.as_bytes();
    let mut i = 0;
    let mut total_ms: f64 = 0.0;
    let mut parsed_any = false;

    while i < bytes.len() {
        let num_start = i;
        let mut saw_dot = false;
        while i < bytes.len() && (bytes[i].is_ascii_digit() || (bytes[i] == b'.' && !saw_dot)) {
            if bytes[i] == b'.' {
                saw_dot = true;
            }
            i += 1;
        }
        if i == num_start {
            return None;
        }
        let val: f64 = s[num_start..i].parse().ok()?;

        let unit_start = i;
        while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
            i += 1;
        }
        let unit = &s[unit_start..i];
        let mult_ms = match unit {
            "ms" => 1.0,
            "s" => 1000.0,
            "m" => 60_000.0,
            "h" => 3_600_000.0,
            "d" => 86_400_000.0,
            _ => return None,
        };
        total_ms += val * mult_ms;
        parsed_any = true;
    }

    if parsed_any && total_ms.is_finite() && total_ms >= 0.0 {
        Some(Duration::milliseconds(total_ms.ceil() as i64))
    } else {
        None
    }
}

/// Extracts duration tokens following phrases like "retry in 20s", "try again in 1m30s", "retry after 45s".
pub fn extract_duration_hint_from_text(text: &str) -> Option<Duration> {
    let lower = text.to_ascii_lowercase();
    for needle in ["try again in ", "retry in ", "retry after ", "wait "] {
        if let Some(idx) = lower.find(needle) {
            let rest = &lower[idx + needle.len()..];
            let token = rest
                .split(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | ')' | ']' | '"'))
                .next()
                .unwrap_or("")
                .trim_end_matches('.');
            if let Some(dur) = parse_duration_str(token) {
                return Some(dur);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use http::HeaderValue;

    #[test]
    fn parses_retry_after_delta_and_http_date() {
        let now = Utc.with_ymd_and_hms(2026, 11, 24, 12, 0, 0).unwrap();

        let mut h = HeaderMap::new();
        h.insert("retry-after", HeaderValue::from_static("45"));
        assert_eq!(
            parse_rate_limit_headers(&h, now),
            Some(now + Duration::seconds(45))
        );

        let mut h2 = HeaderMap::new();
        h2.insert(
            "retry-after",
            HeaderValue::from_static("Tue, 24 Nov 2026 12:02:00 GMT"),
        );
        assert_eq!(
            parse_rate_limit_headers(&h2, now),
            Some(Utc.with_ymd_and_hms(2026, 11, 24, 12, 2, 0).unwrap())
        );
    }

    #[test]
    fn parses_openai_and_anthropic_reset_headers() {
        let now = Utc.with_ymd_and_hms(2026, 11, 24, 12, 0, 0).unwrap();

        let mut h = HeaderMap::new();
        h.insert(
            "x-ratelimit-reset-requests",
            HeaderValue::from_static("1m30s"),
        );
        h.insert(
            "x-ratelimit-reset-tokens",
            HeaderValue::from_static("500ms"),
        );
        assert_eq!(
            parse_rate_limit_headers(&h, now),
            Some(now + Duration::seconds(90))
        );

        let mut h2 = HeaderMap::new();
        h2.insert(
            "anthropic-ratelimit-requests-reset",
            HeaderValue::from_static("2026-11-24T12:05:00Z"),
        );
        h2.insert(
            "anthropic-ratelimit-tokens-reset",
            HeaderValue::from_static("2026-11-24T12:03:00Z"),
        );
        assert_eq!(
            parse_rate_limit_headers(&h2, now),
            Some(Utc.with_ymd_and_hms(2026, 11, 24, 12, 5, 0).unwrap())
        );
    }

    #[test]
    fn parses_duration_strings_and_text_hints() {
        assert_eq!(parse_duration_str("1m30s"), Some(Duration::seconds(90)));
        assert_eq!(parse_duration_str("500ms"), Some(Duration::milliseconds(500)));
        assert_eq!(parse_duration_str("34.5s"), Some(Duration::milliseconds(34500)));
        assert_eq!(
            extract_duration_hint_from_text("Rate limit reached. Please try again in 2m10s."),
            Some(Duration::seconds(130))
        );
    }
}
