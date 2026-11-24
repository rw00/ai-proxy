use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::net::SocketAddr;
use std::path::Path;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config file {0}: {1}")]
    Io(String, #[source] std::io::Error),
    #[error("failed to parse TOML config: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("failed to parse YAML config: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("invalid configuration: {0}")]
    Validation(String),
}

/// Wrapper for raw API keys. Never implements `Serialize` and redacts `Debug`/`Display`.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretKey(String);

impl SecretKey {
    pub fn new(raw: impl Into<String>) -> Self {
        Self(raw.into())
    }

    pub fn expose(&self) -> &str {
        &self.0
    }

    /// First 12 lowercase hex characters of SHA-256(raw_key).
    pub fn key_id(&self) -> String {
        let digest = Sha256::digest(self.0.as_bytes());
        hex::encode(digest)[..12].to_string()
    }

    /// Human-safe masked label (never exposes full key).
    pub fn masked(&self) -> String {
        let chars: Vec<char> = self.0.chars().collect();
        if chars.len() >= 10 {
            let prefix: String = chars[..4].iter().collect();
            let suffix: String = chars[chars.len() - 4..].iter().collect();
            format!("{prefix}...{suffix}")
        } else if chars.len() >= 4 {
            let prefix: String = chars[..2].iter().collect();
            format!("{prefix}...***")
        } else {
            "***".to_string()
        }
    }
}

impl fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretKey([REDACTED])")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthMode {
    Bearer,
    Header(String),
    Query(String),
}

impl AuthMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        let trimmed = s.trim();
        if trimmed.eq_ignore_ascii_case("bearer") {
            return Ok(Self::Bearer);
        }
        if let Some(name) = trimmed
            .strip_prefix("header:")
            .or_else(|| trimmed.strip_prefix("HEADER:"))
        {
            let name = name.trim();
            if name.is_empty() {
                return Err("header auth mode requires a header name (header:<name>)".into());
            }
            return Ok(Self::Header(name.to_string()));
        }
        if let Some(name) = trimmed
            .strip_prefix("query:")
            .or_else(|| trimmed.strip_prefix("QUERY:"))
        {
            let name = name.trim();
            if name.is_empty() {
                return Err("query auth mode requires a parameter name (query:<name>)".into());
            }
            return Ok(Self::Query(name.to_string()));
        }
        Err(format!(
            "unsupported auth mode '{s}' (expected 'bearer', 'header:<name>', or 'query:<name>')"
        ))
    }
}

impl<'de> Deserialize<'de> for AuthMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct AuthModeVisitor;
        impl<'de> Visitor<'de> for AuthModeVisitor {
            type Value = AuthMode;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("'bearer', 'header:<name>', or 'query:<name>'")
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                AuthMode::parse(v).map_err(E::custom)
            }
        }
        deserializer.deserialize_str(AuthModeVisitor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SelectionStrategy {
    #[default]
    Sticky,
    RoundRobin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProviderPreset {
    Gemini,
    #[default]
    OpenAiCompat,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub bind: String,
    pub db_path: String,
    pub api_token: Option<String>,
    pub default_provider: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8080".to_string(),
            db_path: "./ai-proxy.db".to_string(),
            api_token: None,
            default_provider: None,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RoutingConfig {
    pub model_map: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ProviderConfig {
    pub preset: ProviderPreset,
    pub base_url: Option<String>,
    pub openai_compat_path: Option<String>,
    pub auth: Option<AuthMode>,
    pub strategy: SelectionStrategy,
    pub retry_on_503: bool,
    pub fallback_cooldown_secs: u64,
    pub backoff_multiplier: f64,
    pub max_backoff_secs: u64,
    pub env_prefix: Option<String>,
    pub keys: Vec<String>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            preset: ProviderPreset::OpenAiCompat,
            base_url: None,
            openai_compat_path: None,
            auth: None,
            strategy: SelectionStrategy::Sticky,
            retry_on_503: false,
            fallback_cooldown_secs: 60,
            backoff_multiplier: 2.0,
            max_backoff_secs: 3600,
            env_prefix: None,
            keys: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct RawConfig {
    pub server: ServerConfig,
    pub routing: RoutingConfig,
    pub providers: BTreeMap<String, ProviderConfig>,
}

#[derive(Debug, Clone)]
pub struct KeyMaterial {
    pub key_id: String,
    pub masked: String,
    pub secret: SecretKey,
}

#[derive(Debug, Clone)]
pub struct ResolvedProvider {
    pub name: String,
    pub preset: ProviderPreset,
    pub base_url: String,
    pub openai_compat_path: String,
    pub auth: Option<AuthMode>,
    pub strategy: SelectionStrategy,
    pub retry_on_503: bool,
    pub fallback_cooldown_secs: u64,
    pub backoff_multiplier: f64,
    pub max_backoff_secs: u64,
    pub keys: Vec<KeyMaterial>,
}

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub routing: RoutingConfig,
    pub providers: BTreeMap<String, ResolvedProvider>,
}

impl AppConfig {
    pub fn load(path: Option<&Path>) -> Result<Self, ConfigError> {
        let env_vars: Vec<(String, String)> = std::env::vars().collect();
        let mut raw = if let Some(p) = path {
            let content = std::fs::read_to_string(p)
                .map_err(|e| ConfigError::Io(p.display().to_string(), e))?;
            parse_config_str(&content, p)?
        } else {
            let mut default = RawConfig::default();
            // Auto-register built-in providers when env keys are present and no config file was passed.
            if !extract_keys_from_env("GEMINI", &env_vars).is_empty() {
                default.providers.insert(
                    "gemini".to_string(),
                    ProviderConfig {
                        preset: ProviderPreset::Gemini,
                        ..Default::default()
                    },
                );
            }
            if !extract_keys_from_env("OPENAI", &env_vars).is_empty() {
                default.providers.insert(
                    "openai".to_string(),
                    ProviderConfig {
                        preset: ProviderPreset::OpenAiCompat,
                        base_url: Some("https://api.openai.com".to_string()),
                        auth: Some(AuthMode::Bearer),
                        ..Default::default()
                    },
                );
            }
            if !extract_keys_from_env("ANTHROPIC", &env_vars).is_empty() {
                default.providers.insert(
                    "anthropic".to_string(),
                    ProviderConfig {
                        preset: ProviderPreset::OpenAiCompat,
                        base_url: Some("https://api.anthropic.com".to_string()),
                        auth: Some(AuthMode::Header("x-api-key".to_string())),
                        ..Default::default()
                    },
                );
            }
            default
        };

        apply_server_env_overrides(&mut raw.server, &env_vars);
        Self::from_raw(raw, &env_vars)
    }

    pub fn from_raw(raw: RawConfig, env_vars: &[(String, String)]) -> Result<Self, ConfigError> {
        let mut providers = BTreeMap::new();

        for (name, pcfg) in raw.providers {
            let env_prefix = pcfg
                .env_prefix
                .clone()
                .unwrap_or_else(|| sanitize_env_prefix(&name));

            let mut all_raw_keys = Vec::new();
            for k in &pcfg.keys {
                let trimmed = k.trim();
                if !trimmed.is_empty() {
                    all_raw_keys.push(trimmed.to_string());
                }
            }
            all_raw_keys.extend(extract_keys_from_env(&env_prefix, env_vars));

            let mut seen = HashSet::new();
            let mut keys = Vec::new();
            for raw_key in all_raw_keys {
                let secret = SecretKey::new(raw_key);
                let key_id = secret.key_id();
                if seen.insert(key_id.clone()) {
                    let masked = secret.masked();
                    keys.push(KeyMaterial {
                        key_id,
                        masked,
                        secret,
                    });
                }
            }

            let trimmed_url = pcfg
                .base_url
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty());
            let base_url = match (pcfg.preset, trimmed_url) {
                (_, Some(url)) => url.trim_end_matches('/').to_string(),
                (ProviderPreset::Gemini, None) => {
                    "https://generativelanguage.googleapis.com".to_string()
                }
                (ProviderPreset::OpenAiCompat, None) => {
                    return Err(ConfigError::Validation(format!(
                        "provider '{name}' requires 'base_url'"
                    )));
                }
            };

            let openai_compat_path = match (pcfg.preset, pcfg.openai_compat_path) {
                (ProviderPreset::Gemini, None) => "/v1beta/openai".to_string(),
                (ProviderPreset::OpenAiCompat, None) => "/v1".to_string(),
                (_, Some(path)) => normalize_path_prefix(&path),
            };

            let backoff_multiplier = if pcfg.backoff_multiplier < 1.0 {
                1.0
            } else {
                pcfg.backoff_multiplier
            };

            providers.insert(
                name.clone(),
                ResolvedProvider {
                    name,
                    preset: pcfg.preset,
                    base_url,
                    openai_compat_path,
                    auth: pcfg.auth,
                    strategy: pcfg.strategy,
                    retry_on_503: pcfg.retry_on_503,
                    fallback_cooldown_secs: pcfg.fallback_cooldown_secs.max(1),
                    backoff_multiplier,
                    max_backoff_secs: pcfg.max_backoff_secs.max(pcfg.fallback_cooldown_secs.max(1)),
                    keys,
                },
            );
        }

        if let Some(default_provider) = &raw.server.default_provider {
            if !default_provider.is_empty() && !providers.contains_key(default_provider) {
                return Err(ConfigError::Validation(format!(
                    "default_provider '{default_provider}' is not defined in providers"
                )));
            }
        }

        Ok(Self {
            server: raw.server,
            routing: raw.routing,
            providers,
        })
    }

    pub fn is_loopback_bind(&self) -> bool {
        is_loopback_bind(&self.server.bind)
    }
}

pub fn parse_config_str(content: &str, path: &Path) -> Result<RawConfig, ConfigError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "yaml" | "yml" => Ok(serde_yaml::from_str(content)?),
        _ => Ok(toml::from_str(content)?),
    }
}

fn apply_server_env_overrides(server: &mut ServerConfig, env_vars: &[(String, String)]) {
    let lookup = |key: &str| -> Option<String> {
        env_vars
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };
    if let Some(bind) = lookup("AI_PROXY_BIND") {
        server.bind = bind;
    }
    if let Some(db_path) = lookup("AI_PROXY_DB_PATH") {
        server.db_path = db_path;
    }
    if let Some(token) = lookup("AI_PROXY_API_TOKEN").or_else(|| lookup("API_TOKEN")) {
        server.api_token = Some(token);
    } else if server.api_token.as_deref() == Some("") {
        server.api_token = None;
    }
    if let Some(default_provider) = lookup("AI_PROXY_DEFAULT_PROVIDER") {
        server.default_provider = Some(default_provider);
    }
}

pub fn sanitize_env_prefix(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

/// Extracts keys from:
/// - `{PREFIX}_API_KEYS` (JSON array `["k1","k2"]` or comma-separated `k1,k2`)
/// - `{PREFIX}_API_KEY_1..N` (sorted numerically by suffix)
/// - `{PREFIX}_API_KEY`
pub fn extract_keys_from_env(prefix: &str, env_vars: &[(String, String)]) -> Vec<String> {
    let mut keys = Vec::new();
    let multi_var = format!("{prefix}_API_KEYS");
    let single_var = format!("{prefix}_API_KEY");
    let indexed_prefix = format!("{prefix}_API_KEY_");

    if let Some((_, val)) = env_vars.iter().find(|(k, _)| k == &multi_var) {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            if trimmed.starts_with('[') {
                if let Ok(parsed) = serde_json::from_str::<Vec<String>>(trimmed) {
                    for k in parsed {
                        let t = k.trim();
                        if !t.is_empty() {
                            keys.push(t.to_string());
                        }
                    }
                }
            } else {
                for part in trimmed.split(',') {
                    let t = part.trim();
                    if !t.is_empty() {
                        keys.push(t.to_string());
                    }
                }
            }
        }
    }

    let mut indexed: Vec<(u64, String)> = Vec::new();
    for (k, v) in env_vars {
        if let Some(suffix) = k.strip_prefix(&indexed_prefix) {
            if let Ok(idx) = suffix.parse::<u64>() {
                let t = v.trim();
                if !t.is_empty() {
                    indexed.push((idx, t.to_string()));
                }
            }
        }
    }
    indexed.sort_by_key(|(idx, _)| *idx);
    keys.extend(indexed.into_iter().map(|(_, v)| v));

    if let Some((_, val)) = env_vars.iter().find(|(k, _)| k == &single_var) {
        let t = val.trim();
        if !t.is_empty() {
            keys.push(t.to_string());
        }
    }

    keys
}

pub fn normalize_path_prefix(path: &str) -> String {
    let trimmed = path.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        String::new()
    } else if trimmed.starts_with('/') {
        trimmed.to_string()
    } else {
        format!("/{trimmed}")
    }
}

pub fn is_loopback_bind(bind: &str) -> bool {
    if let Ok(addr) = bind.parse::<SocketAddr>() {
        return addr.ip().is_loopback();
    }
    let host = bind
        .strip_prefix('[')
        .and_then(|s| s.split(']').next())
        .or_else(|| bind.rsplit_once(':').map(|(h, _)| h))
        .unwrap_or(bind);
    matches!(host.to_ascii_lowercase().as_str(), "127.0.0.1" | "::1" | "localhost")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_key_hashing_masking_and_redaction() {
        let key = SecretKey::new("AIzaSyTestKey1234567890");
        assert_eq!(key.key_id().len(), 12);
        assert_eq!(key.masked(), "AIza...7890");
        assert_eq!(format!("{key:?}"), "SecretKey([REDACTED])");
        assert!(!format!("{key:?}").contains("AIzaSy"));
    }

    #[test]
    fn env_key_extraction_supports_csv_json_and_indexed() {
        let env = vec![
            ("GEMINI_API_KEYS".into(), "k1, k2".into()),
            ("GEMINI_API_KEY_10".into(), "k10".into()),
            ("GEMINI_API_KEY_2".into(), "k2_idx".into()),
            ("GEMINI_API_KEY_1".into(), "k1_idx".into()),
        ];
        assert_eq!(
            extract_keys_from_env("GEMINI", &env),
            vec!["k1", "k2", "k1_idx", "k2_idx", "k10"]
        );

        let json_env = vec![("OPENAI_API_KEYS".into(), r#"["sk-1","sk-2"]"#.into())];
        assert_eq!(
            extract_keys_from_env("OPENAI", &json_env),
            vec!["sk-1", "sk-2"]
        );
    }

    #[test]
    fn auth_mode_parsing() {
        assert_eq!(AuthMode::parse("bearer").unwrap(), AuthMode::Bearer);
        assert_eq!(
            AuthMode::parse("header:x-api-key").unwrap(),
            AuthMode::Header("x-api-key".into())
        );
        assert_eq!(
            AuthMode::parse("query:key").unwrap(),
            AuthMode::Query("key".into())
        );
        assert!(AuthMode::parse("invalid").is_err());
    }

    #[test]
    fn yaml_and_toml_config_loading() {
        let yaml = r#"
server:
  bind: "127.0.0.1:9090"
providers:
  gemini:
    preset: gemini
    strategy: round_robin
    keys: ["g-key-1"]
"#;
        let raw = parse_config_str(yaml, Path::new("config.yaml")).unwrap();
        let cfg = AppConfig::from_raw(raw, &[]).unwrap();
        assert_eq!(cfg.server.bind, "127.0.0.1:9090");
        assert_eq!(cfg.providers["gemini"].strategy, SelectionStrategy::RoundRobin);
        assert_eq!(cfg.providers["gemini"].keys.len(), 1);
    }
}
