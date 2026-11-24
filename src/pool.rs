use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::RwLock;

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::config::{KeyMaterial, ResolvedProvider, SelectionStrategy};
use crate::provider::QuotaScope;
use crate::store::{Store, StoreError, StoredKeyState};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyHealth {
    Ready,
    CoolingDown,
    Invalid,
}

#[derive(Debug, Clone)]
struct EntryState {
    health: KeyHealth,
    reset_at: Option<DateTime<Utc>>,
    consecutive_hits: u32,
    reason: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct KeyStats {
    pub requests: u64,
    pub rate_limits: u64,
    pub errors: u64,
}

#[derive(Debug, Clone)]
pub struct SelectedKey {
    pub key_id: String,
    pub masked: String,
    pub secret: crate::config::SecretKey,
}

#[derive(Debug, Clone)]
pub enum SelectError {
    UnknownProvider,
    NoKeysConfigured,
    AllExhausted {
        next_reset_at: Option<DateTime<Utc>>,
    },
}

struct PoolInner {
    /// Keyed by `(provider, key_id, model)` where `model == ""` means key-level state.
    states: HashMap<(String, String, String), EntryState>,
    /// Keyed by `(provider, key_id)`.
    stats: HashMap<(String, String), KeyStats>,
    /// Sticky selected key index per `(provider, model)`.
    sticky_index: HashMap<(String, String), usize>,
    /// Round-robin cursor per `(provider, model)`.
    rr_cursor: HashMap<(String, String), usize>,
}

pub struct KeyPool {
    providers: BTreeMap<String, ResolvedProvider>,
    inner: RwLock<PoolInner>,
    store: Store,
}

impl KeyPool {
    pub fn new(
        providers: BTreeMap<String, ResolvedProvider>,
        store: Store,
    ) -> Result<Self, StoreError> {
        let now = Utc::now();
        let mut states = HashMap::new();
        let mut stats = HashMap::new();

        for (pname, pcfg) in &providers {
            for k in &pcfg.keys {
                stats.insert((pname.clone(), k.key_id.clone()), KeyStats::default());
            }
        }

        for rec in store.load_all()? {
            let Some(pcfg) = providers.get(&rec.provider) else {
                continue;
            };
            if !pcfg.keys.iter().any(|k| k.key_id == rec.key_hash) {
                continue;
            }
            let health = match rec.state.as_str() {
                "invalid" => KeyHealth::Invalid,
                "cooling_down" => {
                    if rec.reset_at.is_some_and(|dt| dt <= now) {
                        continue;
                    }
                    KeyHealth::CoolingDown
                }
                _ => continue,
            };
            states.insert(
                (rec.provider, rec.key_hash, rec.model),
                EntryState {
                    health,
                    reset_at: rec.reset_at,
                    consecutive_hits: rec.consecutive_hits,
                    reason: rec.reason,
                },
            );
        }

        Ok(Self {
            providers,
            inner: RwLock::new(PoolInner {
                states,
                stats,
                sticky_index: HashMap::new(),
                rr_cursor: HashMap::new(),
            }),
            store,
        })
    }

    pub fn provider(&self, name: &str) -> Option<&ResolvedProvider> {
        self.providers.get(name)
    }

    pub fn select_key(
        &self,
        provider: &str,
        model: Option<&str>,
        exclude_key_ids: &HashSet<String>,
        now: DateTime<Utc>,
    ) -> Result<SelectedKey, SelectError> {
        let pcfg = self
            .providers
            .get(provider)
            .ok_or(SelectError::UnknownProvider)?;
        if pcfg.keys.is_empty() {
            return Err(SelectError::NoKeysConfigured);
        }

        let model_key = model.unwrap_or("").to_string();
        let mut inner = self.inner.write().expect("pool lock poisoned");
        let n = pcfg.keys.len();

        let start_idx = match pcfg.strategy {
            SelectionStrategy::Sticky => *inner
                .sticky_index
                .get(&(provider.to_string(), model_key.clone()))
                .unwrap_or(&0)
                % n,
            SelectionStrategy::RoundRobin => *inner
                .rr_cursor
                .get(&(provider.to_string(), model_key.clone()))
                .unwrap_or(&0)
                % n,
        };

        let mut earliest_reset: Option<DateTime<Utc>> = None;

        for offset in 0..n {
            let idx = (start_idx + offset) % n;
            let key = &pcfg.keys[idx];

            // Compute availability & track earliest reset even for excluded keys
            let key_block = check_and_prune_state(&mut inner.states, provider, &key.key_id, "", now);
            let model_block = if !model_key.is_empty() {
                check_and_prune_state(&mut inner.states, provider, &key.key_id, &model_key, now)
            } else {
                None
            };

            let effective_block = merge_blocks(key_block, model_block);
            if let Some(block) = effective_block {
                if let Some(reset_at) = block.reset_at {
                    earliest_reset = Some(match earliest_reset {
                        Some(prev) => prev.min(reset_at),
                        None => reset_at,
                    });
                }
                continue;
            }

            if exclude_key_ids.contains(&key.key_id) {
                continue;
            }

            match pcfg.strategy {
                SelectionStrategy::Sticky => {
                    inner
                        .sticky_index
                        .insert((provider.to_string(), model_key), idx);
                }
                SelectionStrategy::RoundRobin => {
                    inner
                        .rr_cursor
                        .insert((provider.to_string(), model_key), (idx + 1) % n);
                }
            }

            return Ok(SelectedKey {
                key_id: key.key_id.clone(),
                masked: key.masked.clone(),
                secret: key.secret.clone(),
            });
        }

        Err(SelectError::AllExhausted {
            next_reset_at: earliest_reset,
        })
    }

    pub fn previous_consecutive_hits(
        &self,
        provider: &str,
        key_id: &str,
        model: Option<&str>,
        scope: QuotaScope,
    ) -> u32 {
        let model_key = match scope {
            QuotaScope::Key => String::new(),
            QuotaScope::Model => model.unwrap_or("").to_string(),
        };
        let inner = self.inner.read().expect("pool lock poisoned");
        inner
            .states
            .get(&(provider.to_string(), key_id.to_string(), model_key))
            .map(|s| s.consecutive_hits)
            .unwrap_or(0)
    }

    pub async fn record_success(&self, provider: &str, key_id: &str, model: Option<&str>) {
        let model_key = model.unwrap_or("").to_string();
        let removed = {
            let mut inner = self.inner.write().expect("pool lock poisoned");
            inner
                .stats
                .entry((provider.to_string(), key_id.to_string()))
                .or_default()
                .requests += 1;
            inner
                .states
                .remove(&(provider.to_string(), key_id.to_string(), model_key.clone()))
                .is_some()
        };
        if removed {
            let _ = self
                .store
                .delete_async(provider.to_string(), key_id.to_string(), model_key)
                .await;
        }
    }

    pub async fn record_quota_hit(
        &self,
        provider: &str,
        key_id: &str,
        model: Option<&str>,
        scope: QuotaScope,
        reset_at: DateTime<Utc>,
        reason: String,
    ) {
        let model_key = match scope {
            QuotaScope::Key => String::new(),
            QuotaScope::Model => model.unwrap_or("").to_string(),
        };
        let now = Utc::now();
        let hits = {
            let mut inner = self.inner.write().expect("pool lock poisoned");
            let st = inner
                .stats
                .entry((provider.to_string(), key_id.to_string()))
                .or_default();
            st.requests += 1;
            st.rate_limits += 1;

            let map_key = (provider.to_string(), key_id.to_string(), model_key.clone());
            let next_hits = inner
                .states
                .get(&map_key)
                .map(|s| s.consecutive_hits.saturating_add(1))
                .unwrap_or(1);
            inner.states.insert(
                map_key,
                EntryState {
                    health: KeyHealth::CoolingDown,
                    reset_at: Some(reset_at),
                    consecutive_hits: next_hits,
                    reason: Some(reason.clone()),
                },
            );
            next_hits
        };

        let rec = StoredKeyState {
            provider: provider.to_string(),
            key_hash: key_id.to_string(),
            model: model_key,
            state: "cooling_down".to_string(),
            reset_at: Some(reset_at),
            consecutive_hits: hits,
            reason: Some(reason),
            updated_at: now,
        };
        if let Err(err) = self.store.upsert_async(rec).await {
            tracing::error!(error = %err, provider, key_id, "failed to persist cooldown state");
        }
    }

    pub async fn record_invalid_key(&self, provider: &str, key_id: &str, reason: String) {
        let now = Utc::now();
        {
            let mut inner = self.inner.write().expect("pool lock poisoned");
            let st = inner
                .stats
                .entry((provider.to_string(), key_id.to_string()))
                .or_default();
            st.requests += 1;
            st.errors += 1;

            inner.states.insert(
                (provider.to_string(), key_id.to_string(), String::new()),
                EntryState {
                    health: KeyHealth::Invalid,
                    reset_at: None,
                    consecutive_hits: 1,
                    reason: Some(reason.clone()),
                },
            );
        }

        let rec = StoredKeyState {
            provider: provider.to_string(),
            key_hash: key_id.to_string(),
            model: String::new(),
            state: "invalid".to_string(),
            reset_at: None,
            consecutive_hits: 1,
            reason: Some(reason),
            updated_at: now,
        };
        if let Err(err) = self.store.upsert_async(rec).await {
            tracing::error!(error = %err, provider, key_id, "failed to persist invalid key state");
        }
    }

    pub fn record_error(&self, provider: &str, key_id: &str) {
        let mut inner = self.inner.write().expect("pool lock poisoned");
        let st = inner
            .stats
            .entry((provider.to_string(), key_id.to_string()))
            .or_default();
        st.requests += 1;
        st.errors += 1;
    }

    pub async fn reset(
        &self,
        provider: Option<&str>,
        key_id: Option<&str>,
    ) -> Result<usize, StoreError> {
        let cleared_mem = {
            let mut inner = self.inner.write().expect("pool lock poisoned");
            let before = inner.states.len();
            inner.states.retain(|(p, k, _), _| {
                let p_match = provider.is_none_or(|pf| pf == p);
                let k_match = key_id.is_none_or(|kf| kf == k);
                !(p_match && k_match)
            });
            before - inner.states.len()
        };
        let cleared_db = self
            .store
            .reset_async(provider.map(str::to_string), key_id.map(str::to_string))
            .await?;
        Ok(cleared_mem.max(cleared_db))
    }

    pub fn status_snapshot(&self, now: DateTime<Utc>) -> StatusReport {
        let inner = self.inner.read().expect("pool lock poisoned");
        let mut providers_out = BTreeMap::new();

        for (pname, pcfg) in &self.providers {
            let mut keys_out = Vec::with_capacity(pcfg.keys.len());
            for km in &pcfg.keys {
                keys_out.push(build_key_report(pname, km, &inner, now));
            }
            providers_out.insert(
                pname.clone(),
                ProviderStatusReport {
                    preset: format!("{:?}", pcfg.preset).to_ascii_lowercase(),
                    strategy: format!("{:?}", pcfg.strategy).to_ascii_lowercase(),
                    base_url: pcfg.base_url.clone(),
                    keys: keys_out,
                },
            );
        }

        StatusReport {
            timestamp: now.to_rfc3339(),
            providers: providers_out,
        }
    }
}

#[derive(Debug, Clone)]
struct BlockInfo {
    health: KeyHealth,
    reset_at: Option<DateTime<Utc>>,
}

fn check_and_prune_state(
    states: &mut HashMap<(String, String, String), EntryState>,
    provider: &str,
    key_id: &str,
    model: &str,
    now: DateTime<Utc>,
) -> Option<BlockInfo> {
    let map_key = (provider.to_string(), key_id.to_string(), model.to_string());
    let should_remove = {
        let st = states.get(&map_key)?;
        match st.health {
            KeyHealth::Ready => true,
            KeyHealth::Invalid => {
                return Some(BlockInfo {
                    health: KeyHealth::Invalid,
                    reset_at: None,
                })
            }
            KeyHealth::CoolingDown => {
                if let Some(reset_at) = st.reset_at {
                    if reset_at > now {
                        return Some(BlockInfo {
                            health: KeyHealth::CoolingDown,
                            reset_at: Some(reset_at),
                        });
                    }
                    true
                } else {
                    true
                }
            }
        }
    };
    if should_remove {
        states.remove(&map_key);
    }
    None
}

fn merge_blocks(key_block: Option<BlockInfo>, model_block: Option<BlockInfo>) -> Option<BlockInfo> {
    match (key_block, model_block) {
        (None, None) => None,
        (Some(k), None) => Some(k),
        (None, Some(m)) => Some(m),
        (Some(k), Some(m)) => {
            if k.health == KeyHealth::Invalid {
                Some(k)
            } else {
                let reset_at = match (k.reset_at, m.reset_at) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                };
                Some(BlockInfo {
                    health: KeyHealth::CoolingDown,
                    reset_at,
                })
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub struct StatusReport {
    pub timestamp: String,
    pub providers: BTreeMap<String, ProviderStatusReport>,
}

#[derive(Debug, Serialize)]
pub struct ProviderStatusReport {
    pub preset: String,
    pub strategy: String,
    pub base_url: String,
    pub keys: Vec<KeyStatusReport>,
}

#[derive(Debug, Serialize)]
pub struct KeyStatusReport {
    pub key_id: String,
    pub masked: String,
    pub state: KeyHealth,
    pub reset_at: Option<String>,
    pub seconds_until_reset: Option<i64>,
    pub models: BTreeMap<String, ModelStateReport>,
    pub stats: KeyStats,
}

#[derive(Debug, Serialize)]
pub struct ModelStateReport {
    pub state: KeyHealth,
    pub reset_at: Option<String>,
    pub seconds_until_reset: Option<i64>,
    pub reason: Option<String>,
}

fn build_key_report(
    provider: &str,
    km: &KeyMaterial,
    inner: &PoolInner,
    now: DateTime<Utc>,
) -> KeyStatusReport {
    let key_level = inner
        .states
        .get(&(provider.to_string(), km.key_id.clone(), String::new()))
        .and_then(|st| active_state(st, now));

    let (state, reset_at, seconds_until_reset) = match key_level {
        Some((h, dt)) => (
            h,
            dt.map(|t| t.to_rfc3339()),
            dt.map(|t| (t - now).num_seconds().max(0)),
        ),
        None => (KeyHealth::Ready, None, None),
    };

    let mut models = BTreeMap::new();
    for ((p, k, m), st) in &inner.states {
        if p == provider && k == &km.key_id && !m.is_empty() {
            if let Some((mh, mdt)) = active_state(st, now) {
                models.insert(
                    m.clone(),
                    ModelStateReport {
                        state: mh,
                        reset_at: mdt.map(|t| t.to_rfc3339()),
                        seconds_until_reset: mdt.map(|t| (t - now).num_seconds().max(0)),
                        reason: st.reason.clone(),
                    },
                );
            }
        }
    }

    let stats = inner
        .stats
        .get(&(provider.to_string(), km.key_id.clone()))
        .cloned()
        .unwrap_or_default();

    KeyStatusReport {
        key_id: km.key_id.clone(),
        masked: km.masked.clone(),
        state,
        reset_at,
        seconds_until_reset,
        models,
        stats,
    }
}

fn active_state(st: &EntryState, now: DateTime<Utc>) -> Option<(KeyHealth, Option<DateTime<Utc>>)> {
    match st.health {
        KeyHealth::Ready => None,
        KeyHealth::Invalid => Some((KeyHealth::Invalid, None)),
        KeyHealth::CoolingDown => st
            .reset_at
            .filter(|dt| *dt > now)
            .map(|dt| (KeyHealth::CoolingDown, Some(dt))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ProviderConfig, ProviderPreset, RawConfig};
    use chrono::Duration;
    use std::path::Path;

    fn make_pool(strategy: SelectionStrategy) -> KeyPool {
        let mut raw = RawConfig::default();
        raw.providers.insert(
            "gemini".into(),
            ProviderConfig {
                preset: ProviderPreset::Gemini,
                strategy,
                keys: vec!["key-alpha-1234".into(), "key-beta-5678".into()],
                ..Default::default()
            },
        );
        let cfg = crate::config::AppConfig::from_raw(raw, &[]).unwrap();
        let store = Store::open(Path::new(":memory:")).unwrap();
        KeyPool::new(cfg.providers, store).unwrap()
    }

    #[test]
    fn sticky_vs_round_robin_selection() {
        let now = Utc::now();
        let empty = HashSet::new();

        let sticky = make_pool(SelectionStrategy::Sticky);
        let k1 = sticky.select_key("gemini", Some("m1"), &empty, now).unwrap();
        let k2 = sticky.select_key("gemini", Some("m1"), &empty, now).unwrap();
        assert_eq!(k1.key_id, k2.key_id);

        let rr = make_pool(SelectionStrategy::RoundRobin);
        let r1 = rr.select_key("gemini", Some("m1"), &empty, now).unwrap();
        let r2 = rr.select_key("gemini", Some("m1"), &empty, now).unwrap();
        assert_ne!(r1.key_id, r2.key_id);
    }

    #[tokio::test]
    async fn per_model_cooldown_isolation() {
        let pool = make_pool(SelectionStrategy::Sticky);
        let now = Utc::now();
        let empty = HashSet::new();

        let first = pool.select_key("gemini", Some("m1"), &empty, now).unwrap();
        pool.record_quota_hit(
            "gemini",
            &first.key_id,
            Some("m1"),
            QuotaScope::Model,
            now + Duration::seconds(60),
            "429".into(),
        )
        .await;

        let next_m1 = pool.select_key("gemini", Some("m1"), &empty, now).unwrap();
        assert_ne!(next_m1.key_id, first.key_id);

        // Same key should still be ready for m2
        let next_m2 = pool.select_key("gemini", Some("m2"), &empty, now).unwrap();
        assert_eq!(next_m2.key_id, first.key_id);
    }
}
