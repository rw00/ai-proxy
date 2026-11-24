use std::path::Path;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("background task failed: {0}")]
    Join(#[from] tokio::task::JoinError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredKeyState {
    pub provider: String,
    pub key_hash: String,
    /// Empty string `""` denotes key-level state (e.g. invalid key or account-wide quota).
    pub model: String,
    /// `"cooling_down"` or `"invalid"`.
    pub state: String,
    pub reset_at: Option<DateTime<Utc>>,
    pub consecutive_hits: u32,
    pub reason: Option<String>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Clone)]
pub struct Store {
    conn: Arc<Mutex<Connection>>,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self, StoreError> {
        if path.to_str() != Some(":memory:") {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
        }
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             CREATE TABLE IF NOT EXISTS key_states (
                 provider TEXT NOT NULL,
                 key_hash TEXT NOT NULL,
                 model TEXT NOT NULL,
                 state TEXT NOT NULL,
                 reset_at INTEGER,
                 consecutive_hits INTEGER NOT NULL DEFAULT 0,
                 reason TEXT,
                 updated_at INTEGER NOT NULL,
                 PRIMARY KEY (provider, key_hash, model)
             );",
        )?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn load_all(&self) -> Result<Vec<StoredKeyState>, StoreError> {
        let conn = self.conn.lock().expect("sqlite mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT provider, key_hash, model, state, reset_at, consecutive_hits, reason, updated_at
             FROM key_states",
        )?;
        let rows = stmt.query_map([], |row| {
            let reset_ts: Option<i64> = row.get(4)?;
            let updated_ts: i64 = row.get(7)?;
            Ok(StoredKeyState {
                provider: row.get(0)?,
                key_hash: row.get(1)?,
                model: row.get(2)?,
                state: row.get(3)?,
                reset_at: reset_ts.and_then(|ts| DateTime::from_timestamp(ts, 0)),
                consecutive_hits: row.get::<_, u32>(5)?,
                reason: row.get(6)?,
                updated_at: DateTime::from_timestamp(updated_ts, 0).unwrap_or_else(Utc::now),
            })
        })?;

        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn upsert(&self, rec: &StoredKeyState) -> Result<(), StoreError> {
        let conn = self.conn.lock().expect("sqlite mutex poisoned");
        conn.execute(
            "INSERT INTO key_states (provider, key_hash, model, state, reset_at, consecutive_hits, reason, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(provider, key_hash, model) DO UPDATE SET
                 state = excluded.state,
                 reset_at = excluded.reset_at,
                 consecutive_hits = excluded.consecutive_hits,
                 reason = excluded.reason,
                 updated_at = excluded.updated_at",
            params![
                rec.provider,
                rec.key_hash,
                rec.model,
                rec.state,
                rec.reset_at.map(|dt| dt.timestamp()),
                rec.consecutive_hits,
                rec.reason,
                rec.updated_at.timestamp(),
            ],
        )?;
        Ok(())
    }

    pub async fn upsert_async(&self, rec: StoredKeyState) -> Result<(), StoreError> {
        let store = self.clone();
        tokio::task::spawn_blocking(move || store.upsert(&rec)).await?
    }

    pub fn delete(&self, provider: &str, key_hash: &str, model: &str) -> Result<usize, StoreError> {
        let conn = self.conn.lock().expect("sqlite mutex poisoned");
        let n = conn.execute(
            "DELETE FROM key_states WHERE provider = ?1 AND key_hash = ?2 AND model = ?3",
            params![provider, key_hash, model],
        )?;
        Ok(n)
    }

    pub async fn delete_async(
        &self,
        provider: String,
        key_hash: String,
        model: String,
    ) -> Result<usize, StoreError> {
        let store = self.clone();
        tokio::task::spawn_blocking(move || store.delete(&provider, &key_hash, &model)).await?
    }

    pub fn reset(
        &self,
        provider: Option<&str>,
        key_hash: Option<&str>,
    ) -> Result<usize, StoreError> {
        let conn = self.conn.lock().expect("sqlite mutex poisoned");
        let n = match (provider, key_hash) {
            (Some(p), Some(k)) => conn.execute(
                "DELETE FROM key_states WHERE provider = ?1 AND key_hash = ?2",
                params![p, k],
            )?,
            (Some(p), None) => {
                conn.execute("DELETE FROM key_states WHERE provider = ?1", params![p])?
            }
            (None, Some(k)) => {
                conn.execute("DELETE FROM key_states WHERE key_hash = ?1", params![k])?
            }
            (None, None) => conn.execute("DELETE FROM key_states", [])?,
        };
        Ok(n)
    }

    pub async fn reset_async(
        &self,
        provider: Option<String>,
        key_hash: Option<String>,
    ) -> Result<usize, StoreError> {
        let store = self.clone();
        tokio::task::spawn_blocking(move || store.reset(provider.as_deref(), key_hash.as_deref()))
            .await?
    }
}
