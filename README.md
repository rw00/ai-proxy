# ai-proxy

Standalone, LLM-provider-agnostic API key rotation and quota-tracking HTTP proxy written in Rust.

## Features

- **Provider-agnostic adapters**: Built-in `gemini` (native `/v1beta/...` + OpenAI-compat `/v1beta/openai/...`) and `openai_compat` (custom `base_url` with `bearer`, `header:<name>`, or `query:<name>` auth).
- **Multi-key rotation**: Load keys from TOML/YAML or env (`<PROVIDER>_API_KEYS` CSV/JSON array, `<PROVIDER>_API_KEY_1..N`), `sticky` (default, prompt-cache friendly) or `round_robin` selection.
- **Quota & reset tracking**: Per-`(provider, key_hash, model)` cooldowns + key-level `invalid` state, persisted to SQLite (by `sha256` hash prefix; raw keys are never logged or stored).
- **Zero-buffer streaming**: Buffers request bodies once for lossless failover; streams `2xx` / SSE responses directly.
- **Security**: Binds `127.0.0.1:8080` with loopback `Host`/`Origin` validation; optional `API_TOKEN` Bearer auth (exempting `/health`).

## Quick Start

```bash
# Run with env vars only (auto-registers gemini/openai/anthropic when keys exist)
export GEMINI_API_KEYS="AIzaKey1,AIzaKey2"
cargo run --release

# Or run with a TOML/YAML config file
cp config.example.toml config.toml
cargo run --release -- --config config.toml
```

### Environment Variables

| Variable | Description |
| :--- | :--- |
| `AI_PROXY_CONFIG` | Path to `.toml` / `.yaml` / `.yml` config file |
| `AI_PROXY_BIND` | Bind address (default: `127.0.0.1:8080`) |
| `AI_PROXY_DB_PATH` | SQLite database path (default: `./ai-proxy.db`) |
| `AI_PROXY_API_TOKEN` / `API_TOKEN` | Optional Bearer token protecting all routes except `/health` |
| `AI_PROXY_DEFAULT_PROVIDER` | Default provider for `/v1/*` routes |
| `<PROVIDER>_API_KEYS` | Comma-separated keys or JSON array `["k1","k2"]` |
| `<PROVIDER>_API_KEY_1..N` | Numbered keys for `<PROVIDER>` |

## Configuration Example (`config.toml`)

```toml
[server]
bind = "127.0.0.1:8080"
db_path = "./ai-proxy.db"
default_provider = "gemini"

[routing.model_map]
"gpt-4o" = "openai"

[providers.gemini]
preset = "gemini"
strategy = "sticky" # or "round_robin"
fallback_cooldown_secs = 60
backoff_multiplier = 2.0
max_backoff_secs = 3600

[providers.openai]
preset = "openai_compat"
base_url = "https://api.openai.com"
auth = "bearer"

[providers.anthropic]
preset = "openai_compat"
base_url = "https://api.anthropic.com"
auth = "header:x-api-key"
```

## HTTP API

| Endpoint | Purpose |
| :--- | :--- |
| `ANY /p/{provider}/{*path}` | Transparent provider passthrough (method, query, SSE) |
| `POST /v1/chat/completions` | Unified OpenAI-compat chat (`provider/model`, `model_map`, or `default_provider`) |
| `POST /v1/embeddings` | Unified OpenAI-compat embeddings |
| `GET /v1/models` | Unified OpenAI-compat models (`?provider=<name>` or `default_provider`) |
| `GET /health` | Health check (always unauthenticated) |
| `GET /status` | Providers, masked keys, per-model cooldowns, `reset_at`, counters |
| `POST /status/reset` | Clear cooldowns (`{}` for all, or `{"provider":"gemini","key_id":"..."}`) |

## Client Setup

### Native Gemini (`GOOGLE_GEMINI_BASE_URL`)

```bash
export GOOGLE_GEMINI_BASE_URL="http://127.0.0.1:8080/p/gemini"
export GEMINI_API_KEY="dummy"
```

### OpenAI SDK (Python / Node)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="dummy", # or your API_TOKEN if configured
)
resp = client.chat.completions.create(
    model="gemini/gemini-2.5-pro",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://127.0.0.1:8080/p/anthropic",
    api_key="dummy",
)
```

### `curl`

```bash
# Native Gemini SSE passthrough
curl -N "http://127.0.0.1:8080/p/gemini/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"Hi"}]}]}'

# Unified OpenAI-compatible route
curl "http://127.0.0.1:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini/gemini-2.5-flash","messages":[{"role":"user","content":"Hi"}]}'

# Inspect status & reset cooldowns
curl "http://127.0.0.1:8080/status"
curl -X POST "http://127.0.0.1:8080/status/reset" -H "Content-Type: application/json" -d '{"provider":"gemini"}'
```

## Verification

```bash
cargo test
```
