use std::path::PathBuf;

use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

use ai_proxy::{build_router, AppConfig, AppState, KeyPool, Store};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config_path = parse_config_arg().or_else(|| {
        std::env::var("AI_PROXY_CONFIG")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .map(PathBuf::from)
    });

    let config = AppConfig::load(config_path.as_deref())?;
    let bind_addr = config.server.bind.clone();
    let db_path = PathBuf::from(&config.server.db_path);

    let store = Store::open(&db_path)?;
    let pool = KeyPool::new(config.providers.clone(), store)?;
    let state = AppState::new(config, pool);

    tracing::info!(
        bind = %bind_addr,
        providers = state.config.providers.len(),
        db = %db_path.display(),
        loopback_guard = state.enforce_loopback_origin,
        token_auth = state.config.server.api_token.is_some(),
        "starting ai-proxy"
    );

    let app = build_router(state);
    let listener = TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

fn parse_config_arg() -> Option<PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--config" || arg == "-c" {
            return args.next().map(PathBuf::from);
        }
        if let Some(rest) = arg.strip_prefix("--config=") {
            return Some(PathBuf::from(rest));
        }
    }
    None
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sig.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("shutdown signal received");
}
