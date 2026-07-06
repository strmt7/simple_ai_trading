#pragma once

namespace simple_ai_trading::native_contract {

struct CommandOptionSpec {
    const wchar_t* flags;
    const wchar_t* dest;
    const wchar_t* choices;
    const wchar_t* default_value;
    const wchar_t* help;
    bool required;
    bool takes_value;
};

struct CommandSpec {
    const wchar_t* name;
    const wchar_t* help;
    const CommandOptionSpec* options;
    int option_count;
};

inline constexpr CommandOptionSpec kOptions_ai[] = {
    {L"--enable", L"enable", L"", L"", L"enable AI decision features", false, false},
    {L"--disable", L"disable", L"", L"", L"disable AI decision features", false, false},
    {L"--provider", L"provider", L"", L"", L"AI provider: auto, local-gpu, ollama, openai-compatible, etc.", false, true},
    {L"--model", L"model", L"", L"", L"AI model identifier or 'auto'", false, true},
    {L"--require-gpu", L"require_gpu", L"", L"", L"", false, false},
    {L"--no-require-gpu", L"no_require_gpu", L"", L"", L"", false, false},
    {L"--min-free-vram-gb", L"min_free_vram_gb", L"", L"", L"", false, true},
    {L"--min-free-ram-gb", L"min_free_ram_gb", L"", L"", L"", false, true},
    {L"--min-model-parameters-b", L"min_model_parameters_b", L"", L"", L"", false, true},
    {L"--allow-paper-fallback", L"allow_paper_fallback", L"", L"", L"", false, false},
    {L"--no-paper-fallback", L"no_paper_fallback", L"", L"", L"", false, false},
    {L"--json", L"json", L"", L"false", L"", false, false},
};

inline constexpr CommandOptionSpec kOptions_ai_review[] = {
    {L"--report", L"report", L"", L"data/model_lab/model_lab_report.json", L"", false, true},
    {L"--output", L"output", L"", L"", L"", false, true},
    {L"--model", L"model", L"", L"", L"", false, true},
    {L"--url", L"url", L"", L"http://127.0.0.1:11434", L"", false, true},
    {L"--timeout", L"timeout", L"", L"20.0", L"", false, true},
    {L"--json", L"json", L"", L"false", L"", false, false},
};

inline constexpr CommandOptionSpec kOptions_api_budget[] = {
    {L"--db", L"db", L"", L"data/market_data.sqlite", L"", false, true},
    {L"--market", L"market", L"spot, futures", L"", L"", false, true},
    {L"--refresh", L"refresh", L"", L"false", L"query Binance exchangeInfo once and cache the latest headers", false, false},
    {L"--cached-only", L"cached_only", L"", L"false", L"do not refresh even when the cached sample is stale", false, false},
    {L"--max-age-seconds", L"max_age_seconds", L"", L"90", L"automatic refresh threshold for cached status", false, true},
    {L"--compact", L"compact", L"", L"false", L"print one status-bar friendly line", false, false},
    {L"--json", L"json", L"", L"false", L"", false, false},
};

inline constexpr CommandOptionSpec kOptions_archive_sync[] = {
    {L"--db", L"db", L"", L"data/market_data.sqlite", L"", false, true},
    {L"--symbol", L"symbol", L"", L"", L"", false, true},
    {L"--symbols", L"symbols", L"", L"", L"comma-separated symbols; overrides --symbol", false, true},
    {L"--top-symbols", L"top_symbols", L"", L"0", L"auto-rank this many high-liquidity symbols", false, true},
    {L"--quote-asset", L"quote_asset", L"", L"", L"quote asset used with --top-symbols", false, true},
    {L"--max-scan", L"max_scan", L"", L"250", L"maximum universe candidates scanned with --top-symbols", false, true},
    {L"--min-history-months", L"min_history_months", L"", L"0", L"with --top-symbols and monthly cadence, require this many monthly archive files before selecting a symbol", false, true},
    {L"--interval", L"interval", L"", L"", L"", false, true},
    {L"--market", L"market", L"spot, futures", L"spot", L"", false, true},
    {L"--cadence", L"cadence", L"monthly, daily", L"monthly", L"", false, true},
    {L"--data-type", L"data_type", L"klines, aggTrades", L"", L"official archive data type; futures 1s defaults to aggTrades and aggregates real trades to 1s candles", false, true},
    {L"--max-files", L"max_files", L"", L"", L"optional safety cap for smoke runs", false, true},
    {L"--start-period", L"start_period", L"", L"", L"inclusive archive period start, YYYY-MM or YYYY-MM-DD", false, true},
    {L"--end-period", L"end_period", L"", L"", L"inclusive archive period end, YYYY-MM or YYYY-MM-DD", false, true},
    {L"--plan-only", L"plan_only", L"", L"false", L"list the bounded archive plan without downloading files", false, false},
    {L"--max-planned-gb", L"max_planned_gb", L"", L"50.0", L"block non-plan archive downloads above this planned S3 ZIP size; use 0 to disable", false, true},
    {L"--timeout", L"timeout", L"", L"120", L"", false, true},
    {L"--force", L"force", L"", L"false", L"", false, false},
    {L"--no-verify-checksum", L"no_verify_checksum", L"", L"false", L"skip Binance .CHECKSUM sidecar verification", false, false},
    {L"--require-checksum", L"require_checksum", L"", L"false", L"fail archive files without a readable .CHECKSUM sidecar", false, false},
    {L"--json", L"json", L"", L"false", L"", false, false},
};

inline constexpr CommandOptionSpec kOptions_audit[] = {
    {L"--input", L"input", L"", L"data/historical_btcusdc.json", L"", false, true},
    {L"--model", L"model", L"", L"data/model.json", L"", false, true},
};

inline constexpr CommandOptionSpec kOptions_autonomous[] = {
    {L"--objective", L"objective", L"", L"conservative", L"", false, true},
    {L"--model", L"model", L"", L"data/model.json", L"model artifact used for autonomous decisions", false, true},
    {L"--poll-seconds", L"poll_seconds", L"", L"30.0", L"seconds between autonomous iterations", false, true},
    {L"--iterations", L"iterations", L"", L"", L"stop after N iterations; default runs until stopped", false, true},
    {L"--heartbeat-every", L"heartbeat_every", L"", L"1", L"write heartbeat every N iterations", false, true},
    {L"--starting-cash", L"starting_cash", L"", L"1000.0", L"reference cash for local autonomous risk stats", false, true},
    {L"--paper", L"paper", L"", L"false", L"force autonomous paper mode", false, false},
    {L"--live", L"live", L"", L"false", L"force authenticated non-mainnet autonomous mode", false, false},
    {L"action", L"action", L"start, pause, resume, stop, status", L"", L"autonomous action to perform", true, true},
};

inline constexpr CommandOptionSpec kOptions_backtest[] = {
    {L"--input", L"input", L"", L"data/historical_btcusdc.json", L"", false, true},
    {L"--model", L"model", L"", L"data/model.json", L"", false, true},
    {L"--start-cash", L"start_cash", L"", L"1000.0", L"", false, true},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"model-scoring backend override; default uses saved runtime compute_backend", false, true},
    {L"--score-batch-size", L"score_batch_size", L"", L"8192", L"batch size for GPU-assisted probability scoring", false, true},
    {L"--execution-db", L"execution_db", L"", L"", L"optional SQLite market-data DB; latest typed top-of-book row becomes symbol-specific fill stress", false, true},
};

inline constexpr CommandOptionSpec kOptions_backtest_chart[] = {
    {L"--input", L"input", L"", L"data/historical_btcusdc.json", L"", false, true},
    {L"--model", L"model", L"", L"data/model.json", L"", false, true},
    {L"--output", L"output", L"", L"data/backtest_performance.svg", L"", false, true},
    {L"--start-cash", L"start_cash", L"", L"1000.0", L"", false, true},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"", false, true},
    {L"--score-batch-size", L"score_batch_size", L"", L"8192", L"", false, true},
    {L"--execution-db", L"execution_db", L"", L"", L"optional SQLite market-data DB for symbol-specific top-of-book fill stress", false, true},
};

inline constexpr CommandOptionSpec kOptions_backtest_panel[] = {
    {L"--interval", L"interval", L"", L"", L"", true, true},
    {L"--market", L"market", L"", L"", L"override runtime market type", false, true},
    {L"--from-date", L"from_date", L"", L"", L"", false, true},
    {L"--to-date", L"to_date", L"", L"", L"", false, true},
    {L"--input", L"input", L"", L"data/historical_btcusdc.json", L"", false, true},
    {L"--model", L"model", L"", L"", L"", false, true},
    {L"--objective", L"objective", L"", L"", L"", false, true},
    {L"--tag", L"tag", L"", L"", L"", false, true},
    {L"--notes", L"notes", L"", L"", L"", false, true},
    {L"--starting-cash", L"starting_cash", L"", L"1000.0", L"", false, true},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"feature/scoring backend override; default uses saved runtime compute_backend", false, true},
    {L"--execution-db", L"execution_db", L"", L"", L"optional SQLite market-data DB for symbol-specific top-of-book fill stress", false, true},
};

inline constexpr CommandOptionSpec kOptions_close[] = {
    {L"position_id", L"position_id", L"", L"", L"position id or 'all'", true, true},
};

inline constexpr CommandOptionSpec kOptions_compute[] = {
    {L"--backend", L"backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"", false, true},
};

inline constexpr CommandOptionSpec kOptions_coordinator[] = {
    {L"--model", L"model", L"", L"data/model.json", L"", false, true},
    {L"--positions-root", L"positions_root", L"", L"data/autonomous", L"", false, true},
    {L"--json", L"json", L"", L"false", L"", false, false},
};

inline constexpr CommandOptionSpec kOptions_data_health[] = {
    {L"--db", L"db", L"", L"data/market_data.sqlite", L"", false, true},
    {L"--symbol", L"symbol", L"", L"", L"", false, true},
    {L"--symbols", L"symbols", L"", L"", L"comma-separated symbols; defaults to stored series", false, true},
    {L"--interval", L"interval", L"", L"", L"", false, true},
    {L"--market", L"market", L"spot, futures", L"", L"", false, true},
    {L"--min-rows", L"min_rows", L"", L"0", L"", false, true},
    {L"--min-coverage-ratio", L"min_coverage_ratio", L"", L"0.995", L"", false, true},
    {L"--max-gap-count", L"max_gap_count", L"", L"0", L"", false, true},
    {L"--require-verified-checksum", L"require_verified_checksum", L"", L"false", L"", false, false},
    {L"--json", L"json", L"", L"false", L"", false, false},
};

inline constexpr CommandOptionSpec kOptions_data_sync[] = {
    {L"--db", L"db", L"", L"data/market_data.sqlite", L"", false, true},
    {L"--symbol", L"symbol", L"", L"", L"", false, true},
    {L"--interval", L"interval", L"", L"", L"", false, true},
    {L"--market", L"market", L"spot, futures", L"", L"", false, true},
    {L"--rows", L"rows", L"", L"500", L"", false, true},
    {L"--full-history", L"full_history", L"", L"false", L"page historical klines backward until the exchange has no older closed candles", false, false},
    {L"--batch-size", L"batch_size", L"", L"1000", L"", false, true},
    {L"--include-futures-metrics", L"include_futures_metrics", L"", L"true", L"", false, false},
    {L"--no-include-futures-metrics", L"include_futures_metrics", L"", L"true", L"", false, false},
    {L"--loop", L"loop", L"", L"false", L"keep syncing in the foreground", false, false},
    {L"--iterations", L"iterations", L"", L"1", L"foreground loop iterations; 0 means unlimited", false, true},
    {L"--sleep", L"sleep", L"", L"300", L"seconds between loop iterations", false, true},
    {L"--background", L"background", L"", L"false", L"start a detached downloader process", false, false},
    {L"--pid-file", L"pid_file", L"", L"data/market_data_sync.pid", L"", false, true},
    {L"--log-file", L"log_file", L"", L"data/market_data_sync.log", L"", false, true},
    {L"--json", L"json", L"", L"false", L"", false, false},
};

inline constexpr CommandOptionSpec kOptions_doctor[] = {
    {L"--input", L"input", L"", L"data/historical_btcusdc.json", L"", false, true},
    {L"--model", L"model", L"", L"data/model.json", L"", false, true},
    {L"--online", L"online", L"", L"false", L"also check exchange connectivity", false, false},
};

inline constexpr CommandOptionSpec kOptions_evaluate[] = {
    {L"--input", L"input", L"", L"data/historical_btcusdc.json", L"", false, true},
    {L"--model", L"model", L"", L"data/model.json", L"", false, true},
    {L"--threshold", L"threshold", L"", L"", L"", false, true},
    {L"--calibrate-threshold", L"calibrate_threshold", L"", L"false", L"", false, false},
};

inline constexpr CommandOptionSpec kOptions_fetch[] = {
    {L"--symbol", L"symbol", L"", L"", L"", false, true},
    {L"--interval", L"interval", L"", L"", L"", false, true},
    {L"--limit", L"limit", L"", L"500", L"", false, true},
    {L"--batch-size", L"batch_size", L"", L"1000", L"klines per request (spot max 1000, futures max 1500)", false, true},
    {L"--output", L"output", L"", L"data/historical_btcusdc.json", L"", false, true},
};

inline constexpr CommandOptionSpec kOptions_live[] = {
    {L"--model", L"model", L"", L"data/model.json", L"", false, true},
    {L"--steps", L"steps", L"", L"20", L"", false, true},
    {L"--sleep", L"sleep", L"", L"5", L"", false, true},
    {L"--leverage", L"leverage", L"", L"", L"override leverage for this run (futures only)", false, true},
    {L"--retrain-interval", L"retrain_interval", L"", L"0", L"retrain model every N steps (0 disables, for adaptive paper/live behavior)", false, true},
    {L"--retrain-window", L"retrain_window", L"", L"300", L"number of recent rows used for each live retrain", false, true},
    {L"--retrain-min-rows", L"retrain_min_rows", L"", L"240", L"minimum rows required before a retrain is attempted", false, true},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"", false, true},
    {L"--batch-size", L"batch_size", L"", L"8192", L"mini-batch size for live retraining", false, true},
    {L"--paper", L"paper", L"", L"false", L"force paper mode for this run even when runtime.dry_run is false", false, false},
    {L"--live", L"live", L"", L"false", L"force authenticated testnet execution even when runtime.dry_run is true", false, false},
    {L"--external-signals", L"external_signals", L"", L"", L"enable cached free external signal adjustment for this run", false, false},
    {L"--no-external-signals", L"external_signals", L"", L"true", L"disable cached free external signal adjustment for this run", false, false},
};

inline constexpr CommandOptionSpec kOptions_model_blueprint[] = {
    {L"--risk-level", L"risk_level", L"conservative, regular, aggressive, default, balanced, risky", L"", L"filter the roadmap to one risk level", false, true},
    {L"--implemented-only", L"implemented_only", L"", L"false", L"hide research-only, blocked, and sandbox model families", false, false},
    {L"--json", L"json", L"", L"false", L"", false, false},
};

inline constexpr CommandOptionSpec kOptions_model_lab[] = {
    {L"--output-dir", L"output_dir", L"", L"data/model_lab", L"", false, true},
    {L"--starting-cash", L"starting_cash", L"", L"1000.0", L"", false, true},
    {L"--objective", L"objective", L"", L"", L"objective/risk level to run; repeatable", false, true},
    {L"--max-symbols", L"max_symbols", L"", L"6", L"", false, true},
    {L"--max-scan", L"max_scan", L"", L"250", L"", false, true},
    {L"--limit", L"limit", L"", L"1000", L"candles per selected symbol", false, true},
    {L"--quote-asset", L"quote_asset", L"", L"", L"override runtime quote asset for this lab run", false, true},
    {L"--interval", L"interval", L"", L"", L"override runtime interval for this lab run", false, true},
    {L"--full-history", L"full_history", L"", L"false", L"page klines backward for each selected symbol until no older closed candles are returned", false, false},
    {L"--market-db", L"market_db", L"", L"", L"SQLite market-data database to train from instead of exchange API klines", false, true},
    {L"--require-db-data", L"require_db_data", L"", L"false", L"force model-lab to train from SQLite market data; defaults to data/market_data.sqlite when --market-db is omitted", false, false},
    {L"--market", L"market", L"spot, futures", L"", L"override runtime market type for this lab run", false, true},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"", false, true},
    {L"--batch-size", L"batch_size", L"", L"8192", L"", false, true},
    {L"--score-batch-size", L"score_batch_size", L"", L"", L"", false, true},
    {L"--max-candidates", L"max_candidates", L"", L"", L"smoke/research cap per objective; default evaluates the full grid", false, true},
    {L"--learning-feedback", L"learning_feedback", L"", L"", L"optional learning_feedback.json artifact; default uses data/autonomous/learning_feedback.json when present", false, true},
};

inline constexpr CommandOptionSpec kOptions_positions[] = {
    {L"--stats", L"stats", L"", L"false", L"also print realized + unrealized stats", false, false},
    {L"--learning", L"learning", L"", L"false", L"also print bounded post-trade learning feedback", false, false},
};

inline constexpr CommandOptionSpec kOptions_prepare[] = {
    {L"--historical", L"historical", L"", L"data/historical_btcusdc.json", L"", false, true},
    {L"--model", L"model", L"", L"data/model.json", L"", false, true},
    {L"--limit", L"limit", L"", L"500", L"", false, true},
    {L"--batch-size", L"batch_size", L"", L"1000", L"klines per fetch request (spot max 1000, futures max 1500)", false, true},
    {L"--preset", L"preset", L"balanced, custom, quick, thorough", L"balanced", L"", false, true},
    {L"--epochs", L"epochs", L"", L"", L"override preset training epochs", false, true},
    {L"--learning-rate", L"learning_rate", L"", L"0.05", L"", false, true},
    {L"--l2-penalty", L"l2_penalty", L"", L"0.0001", L"", false, true},
    {L"--seed", L"seed", L"", L"7", L"", false, true},
    {L"--start-cash", L"start_cash", L"", L"1000.0", L"", false, true},
    {L"--walk-forward", L"walk_forward", L"", L"", L"force walk-forward validation", false, false},
    {L"--no-walk-forward", L"walk_forward", L"", L"", L"skip walk-forward validation", false, false},
    {L"--walk-forward-train", L"walk_forward_train", L"", L"", L"override walk-forward training window", false, true},
    {L"--walk-forward-test", L"walk_forward_test", L"", L"", L"override walk-forward test window", false, true},
    {L"--walk-forward-step", L"walk_forward_step", L"", L"", L"override walk-forward step", false, true},
    {L"--calibrate-threshold", L"calibrate_threshold", L"", L"", L"force threshold calibration", false, false},
    {L"--no-calibrate-threshold", L"calibrate_threshold", L"", L"", L"skip threshold calibration", false, false},
    {L"--online-doctor", L"online_doctor", L"", L"false", L"include exchange connectivity in final readiness checks", false, false},
};

inline constexpr CommandOptionSpec kOptions_reconcile[] = {
    {L"--json", L"json", L"", L"false", L"", false, false},
    {L"--output", L"output", L"", L"data/autonomous/reconciliation.json", L"", false, true},
    {L"--quantity-tolerance", L"quantity_tolerance", L"", L"1e-08", L"", false, true},
};

inline constexpr CommandOptionSpec kOptions_report[] = {
    {L"--account", L"account", L"", L"false", L"include authenticated account state", false, false},
    {L"--doctor", L"doctor", L"", L"true", L"include readiness checks", false, false},
    {L"--no-doctor", L"doctor", L"", L"true", L"omit readiness checks", false, false},
    {L"--online", L"online", L"", L"false", L"include exchange connectivity in readiness checks", false, false},
    {L"--input", L"input", L"", L"data/historical_btcusdc.json", L"", false, true},
    {L"--model", L"model", L"", L"data/model.json", L"", false, true},
};

inline constexpr CommandOptionSpec kOptions_risk[] = {
    {L"--model", L"model", L"", L"data/model.json", L"", false, true},
    {L"--paper", L"paper", L"", L"false", L"assess paper/dry-run execution", false, false},
    {L"--live", L"live", L"", L"false", L"assess authenticated testnet/demo execution", false, false},
    {L"--leverage", L"leverage", L"", L"", L"optional futures leverage override", false, true},
    {L"--json", L"json", L"", L"false", L"", false, false},
};

inline constexpr CommandOptionSpec kOptions_signals[] = {
    {L"--model", L"model", L"", L"data/model.json", L"model path used to derive default cache location", false, true},
    {L"--cache", L"cache", L"", L"", L"signal cache path (default: model-adjacent data/signals)", false, true},
    {L"--ttl", L"ttl", L"", L"300", L"cache TTL seconds", false, true},
    {L"--timeout", L"timeout", L"", L"3.0", L"per-provider timeout seconds", false, true},
    {L"--max-adjustment", L"max_adjustment", L"", L"0.04", L"maximum model score adjustment", false, true},
    {L"--min-providers", L"min_providers", L"", L"2", L"minimum usable providers for positive boosts", false, true},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"optional backend for news keyword scoring", false, true},
    {L"--short-reaction-refresh", L"short_reaction_refresh", L"", L"30", L"seconds after which cached short-horizon reaction news must refresh", false, true},
    {L"--news-provider-limit", L"news_provider_limit", L"", L"", L"maximum RSS/news providers to poll", false, true},
    {L"--news-items-per-provider", L"news_items_per_provider", L"", L"", L"feed items scored per news provider", false, true},
    {L"--provider-parallelism", L"provider_parallelism", L"", L"", L"maximum simultaneous news provider requests", false, true},
    {L"--provider-jitter", L"provider_jitter", L"", L"", L"random per-provider delay ceiling in seconds", false, true},
    {L"--ollama-news", L"ollama_news", L"", L"", L"enable Ollama AI headline evaluation", false, false},
    {L"--no-ollama-news", L"ollama_news", L"", L"true", L"disable Ollama AI headline evaluation", false, false},
    {L"--ollama-model", L"ollama_model", L"", L"", L"", false, true},
    {L"--ollama-url", L"ollama_url", L"", L"", L"", false, true},
    {L"--ollama-timeout", L"ollama_timeout", L"", L"", L"", false, true},
    {L"--telemetry-db", L"telemetry_db", L"", L"", L"SQLite raw telemetry DB path", false, true},
    {L"--source-grade-max-age-hours", L"source_grade_max_age_hours", L"", L"", L"ignore source grades older than this; 0 disables the age cap", false, true},
    {L"--no-telemetry", L"no_telemetry", L"", L"false", L"do not journal raw provider/model payloads", false, false},
    {L"--loop", L"loop", L"", L"false", L"poll repeatedly with jitter instead of one collection", false, false},
    {L"--iterations", L"iterations", L"", L"0", L"loop iterations; 0 means until interrupted", false, true},
    {L"--sleep", L"sleep", L"", L"", L"base loop interval seconds", false, true},
    {L"--jitter", L"jitter", L"", L"", L"random loop delay ceiling in seconds", false, true},
    {L"--refresh", L"refresh", L"", L"false", L"ignore cache and fetch every provider", false, false},
    {L"--json", L"json", L"", L"false", L"print machine-readable report", false, false},
};

inline constexpr CommandOptionSpec kOptions_signals_benchmark[] = {
    {L"--provider-limit", L"provider_limit", L"", L"", L"", false, true},
    {L"--parallelism", L"parallelism", L"", L"", L"", false, true},
    {L"--iterations", L"iterations", L"", L"1", L"", false, true},
    {L"--timeout", L"timeout", L"", L"3.0", L"", false, true},
    {L"--provider-jitter", L"provider_jitter", L"", L"0.0", L"", false, true},
    {L"--ollama-news", L"ollama_news", L"", L"", L"", false, false},
    {L"--no-ollama-news", L"ollama_news", L"", L"true", L"", false, false},
    {L"--ollama-model", L"ollama_model", L"", L"", L"", false, true},
    {L"--ollama-url", L"ollama_url", L"", L"", L"", false, true},
    {L"--ollama-timeout", L"ollama_timeout", L"", L"", L"", false, true},
    {L"--cache", L"cache", L"", L"data/signals/benchmark_external_signals.json", L"", false, true},
    {L"--no-telemetry", L"no_telemetry", L"", L"false", L"", false, false},
    {L"--json", L"json", L"", L"false", L"", false, false},
};

inline constexpr CommandOptionSpec kOptions_source_grades[] = {
    {L"--db", L"db", L"", L"", L"SQLite raw telemetry DB path", false, true},
    {L"--window-hours", L"window_hours", L"", L"", L"", false, true},
    {L"--ollama", L"ollama", L"", L"", L"enable Ollama grading", false, false},
    {L"--no-ollama", L"ollama", L"", L"true", L"disable Ollama grading", false, false},
    {L"--ollama-model", L"ollama_model", L"", L"", L"", false, true},
    {L"--ollama-url", L"ollama_url", L"", L"", L"", false, true},
    {L"--ollama-timeout", L"ollama_timeout", L"", L"", L"", false, true},
    {L"--json", L"json", L"", L"false", L"", false, false},
};

inline constexpr CommandOptionSpec kOptions_spot_roundtrip[] = {
    {L"--quantity", L"quantity", L"", L"8e-05", L"base-asset quantity to test", false, true},
    {L"--mode", L"mode", L"auto, buy-sell, sell-buy", L"auto", L"order sequence; auto buys first when quote balance is available, otherwise sells first when base balance is available", false, true},
    {L"--yes", L"yes", L"", L"false", L"confirm signed testnet/demo order placement", false, false},
};

inline constexpr CommandOptionSpec kOptions_strategy[] = {
    {L"--profile", L"profile", L"active, aggressive, balanced, conservative, custom, regular", L"custom", L"", false, true},
    {L"--risk-level", L"risk_level", L"conservative, regular, aggressive", L"", L"", false, true},
    {L"--reinvest-profits", L"reinvest_profits", L"", L"", L"", false, false},
    {L"--no-reinvest-profits", L"no_reinvest_profits", L"", L"", L"", false, false},
    {L"--leverage", L"leverage", L"", L"", L"", false, true},
    {L"--risk", L"risk", L"", L"", L"", false, true},
    {L"--max-position", L"max_position", L"", L"", L"", false, true},
    {L"--stop", L"stop", L"", L"", L"", false, true},
    {L"--take", L"take", L"", L"", L"", false, true},
    {L"--cooldown", L"cooldown", L"", L"", L"", false, true},
    {L"--max-open", L"max_open", L"", L"", L"", false, true},
    {L"--min-diversified-assets", L"min_diversified_assets", L"", L"", L"", false, true},
    {L"--max-asset-allocation", L"max_asset_allocation", L"", L"", L"", false, true},
    {L"--max-portfolio-risk", L"max_portfolio_risk", L"", L"", L"", false, true},
    {L"--min-quote-volume-usdc", L"min_quote_volume_usdc", L"", L"", L"", false, true},
    {L"--min-trade-count-24h", L"min_trade_count_24h", L"", L"", L"", false, true},
    {L"--max-spread-bps", L"max_spread_bps", L"", L"", L"", false, true},
    {L"--min-liquidity-score", L"min_liquidity_score", L"", L"", L"", false, true},
    {L"--unpredictability-cooldown", L"unpredictability_cooldown", L"", L"", L"", false, true},
    {L"--max-regime-unpredictability", L"max_regime_unpredictability", L"", L"", L"", false, true},
    {L"--max-prediction-entropy", L"max_prediction_entropy", L"", L"", L"", false, true},
    {L"--min-model-confidence", L"min_model_confidence", L"", L"", L"", false, true},
    {L"--max-trades-per-day", L"max_trades_per_day", L"", L"", L"", false, true},
    {L"--signal-threshold", L"signal_threshold", L"", L"", L"", false, true},
    {L"--max-drawdown", L"max_drawdown", L"", L"", L"", false, true},
    {L"--max-daily-loss", L"max_daily_loss", L"", L"", L"", false, true},
    {L"--max-session-loss", L"max_session_loss", L"", L"", L"", false, true},
    {L"--max-consecutive-losses", L"max_consecutive_losses", L"", L"", L"", false, true},
    {L"--max-network-errors", L"max_network_errors", L"", L"", L"", false, true},
    {L"--recovery-cooldown-seconds", L"recovery_cooldown_seconds", L"", L"", L"", false, true},
    {L"--taker-fee-bps", L"taker_fee_bps", L"", L"", L"", false, true},
    {L"--slippage-bps", L"slippage_bps", L"", L"", L"", false, true},
    {L"--label-threshold", L"label_threshold", L"", L"", L"", false, true},
    {L"--model-lookback", L"model_lookback", L"", L"", L"", false, true},
    {L"--training-epochs", L"training_epochs", L"", L"", L"", false, true},
    {L"--confidence-beta", L"confidence_beta", L"", L"", L"", false, true},
    {L"--feature-window-short", L"feature_window_short", L"", L"", L"", false, true},
    {L"--feature-window-long", L"feature_window_long", L"", L"", L"", false, true},
    {L"--set-features", L"set_features", L"", L"", L"comma-separated ordered feature list for retraining", false, true},
    {L"--enable-feature", L"enable_feature", L"", L"", L"enable a feature by name", false, true},
    {L"--disable-feature", L"disable_feature", L"", L"", L"disable a feature by name", false, true},
    {L"--external-signals", L"external_signals", L"", L"", L"enable live free external signals", false, false},
    {L"--no-external-signals", L"external_signals", L"", L"true", L"disable live free external signals", false, false},
    {L"--external-signal-max-adjustment", L"external_signal_max_adjustment", L"", L"", L"", false, true},
    {L"--external-signal-min-providers", L"external_signal_min_providers", L"", L"", L"", false, true},
    {L"--external-signal-ttl", L"external_signal_ttl", L"", L"", L"", false, true},
    {L"--external-signal-timeout", L"external_signal_timeout", L"", L"", L"", false, true},
    {L"--external-news-ai", L"external_news_ai", L"", L"", L"", false, false},
    {L"--no-external-news-ai", L"external_news_ai", L"", L"true", L"", false, false},
    {L"--external-news-ai-model", L"external_news_ai_model", L"", L"", L"", false, true},
    {L"--external-news-ai-url", L"external_news_ai_url", L"", L"", L"", false, true},
    {L"--external-news-ai-timeout", L"external_news_ai_timeout", L"", L"", L"", false, true},
    {L"--external-news-provider-limit", L"external_news_provider_limit", L"", L"", L"", false, true},
    {L"--external-provider-parallelism", L"external_provider_parallelism", L"", L"", L"", false, true},
    {L"--external-provider-jitter", L"external_provider_jitter", L"", L"", L"", false, true},
    {L"--external-poll-jitter", L"external_poll_jitter", L"", L"", L"", false, true},
    {L"--telemetry-db", L"telemetry_db", L"", L"", L"", false, true},
    {L"--no-telemetry", L"no_telemetry", L"", L"", L"", false, false},
    {L"--source-grading", L"source_grading", L"", L"", L"", false, false},
    {L"--no-source-grading", L"source_grading", L"", L"true", L"", false, false},
    {L"--source-grading-interval", L"source_grading_interval", L"", L"", L"", false, true},
    {L"--source-grading-window-hours", L"source_grading_window_hours", L"", L"", L"", false, true},
    {L"--source-grade-max-age-hours", L"source_grade_max_age_hours", L"", L"", L"", false, true},
};

inline constexpr CommandOptionSpec kOptions_train[] = {
    {L"--input", L"input", L"", L"data/historical_btcusdc.json", L"", false, true},
    {L"--output", L"output", L"", L"data/model.json", L"", false, true},
    {L"--source", L"source", L"auto, file, db", L"auto", L"", false, true},
    {L"--db", L"db", L"", L"data/market_data.sqlite", L"", false, true},
    {L"--interval", L"interval", L"", L"", L"", false, true},
    {L"--market", L"market", L"spot, futures", L"", L"", false, true},
    {L"--min-rows", L"min_rows", L"", L"120", L"", false, true},
    {L"--download-missing", L"download_missing", L"", L"false", L"", false, false},
    {L"--preset", L"preset", L"balanced, custom, quick, thorough", L"custom", L"", false, true},
    {L"--epochs", L"epochs", L"", L"250", L"", false, true},
    {L"--learning-rate", L"learning_rate", L"", L"0.05", L"", false, true},
    {L"--l2-penalty", L"l2_penalty", L"", L"0.0001", L"", false, true},
    {L"--seed", L"seed", L"", L"7", L"", false, true},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"training backend override; default uses saved runtime compute_backend", false, true},
    {L"--batch-size", L"batch_size", L"", L"8192", L"mini-batch size for GPU training", false, true},
    {L"--walk-forward", L"walk_forward", L"", L"false", L"run walk-forward validation before final training", false, false},
    {L"--walk-forward-train", L"walk_forward_train", L"", L"300", L"", false, true},
    {L"--walk-forward-test", L"walk_forward_test", L"", L"60", L"", false, true},
    {L"--walk-forward-step", L"walk_forward_step", L"", L"30", L"", false, true},
    {L"--calibrate-threshold", L"calibrate_threshold", L"", L"false", L"optimize a probability threshold on validation split", false, false},
};

inline constexpr CommandOptionSpec kOptions_train_suite[] = {
    {L"--input", L"input", L"", L"data/historical_btcusdc.json", L"", false, true},
    {L"--output-dir", L"output_dir", L"", L"data", L"", false, true},
    {L"--starting-cash", L"starting_cash", L"", L"1000.0", L"", false, true},
    {L"--objective", L"objective", L"", L"", L"restrict suite to named objective(s); repeat to list multiple.", false, true},
    {L"--max-workers", L"max_workers", L"", L"", L"parallel candidate workers; defaults to available CPU cores", false, true},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"training backend override; GPU backends run candidates sequentially to protect VRAM", false, true},
    {L"--batch-size", L"batch_size", L"", L"8192", L"mini-batch size for GPU training", false, true},
    {L"--max-candidates", L"max_candidates", L"", L"", L"smoke/research cap per objective; default evaluates the full grid", false, true},
};

inline constexpr CommandOptionSpec kOptions_tune[] = {
    {L"--input", L"input", L"", L"data/historical_btcusdc.json", L"", false, true},
    {L"--save-best", L"save_best", L"", L"false", L"", false, false},
    {L"--min-risk", L"min_risk", L"", L"0.002", L"", false, true},
    {L"--max-risk", L"max_risk", L"", L"0.02", L"", false, true},
    {L"--steps", L"steps", L"", L"5", L"", false, true},
    {L"--min-leverage", L"min_leverage", L"", L"1.0", L"", false, true},
    {L"--max-leverage", L"max_leverage", L"", L"20.0", L"", false, true},
    {L"--min-threshold", L"min_threshold", L"", L"0.52", L"", false, true},
    {L"--max-threshold", L"max_threshold", L"", L"0.88", L"", false, true},
    {L"--min-take", L"min_take", L"", L"0.01", L"", false, true},
    {L"--max-take", L"max_take", L"", L"0.06", L"", false, true},
    {L"--min-stop", L"min_stop", L"", L"0.008", L"", false, true},
    {L"--max-stop", L"max_stop", L"", L"0.04", L"", false, true},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"", false, true},
    {L"--batch-size", L"batch_size", L"", L"8192", L"mini-batch size for accelerated tuning", false, true},
    {L"--lookback-days", L"lookback_days", L"", L"", L"use only the most recent N days of candles for tuning", false, true},
    {L"--from-date", L"from_date", L"", L"", L"inclusive start date for tuning window (YYYY-MM-DD)", false, true},
    {L"--to-date", L"to_date", L"", L"", L"inclusive end date for tuning window (YYYY-MM-DD)", false, true},
};

inline constexpr CommandOptionSpec kOptions_universe[] = {
    {L"--symbols", L"symbols", L"", L"", L"comma-separated symbols; default uses runtime.symbols", false, true},
    {L"--json", L"json", L"", L"false", L"", false, false},
};

inline constexpr CommandSpec kCommands[] = {
    {L"ai", L"usage: simple-ai-trading ai [-h] [--enable] [--disable] [--provider PROVIDER]                             [--model MODEL] [--require-gpu] [--no-require-gpu]                             [--min-free-vram-gb MIN_FREE_VRAM_GB]                             [--min-free-ram-gb MIN_FREE_RAM_GB]                             [--min-model-parameters-b MIN_MODEL_PARAMETERS_B]                             [--allow-paper-fallback] [--no-paper-fallback]                             [--json]", kOptions_ai, 12},
    {L"ai-review", L"usage: simple-ai-trading ai-review [-h] [--report REPORT] [--output OUTPUT]                                    [--model MODEL] [--url URL]                                    [--timeout TIMEOUT] [--json]", kOptions_ai_review, 6},
    {L"api-budget", L"usage: simple-ai-trading api-budget [-h] [--db DB] [--market {spot,futures}]                                     [--refresh] [--cached-only]                                     [--max-age-seconds MAX_AGE_SECONDS]                                     [--compact] [--json]", kOptions_api_budget, 7},
    {L"archive-sync", L"usage: simple-ai-trading archive-sync [-h] [--db DB] [--symbol SYMBOL]                                       [--symbols SYMBOLS]                                       [--top-symbols TOP_SYMBOLS]                                       [--quote-asset QUOTE_ASSET]                                       [--max-scan MAX_SCAN]                                       [--min-history-months MIN_HISTORY_MONTHS]                                       [--interval INTERVAL]                                       [--market {spot,futures}]                                       [--cadence {monthly,daily}]                                       [--data-type {klines,aggTrades}]                                       [--max-files MAX_FILES]                                       [--start-period START_PERIOD]                                       [--end-period END_PERIOD] [--plan-only]                                       [--max-planned-gb MAX_PLANNED_GB]                                       [--timeout TIMEOUT] [--force]                                       [--no-verify-checksum]                                       [--require-checksum] [--json]", kOptions_archive_sync, 21},
    {L"audit", L"usage: simple-ai-trading audit [-h] [--input INPUT] [--model MODEL]", kOptions_audit, 2},
    {L"autonomous", L"usage: simple-ai-trading autonomous [-h] [--objective OBJECTIVE]                                     [--model MODEL]                                     [--poll-seconds POLL_SECONDS]                                     [--iterations ITERATIONS]                                     [--heartbeat-every HEARTBEAT_EVERY]                                     [--starting-cash STARTING_CASH] [--paper]                                     [--live]                                     {start,pause,resume,stop,status}", kOptions_autonomous, 9},
    {L"backtest", L"usage: simple-ai-trading backtest [-h] [--input INPUT] [--model MODEL]                                   [--start-cash START_CASH]                                   [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                                   [--score-batch-size SCORE_BATCH_SIZE]                                   [--execution-db EXECUTION_DB]", kOptions_backtest, 6},
    {L"backtest-chart", L"usage: simple-ai-trading backtest-chart [-h] [--input INPUT] [--model MODEL]                                         [--output OUTPUT]                                         [--start-cash START_CASH]                                         [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                                         [--score-batch-size SCORE_BATCH_SIZE]                                         [--execution-db EXECUTION_DB]", kOptions_backtest_chart, 7},
    {L"backtest-panel", L"usage: simple-ai-trading backtest-panel [-h] --interval INTERVAL                                         [--market MARKET]                                         [--from-date FROM_DATE]                                         [--to-date TO_DATE] [--input INPUT]                                         [--model MODEL]                                         [--objective OBJECTIVE] [--tag TAG]                                         [--notes NOTES]                                         [--starting-cash STARTING_CASH]                                         [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                                         [--execution-db EXECUTION_DB]", kOptions_backtest_panel, 12},
    {L"close", L"usage: simple-ai-trading close [-h] position_id", kOptions_close, 1},
    {L"compute", L"usage: simple-ai-trading compute [-h]                                  [--backend {cpu,cuda,rocm,directml,mps,auto}]", kOptions_compute, 1},
    {L"configure", L"usage: simple-ai-trading configure [-h]", nullptr, 0},
    {L"connect", L"usage: simple-ai-trading connect [-h]", nullptr, 0},
    {L"coordinator", L"usage: simple-ai-trading coordinator [-h] [--model MODEL]                                      [--positions-root POSITIONS_ROOT]                                      [--json]", kOptions_coordinator, 3},
    {L"data-health", L"usage: simple-ai-trading data-health [-h] [--db DB] [--symbol SYMBOL]                                      [--symbols SYMBOLS] [--interval INTERVAL]                                      [--market {spot,futures}]                                      [--min-rows MIN_ROWS]                                      [--min-coverage-ratio MIN_COVERAGE_RATIO]                                      [--max-gap-count MAX_GAP_COUNT]                                      [--require-verified-checksum] [--json]", kOptions_data_health, 10},
    {L"data-sync", L"usage: simple-ai-trading data-sync [-h] [--db DB] [--symbol SYMBOL]                                    [--interval INTERVAL]                                    [--market {spot,futures}] [--rows ROWS]                                    [--full-history] [--batch-size BATCH_SIZE]                                    [--include-futures-metrics]                                    [--no-include-futures-metrics] [--loop]                                    [--iterations ITERATIONS] [--sleep SLEEP]                                    [--background] [--pid-file PID_FILE]                                    [--log-file LOG_FILE] [--json]", kOptions_data_sync, 16},
    {L"doctor", L"usage: simple-ai-trading doctor [-h] [--input INPUT] [--model MODEL]                                 [--online]", kOptions_doctor, 3},
    {L"evaluate", L"usage: simple-ai-trading evaluate [-h] [--input INPUT] [--model MODEL]                                   [--threshold THRESHOLD]                                   [--calibrate-threshold]", kOptions_evaluate, 4},
    {L"fetch", L"usage: simple-ai-trading fetch [-h] [--symbol SYMBOL] [--interval INTERVAL]                                [--limit LIMIT] [--batch-size BATCH_SIZE]                                [--output OUTPUT]", kOptions_fetch, 5},
    {L"live", L"usage: simple-ai-trading live [-h] [--model MODEL] [--steps STEPS]                               [--sleep SLEEP] [--leverage LEVERAGE]                               [--retrain-interval RETRAIN_INTERVAL]                               [--retrain-window RETRAIN_WINDOW]                               [--retrain-min-rows RETRAIN_MIN_ROWS]                               [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                               [--batch-size BATCH_SIZE] [--paper] [--live]                               [--external-signals] [--no-external-signals]", kOptions_live, 13},
    {L"menu", L"usage: simple-ai-trading menu [-h]", nullptr, 0},
    {L"model-blueprint", L"usage: simple-ai-trading model-blueprint [-h]                                          [--risk-level {conservative,regular,aggressive,default,balanced,risky}]                                          [--implemented-only] [--json]", kOptions_model_blueprint, 3},
    {L"model-lab", L"usage: simple-ai-trading model-lab [-h] [--output-dir OUTPUT_DIR]                                    [--starting-cash STARTING_CASH]                                    [--objective OBJECTIVE]                                    [--max-symbols MAX_SYMBOLS]                                    [--max-scan MAX_SCAN] [--limit LIMIT]                                    [--quote-asset QUOTE_ASSET]                                    [--interval INTERVAL] [--full-history]                                    [--market-db MARKET_DB] [--require-db-data]                                    [--market {spot,futures}]                                    [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                                    [--batch-size BATCH_SIZE]                                    [--score-batch-size SCORE_BATCH_SIZE]                                    [--max-candidates MAX_CANDIDATES]                                    [--learning-feedback LEARNING_FEEDBACK]", kOptions_model_lab, 17},
    {L"objectives", L"usage: simple-ai-trading objectives [-h]", nullptr, 0},
    {L"positions", L"usage: simple-ai-trading positions [-h] [--stats] [--learning]", kOptions_positions, 2},
    {L"prepare", L"usage: simple-ai-trading prepare [-h] [--historical HISTORICAL]                                  [--model MODEL] [--limit LIMIT]                                  [--batch-size BATCH_SIZE]                                  [--preset {balanced,custom,quick,thorough}]                                  [--epochs EPOCHS]                                  [--learning-rate LEARNING_RATE]                                  [--l2-penalty L2_PENALTY] [--seed SEED]                                  [--start-cash START_CASH] [--walk-forward]                                  [--no-walk-forward]                                  [--walk-forward-train WALK_FORWARD_TRAIN]                                  [--walk-forward-test WALK_FORWARD_TEST]                                  [--walk-forward-step WALK_FORWARD_STEP]                                  [--calibrate-threshold]                                  [--no-calibrate-threshold] [--online-doctor]", kOptions_prepare, 18},
    {L"reconcile", L"usage: simple-ai-trading reconcile [-h] [--json] [--output OUTPUT]                                    [--quantity-tolerance QUANTITY_TOLERANCE]", kOptions_reconcile, 3},
    {L"report", L"usage: simple-ai-trading report [-h] [--account] [--doctor] [--no-doctor]                                 [--online] [--input INPUT] [--model MODEL]", kOptions_report, 6},
    {L"risk", L"usage: simple-ai-trading risk [-h] [--model MODEL] [--paper] [--live]                               [--leverage LEVERAGE] [--json]", kOptions_risk, 5},
    {L"shell", L"usage: simple-ai-trading shell [-h]", nullptr, 0},
    {L"signals", L"usage: simple-ai-trading signals [-h] [--model MODEL] [--cache CACHE]                                  [--ttl TTL] [--timeout TIMEOUT]                                  [--max-adjustment MAX_ADJUSTMENT]                                  [--min-providers MIN_PROVIDERS]                                  [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                                  [--short-reaction-refresh SHORT_REACTION_REFRESH]                                  [--news-provider-limit NEWS_PROVIDER_LIMIT]                                  [--news-items-per-provider NEWS_ITEMS_PER_PROVIDER]                                  [--provider-parallelism PROVIDER_PARALLELISM]                                  [--provider-jitter PROVIDER_JITTER]                                  [--ollama-news] [--no-ollama-news]                                  [--ollama-model OLLAMA_MODEL]                                  [--ollama-url OLLAMA_URL]                                  [--ollama-timeout OLLAMA_TIMEOUT]                                  [--telemetry-db TELEMETRY_DB]                                  [--source-grade-max-age-hours SOURCE_GRADE_MAX_AGE_HOURS]                                  [--no-telemetry] [--loop]                                  [--iterations ITERATIONS] [--sleep SLEEP]                                  [--jitter JITTER] [--refresh] [--json]", kOptions_signals, 26},
    {L"signals-benchmark", L"usage: simple-ai-trading signals-benchmark [-h]                                            [--provider-limit PROVIDER_LIMIT]                                            [--parallelism PARALLELISM]                                            [--iterations ITERATIONS]                                            [--timeout TIMEOUT]                                            [--provider-jitter PROVIDER_JITTER]                                            [--ollama-news] [--no-ollama-news]                                            [--ollama-model OLLAMA_MODEL]                                            [--ollama-url OLLAMA_URL]                                            [--ollama-timeout OLLAMA_TIMEOUT]                                            [--cache CACHE] [--no-telemetry]                                            [--json]", kOptions_signals_benchmark, 13},
    {L"source-grades", L"usage: simple-ai-trading source-grades [-h] [--db DB]                                        [--window-hours WINDOW_HOURS]                                        [--ollama] [--no-ollama]                                        [--ollama-model OLLAMA_MODEL]                                        [--ollama-url OLLAMA_URL]                                        [--ollama-timeout OLLAMA_TIMEOUT]                                        [--json]", kOptions_source_grades, 8},
    {L"spot-roundtrip", L"usage: simple-ai-trading spot-roundtrip [-h] [--quantity QUANTITY]                                         [--mode {auto,buy-sell,sell-buy}]                                         [--yes]", kOptions_spot_roundtrip, 3},
    {L"status", L"usage: simple-ai-trading status [-h]", nullptr, 0},
    {L"strategy", L"usage: simple-ai-trading strategy [-h]                                   [--profile {active,aggressive,balanced,conservative,custom,regular}]                                   [--risk-level {conservative,regular,aggressive}]                                   [--reinvest-profits] [--no-reinvest-profits]                                   [--leverage LEVERAGE] [--risk RISK]                                   [--max-position MAX_POSITION] [--stop STOP]                                   [--take TAKE] [--cooldown COOLDOWN]                                   [--max-open MAX_OPEN]                                   [--min-diversified-assets MIN_DIVERSIFIED_ASSETS]                                   [--max-asset-allocation MAX_ASSET_ALLOCATION]                                   [--max-portfolio-risk MAX_PORTFOLIO_RISK]                                   [--min-quote-volume-usdc MIN_QUOTE_VOLUME_USDC]                                   [--min-trade-count-24h MIN_TRADE_COUNT_24H]                                   [--max-spread-bps MAX_SPREAD_BPS]                                   [--min-liquidity-score MIN_LIQUIDITY_SCORE]                                   [--unpredictability-cooldown UNPREDICTABILITY_COOLDOWN]                                   [--max-regime-unpredictability MAX_REGIME_UNPREDICTABILITY]                                   [--max-prediction-entropy MAX_PREDICTION_ENTROPY]                                   [--min-model-confidence MIN_MODEL_CONFIDENCE]                                   [--max-trades-per-day MAX_TRADES_PER_DAY]                                   [--signal-threshold SIGNAL_THRESHOLD]                                   [--max-drawdown MAX_DRAWDOWN]                                   [--max-daily-loss MAX_DAILY_LOSS]                                   [--max-session-loss MAX_SESSION_LOSS]                                   [--max-consecutive-losses MAX_CONSECUTIVE_LOSSES]                                   [--max-network-errors MAX_NETWORK_ERRORS]                                   [--recovery-cooldown-seconds RECOVERY_COOLDOWN_SECONDS]                                   [--taker-fee-bps TAKER_FEE_BPS]                                   [--slippage-bps SLIPPAGE_BPS]                                   [--label-threshold LABEL_THRESHOLD]                                   [--model-lookback MODEL_LOOKBACK]                                   [--training-epochs TRAINING_EPOCHS]                                   [--confidence-beta CONFIDENCE_BETA]                                   [--feature-window-short FEATURE_WINDOW_SHORT]                                   [--feature-window-long FEATURE_WINDOW_LONG]                                   [--set-features SET_FEATURES]                                   [--enable-feature ENABLE_FEATURE]                                   [--disable-feature DISABLE_FEATURE]                                   [--external-signals] [--no-external-signals]                                   [--external-signal-max-adjustment EXTERNAL_SIGNAL_MAX_ADJUSTMENT]                                   [--external-signal-min-providers EXTERNAL_SIGNAL_MIN_PROVIDERS]                                   [--external-signal-ttl EXTERNAL_SIGNAL_TTL]                                   [--external-signal-timeout EXTERNAL_SIGNAL_TIMEOUT]                                   [--external-news-ai] [--no-external-news-ai]                                   [--external-news-ai-model EXTERNAL_NEWS_AI_MODEL]                                   [--external-news-ai-url EXTERNAL_NEWS_AI_URL]                                   [--external-news-ai-timeout EXTERNAL_NEWS_AI_TIMEOUT]                                   [--external-news-provider-limit EXTERNAL_NEWS_PROVIDER_LIMIT]                                   [--external-provider-parallelism EXTERNAL_PROVIDER_PARALLELISM]                                   [--external-provider-jitter EXTERNAL_PROVIDER_JITTER]                                   [--external-poll-jitter EXTERNAL_POLL_JITTER]                                   [--telemetry-db TELEMETRY_DB]                                   [--no-telemetry] [--source-grading]                                   [--no-source-grading]                                   [--source-grading-interval SOURCE_GRADING_INTERVAL]                                   [--source-grading-window-hours SOURCE_GRADING_WINDOW_HOURS]                                   [--source-grade-max-age-hours SOURCE_GRADE_MAX_AGE_HOURS]", kOptions_strategy, 63},
    {L"train", L"usage: simple-ai-trading train [-h] [--input INPUT] [--output OUTPUT]                                [--source {auto,file,db}] [--db DB]                                [--interval INTERVAL] [--market {spot,futures}]                                [--min-rows MIN_ROWS] [--download-missing]                                [--preset {balanced,custom,quick,thorough}]                                [--epochs EPOCHS]                                [--learning-rate LEARNING_RATE]                                [--l2-penalty L2_PENALTY] [--seed SEED]                                [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                                [--batch-size BATCH_SIZE] [--walk-forward]                                [--walk-forward-train WALK_FORWARD_TRAIN]                                [--walk-forward-test WALK_FORWARD_TEST]                                [--walk-forward-step WALK_FORWARD_STEP]                                [--calibrate-threshold]", kOptions_train, 20},
    {L"train-suite", L"usage: simple-ai-trading train-suite [-h] [--input INPUT]                                      [--output-dir OUTPUT_DIR]                                      [--starting-cash STARTING_CASH]                                      [--objective OBJECTIVE]                                      [--max-workers MAX_WORKERS]                                      [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                                      [--batch-size BATCH_SIZE]                                      [--max-candidates MAX_CANDIDATES]", kOptions_train_suite, 8},
    {L"tune", L"usage: simple-ai-trading tune [-h] [--input INPUT] [--save-best]                               [--min-risk MIN_RISK] [--max-risk MAX_RISK]                               [--steps STEPS] [--min-leverage MIN_LEVERAGE]                               [--max-leverage MAX_LEVERAGE]                               [--min-threshold MIN_THRESHOLD]                               [--max-threshold MAX_THRESHOLD]                               [--min-take MIN_TAKE] [--max-take MAX_TAKE]                               [--min-stop MIN_STOP] [--max-stop MAX_STOP]                               [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                               [--batch-size BATCH_SIZE]                               [--lookback-days LOOKBACK_DAYS]                               [--from-date FROM_DATE] [--to-date TO_DATE]", kOptions_tune, 18},
    {L"universe", L"usage: simple-ai-trading universe [-h] [--symbols SYMBOLS] [--json]", kOptions_universe, 2},
};
inline constexpr int kCommandCount = static_cast<int>(sizeof(kCommands) / sizeof(kCommands[0]));

} // namespace simple_ai_trading::native_contract
