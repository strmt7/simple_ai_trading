#pragma once

namespace simple_ai_trading::native_contract {

struct CommandOptionSpec {
    const wchar_t* flags;
    const wchar_t* dest;
    const wchar_t* choices;
    const wchar_t* default_value;
    const wchar_t* help;
    const wchar_t* value_arity;
    bool required;
    bool takes_value;
    bool repeatable;
};

struct CommandSpec {
    const wchar_t* name;
    const wchar_t* help;
    const CommandOptionSpec* options;
    int option_count;
};

inline constexpr CommandOptionSpec kOptions_ai[] = {
    {L"--enable", L"enable", L"", L"", L"enable AI decision features", L"0", false, false, false},
    {L"--disable", L"disable", L"", L"", L"disable AI decision features", L"0", false, false, false},
    {L"--provider", L"provider", L"", L"", L"AI provider: auto, local-gpu, ollama, openai-compatible, etc.", L"1", false, true, false},
    {L"--model", L"model", L"", L"", L"AI model identifier or 'auto'", L"1", false, true, false},
    {L"--require-gpu", L"require_gpu", L"", L"", L"", L"0", false, false, false},
    {L"--no-require-gpu", L"no_require_gpu", L"", L"", L"", L"0", false, false, false},
    {L"--min-free-vram-gb", L"min_free_vram_gb", L"", L"", L"", L"1", false, true, false},
    {L"--min-free-ram-gb", L"min_free_ram_gb", L"", L"", L"", L"1", false, true, false},
    {L"--min-model-parameters-b", L"min_model_parameters_b", L"", L"", L"", L"1", false, true, false},
    {L"--allow-paper-fallback", L"allow_paper_fallback", L"", L"", L"", L"0", false, false, false},
    {L"--no-paper-fallback", L"no_paper_fallback", L"", L"", L"", L"0", false, false, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_ai_benchmark[] = {
    {L"--models", L"models", L"", L"", L"comma-separated Ollama model names; defaults to installed curated candidates", L"1", false, true, false},
    {L"--url", L"url", L"", L"http://127.0.0.1:11434", L"", L"1", false, true, false},
    {L"--timeout", L"timeout", L"", L"20.0", L"", L"1", false, true, false},
    {L"--minimum-score", L"minimum_score", L"", L"0.78", L"", L"1", false, true, false},
    {L"--output", L"output", L"", L"data/ai_model_benchmark.json", L"", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_ai_forecast_benchmark[] = {
    {L"--database", L"database", L"", L"data/market_data.sqlite", L"", L"1", false, true, false},
    {L"--model-size", L"model_size", L"small, base", L"base", L"", L"1", false, true, false},
    {L"--backend", L"backend", L"cpu, cuda, rocm, directml, mps, auto", L"directml", L"", L"1", false, true, false},
    {L"--source-cache", L"source_cache", L"", L"", L"", L"1", false, true, false},
    {L"--bootstrap-source", L"bootstrap_source", L"", L"false", L"", L"0", false, false, false},
    {L"--repair-source", L"repair_source", L"", L"false", L"", L"0", false, false, false},
    {L"--allow-cpu", L"allow_cpu", L"", L"false", L"", L"0", false, false, false},
    {L"--start", L"start", L"", L"2024-07-01T00:00:00Z", L"", L"1", false, true, false},
    {L"--end-exclusive", L"end_exclusive", L"", L"2026-01-01T00:00:00Z", L"", L"1", false, true, false},
    {L"--samples-per-symbol", L"samples_per_symbol", L"", L"128", L"", L"1", false, true, false},
    {L"--lookback-bars", L"lookback_bars", L"", L"480", L"", L"1", false, true, false},
    {L"--prediction-bars", L"prediction_bars", L"", L"12", L"", L"1", false, true, false},
    {L"--batch-size", L"batch_size", L"", L"3", L"", L"1", false, true, false},
    {L"--inference-samples", L"inference_samples", L"", L"10", L"", L"1", false, true, false},
    {L"--temperature", L"temperature", L"", L"0.6", L"", L"1", false, true, false},
    {L"--top-k", L"top_k", L"", L"0", L"", L"1", false, true, false},
    {L"--top-p", L"top_p", L"", L"0.9", L"", L"1", false, true, false},
    {L"--include-volume", L"include_volume", L"", L"false", L"include volume/amount despite the upstream crypto evaluation using OHLC only", L"0", false, false, false},
    {L"--seed", L"seed", L"", L"17", L"", L"1", false, true, false},
    {L"--bootstrap-samples", L"bootstrap_samples", L"", L"2000", L"", L"1", false, true, false},
    {L"--worker-timeout", L"worker_timeout", L"", L"60.0", L"", L"1", false, true, false},
    {L"--max-worker-restarts", L"max_worker_restarts", L"", L"5", L"", L"1", false, true, false},
    {L"--worker-rotation-batches", L"worker_rotation_batches", L"", L"20", L"", L"1", false, true, false},
    {L"--observations", L"observations", L"", L"data/foundation_ai/kronos_observations.csv", L"", L"1", false, true, false},
    {L"--output", L"output", L"", L"data/foundation_ai/kronos_benchmark.json", L"", L"1", false, true, false},
    {L"--chart", L"chart", L"", L"data/foundation_ai/kronos_benchmark.svg", L"", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_ai_review[] = {
    {L"--report", L"report", L"", L"data/model_lab/model_lab_report.json", L"", L"1", false, true, false},
    {L"--output", L"output", L"", L"", L"", L"1", false, true, false},
    {L"--model", L"model", L"", L"", L"", L"1", false, true, false},
    {L"--url", L"url", L"", L"http://127.0.0.1:11434", L"", L"1", false, true, false},
    {L"--timeout", L"timeout", L"", L"20.0", L"", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_api_budget[] = {
    {L"--db", L"db", L"", L"data/market_data.sqlite", L"", L"1", false, true, false},
    {L"--market", L"market", L"spot, futures", L"", L"", L"1", false, true, false},
    {L"--refresh", L"refresh", L"", L"false", L"query Binance exchangeInfo once and cache the latest headers", L"0", false, false, false},
    {L"--cached-only", L"cached_only", L"", L"false", L"do not refresh even when the cached sample is stale", L"0", false, false, false},
    {L"--max-age-seconds", L"max_age_seconds", L"", L"90", L"automatic refresh threshold for cached status", L"1", false, true, false},
    {L"--compact", L"compact", L"", L"false", L"print one status-bar friendly line", L"0", false, false, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_archive_sync[] = {
    {L"--db", L"db", L"", L"data/market_data.sqlite", L"", L"1", false, true, false},
    {L"--symbol", L"symbol", L"", L"", L"", L"1", false, true, false},
    {L"--symbols", L"symbols", L"", L"", L"comma-separated symbols; overrides --symbol", L"1", false, true, false},
    {L"--top-symbols", L"top_symbols", L"", L"0", L"auto-rank this many high-liquidity symbols", L"1", false, true, false},
    {L"--quote-asset", L"quote_asset", L"", L"", L"quote asset used with --top-symbols", L"1", false, true, false},
    {L"--max-scan", L"max_scan", L"", L"250", L"maximum universe candidates scanned with --top-symbols", L"1", false, true, false},
    {L"--min-history-months", L"min_history_months", L"", L"0", L"with --top-symbols and monthly cadence, require this many monthly archive files before selecting a symbol", L"1", false, true, false},
    {L"--interval", L"interval", L"", L"", L"", L"1", false, true, false},
    {L"--market", L"market", L"spot, futures", L"spot", L"", L"1", false, true, false},
    {L"--cadence", L"cadence", L"monthly, daily", L"monthly", L"", L"1", false, true, false},
    {L"--data-type", L"data_type", L"klines, aggTrades", L"", L"official archive data type; futures 1s defaults to aggTrades and aggregates real trades to 1s candles", L"1", false, true, false},
    {L"--max-files", L"max_files", L"", L"", L"optional safety cap for smoke runs", L"1", false, true, false},
    {L"--start-period", L"start_period", L"", L"", L"inclusive archive period start, YYYY-MM or YYYY-MM-DD", L"1", false, true, false},
    {L"--end-period", L"end_period", L"", L"", L"inclusive archive period end, YYYY-MM or YYYY-MM-DD", L"1", false, true, false},
    {L"--plan-only", L"plan_only", L"", L"false", L"list the bounded archive plan without downloading files", L"0", false, false, false},
    {L"--max-planned-gb", L"max_planned_gb", L"", L"50.0", L"block non-plan archive downloads above this planned S3 ZIP size; use 0 to disable", L"1", false, true, false},
    {L"--timeout", L"timeout", L"", L"120", L"", L"1", false, true, false},
    {L"--force", L"force", L"", L"false", L"", L"0", false, false, false},
    {L"--aggregate-only", L"aggregate_only", L"", L"false", L"for aggTrades, persist derived 1s candles without duplicating raw trades", L"0", false, false, false},
    {L"--no-verify-checksum", L"no_verify_checksum", L"", L"false", L"skip Binance .CHECKSUM sidecar verification", L"0", false, false, false},
    {L"--require-checksum", L"require_checksum", L"", L"false", L"fail archive files without a readable .CHECKSUM sidecar", L"0", false, false, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_audit[] = {
    {L"--input", L"input", L"", L"data/historical_market.json", L"", L"1", false, true, false},
    {L"--model", L"model", L"", L"data/model.json", L"", L"1", false, true, false},
};

inline constexpr CommandOptionSpec kOptions_autonomous[] = {
    {L"--objective", L"objective", L"", L"conservative", L"", L"1", false, true, false},
    {L"--model", L"model", L"", L"data/model.json", L"model artifact used for autonomous decisions", L"1", false, true, false},
    {L"--poll-seconds", L"poll_seconds", L"", L"30.0", L"seconds between autonomous iterations", L"1", false, true, false},
    {L"--iterations", L"iterations", L"", L"", L"stop after N iterations; default runs until stopped", L"1", false, true, false},
    {L"--heartbeat-every", L"heartbeat_every", L"", L"1", L"write heartbeat every N iterations", L"1", false, true, false},
    {L"--starting-cash", L"starting_cash", L"", L"1000.0", L"reference cash for local autonomous risk stats", L"1", false, true, false},
    {L"--paper", L"paper", L"", L"false", L"force autonomous paper mode", L"0", false, false, false},
    {L"--live", L"live", L"", L"false", L"force authenticated non-mainnet autonomous mode", L"0", false, false, false},
    {L"action", L"action", L"start, pause, resume, stop, status", L"", L"autonomous action to perform", L"1", true, true, false},
};

inline constexpr CommandOptionSpec kOptions_backtest[] = {
    {L"--input", L"input", L"", L"data/historical_market.json", L"", L"1", false, true, false},
    {L"--model", L"model", L"", L"data/model.json", L"", L"1", false, true, false},
    {L"--start-cash", L"start_cash", L"", L"1000.0", L"", L"1", false, true, false},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"model-scoring backend override; default uses saved runtime compute_backend", L"1", false, true, false},
    {L"--score-batch-size", L"score_batch_size", L"", L"8192", L"batch size for GPU-assisted probability scoring", L"1", false, true, false},
    {L"--execution-db", L"execution_db", L"", L"", L"optional SQLite market-data DB; latest typed top-of-book row becomes symbol-specific fill stress", L"1", false, true, false},
};

inline constexpr CommandOptionSpec kOptions_backtest_chart[] = {
    {L"--input", L"input", L"", L"data/historical_market.json", L"", L"1", false, true, false},
    {L"--model", L"model", L"", L"data/model.json", L"", L"1", false, true, false},
    {L"--output", L"output", L"", L"data/backtest_performance.svg", L"", L"1", false, true, false},
    {L"--start-cash", L"start_cash", L"", L"1000.0", L"", L"1", false, true, false},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"", L"1", false, true, false},
    {L"--score-batch-size", L"score_batch_size", L"", L"8192", L"", L"1", false, true, false},
    {L"--execution-db", L"execution_db", L"", L"", L"optional SQLite market-data DB for symbol-specific top-of-book fill stress", L"1", false, true, false},
};

inline constexpr CommandOptionSpec kOptions_backtest_panel[] = {
    {L"--interval", L"interval", L"", L"", L"", L"1", true, true, false},
    {L"--market", L"market", L"", L"", L"override runtime market type", L"1", false, true, false},
    {L"--from-date", L"from_date", L"", L"", L"", L"1", false, true, false},
    {L"--to-date", L"to_date", L"", L"", L"", L"1", false, true, false},
    {L"--input", L"input", L"", L"data/historical_market.json", L"", L"1", false, true, false},
    {L"--model", L"model", L"", L"", L"", L"1", false, true, false},
    {L"--objective", L"objective", L"", L"", L"", L"1", false, true, false},
    {L"--tag", L"tag", L"", L"", L"", L"1", false, true, false},
    {L"--notes", L"notes", L"", L"", L"", L"1", false, true, false},
    {L"--starting-cash", L"starting_cash", L"", L"1000.0", L"", L"1", false, true, false},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"feature/scoring backend override; default uses saved runtime compute_backend", L"1", false, true, false},
    {L"--execution-db", L"execution_db", L"", L"", L"optional SQLite market-data DB for symbol-specific top-of-book fill stress", L"1", false, true, false},
};

inline constexpr CommandOptionSpec kOptions_close[] = {
    {L"position_id", L"position_id", L"", L"", L"position id or 'all'", L"1", true, true, false},
};

inline constexpr CommandOptionSpec kOptions_compute[] = {
    {L"--backend", L"backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"", L"1", false, true, false},
};

inline constexpr CommandOptionSpec kOptions_coordinator[] = {
    {L"--model", L"model", L"", L"data/model.json", L"", L"1", false, true, false},
    {L"--positions-root", L"positions_root", L"", L"data/autonomous", L"", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_data_health[] = {
    {L"--db", L"db", L"", L"data/market_data.sqlite", L"", L"1", false, true, false},
    {L"--symbol", L"symbol", L"", L"", L"", L"1", false, true, false},
    {L"--symbols", L"symbols", L"", L"", L"comma-separated symbols; defaults to stored series", L"1", false, true, false},
    {L"--interval", L"interval", L"", L"", L"", L"1", false, true, false},
    {L"--market", L"market", L"spot, futures", L"", L"", L"1", false, true, false},
    {L"--min-rows", L"min_rows", L"", L"0", L"", L"1", false, true, false},
    {L"--min-coverage-ratio", L"min_coverage_ratio", L"", L"0.995", L"", L"1", false, true, false},
    {L"--max-gap-count", L"max_gap_count", L"", L"0", L"", L"1", false, true, false},
    {L"--require-verified-checksum", L"require_verified_checksum", L"", L"false", L"", L"0", false, false, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_data_sync[] = {
    {L"--db", L"db", L"", L"data/market_data.sqlite", L"", L"1", false, true, false},
    {L"--symbol", L"symbol", L"", L"", L"", L"1", false, true, false},
    {L"--interval", L"interval", L"", L"", L"", L"1", false, true, false},
    {L"--market", L"market", L"spot, futures", L"", L"", L"1", false, true, false},
    {L"--rows", L"rows", L"", L"500", L"", L"1", false, true, false},
    {L"--full-history", L"full_history", L"", L"false", L"page historical klines backward until the exchange has no older closed candles", L"0", false, false, false},
    {L"--batch-size", L"batch_size", L"", L"1000", L"", L"1", false, true, false},
    {L"--include-futures-metrics", L"include_futures_metrics", L"", L"true", L"", L"0", false, false, false},
    {L"--no-include-futures-metrics", L"include_futures_metrics", L"", L"true", L"", L"0", false, false, false},
    {L"--loop", L"loop", L"", L"false", L"keep syncing in the foreground", L"0", false, false, false},
    {L"--iterations", L"iterations", L"", L"1", L"foreground loop iterations; 0 means unlimited", L"1", false, true, false},
    {L"--sleep", L"sleep", L"", L"300", L"seconds between loop iterations", L"1", false, true, false},
    {L"--background", L"background", L"", L"false", L"start a detached downloader process", L"0", false, false, false},
    {L"--pid-file", L"pid_file", L"", L"data/market_data_sync.pid", L"", L"1", false, true, false},
    {L"--log-file", L"log_file", L"", L"data/market_data_sync.log", L"", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_doctor[] = {
    {L"--input", L"input", L"", L"data/historical_market.json", L"", L"1", false, true, false},
    {L"--model", L"model", L"", L"data/model.json", L"", L"1", false, true, false},
    {L"--online", L"online", L"", L"false", L"also check exchange connectivity", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_evaluate[] = {
    {L"--input", L"input", L"", L"data/historical_market.json", L"", L"1", false, true, false},
    {L"--model", L"model", L"", L"data/model.json", L"", L"1", false, true, false},
    {L"--threshold", L"threshold", L"", L"", L"", L"1", false, true, false},
    {L"--calibrate-threshold", L"calibrate_threshold", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_fetch[] = {
    {L"--symbol", L"symbol", L"", L"", L"", L"1", false, true, false},
    {L"--interval", L"interval", L"", L"", L"", L"1", false, true, false},
    {L"--limit", L"limit", L"", L"500", L"", L"1", false, true, false},
    {L"--batch-size", L"batch_size", L"", L"1000", L"klines per request (spot max 1000, futures max 1500)", L"1", false, true, false},
    {L"--output", L"output", L"", L"data/historical_market.json", L"", L"1", false, true, false},
};

inline constexpr CommandOptionSpec kOptions_live[] = {
    {L"--model", L"model", L"", L"data/model.json", L"", L"1", false, true, false},
    {L"--steps", L"steps", L"", L"20", L"", L"1", false, true, false},
    {L"--sleep", L"sleep", L"", L"5", L"", L"1", false, true, false},
    {L"--leverage", L"leverage", L"", L"", L"override leverage for this run (futures only)", L"1", false, true, false},
    {L"--retrain-interval", L"retrain_interval", L"", L"0", L"retrain model every N steps (0 disables, for adaptive paper/live behavior)", L"1", false, true, false},
    {L"--retrain-window", L"retrain_window", L"", L"300", L"number of recent rows used for each live retrain", L"1", false, true, false},
    {L"--retrain-min-rows", L"retrain_min_rows", L"", L"240", L"minimum rows required before a retrain is attempted", L"1", false, true, false},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"", L"1", false, true, false},
    {L"--batch-size", L"batch_size", L"", L"8192", L"mini-batch size for live retraining", L"1", false, true, false},
    {L"--paper", L"paper", L"", L"false", L"force paper mode for this run even when runtime.dry_run is false", L"0", false, false, false},
    {L"--live", L"live", L"", L"false", L"force authenticated testnet execution even when runtime.dry_run is true", L"0", false, false, false},
    {L"--external-signals", L"external_signals", L"", L"", L"enable cached free external signal adjustment for this run", L"0", false, false, false},
    {L"--no-external-signals", L"external_signals", L"", L"true", L"disable cached free external signal adjustment for this run", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_microstructure_capture[] = {
    {L"--symbols", L"symbols", L"", L"", L"comma-separated supported futures symbols; defaults to configured BTC/ETH/SOL symbols", L"1", false, true, false},
    {L"--seconds", L"seconds", L"", L"60.0", L"", L"1", false, true, false},
    {L"--output-root", L"output_root", L"", L"data/microstructure", L"", L"1", false, true, false},
    {L"--db", L"db", L"", L"data/market_data.sqlite", L"", L"1", false, true, false},
    {L"--timeout", L"timeout", L"", L"10.0", L"", L"1", false, true, false},
    {L"--no-convert", L"convert", L"", L"true", L"capture and validate raw feeds without producing HftBacktest NPZ files", L"0", false, false, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_microstructure_prequential[] = {
    {L"--input", L"input", L"", L"data/microstructure-model.json", L"", L"1", false, true, false},
    {L"--warehouse", L"warehouse", L"", L"data/microstructure.duckdb", L"", L"1", false, true, false},
    {L"--cache-root", L"cache_root", L"", L"data/archive-cache", L"", L"1", false, true, false},
    {L"--output", L"output", L"", L"data/microstructure-prequential.json", L"", L"1", false, true, false},
    {L"--predictions", L"predictions", L"", L"data/microstructure-prequential-predictions.csv", L"", L"1", false, true, false},
    {L"--chart", L"chart", L"", L"data/microstructure-prequential.svg", L"", L"1", false, true, false},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"auto", L"", L"1", false, true, false},
    {L"--training-window-days", L"training_window_days", L"", L"180", L"", L"1", false, true, false},
    {L"--minimum-training-days", L"minimum_training_days", L"", L"60", L"", L"1", false, true, false},
    {L"--calibration-days", L"calibration_days", L"", L"14", L"", L"1", false, true, false},
    {L"--policy-days", L"policy_days", L"", L"14", L"", L"1", false, true, false},
    {L"--evaluation-block-days", L"evaluation_block_days", L"", L"7", L"", L"1", false, true, false},
    {L"--minimum-segment-rows", L"minimum_segment_rows", L"", L"256", L"", L"1", false, true, false},
    {L"--minimum-class-rows", L"minimum_class_rows", L"", L"128", L"", L"1", false, true, false},
    {L"--bootstrap-samples", L"bootstrap_samples", L"", L"2000", L"", L"1", false, true, false},
    {L"--max-folds", L"max_folds", L"", L"0", L"diagnostic cap; any truncated run is ineligible to pass", L"1", false, true, false},
    {L"--memory-limit", L"memory_limit", L"", L"8GB", L"", L"1", false, true, false},
    {L"--threads", L"threads", L"", L"8", L"", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_microstructure_promote[] = {
    {L"--input", L"input", L"", L"data/microstructure-model.json", L"", L"1", false, true, false},
    {L"--prequential-report", L"prequential_report", L"", L"data/microstructure-prequential.json", L"", L"1", false, true, false},
    {L"--prequential-predictions", L"prequential_predictions", L"", L"data/microstructure-prequential-predictions.csv", L"", L"1", false, true, false},
    {L"--prequential-chart", L"prequential_chart", L"", L"data/microstructure-prequential.svg", L"", L"1", false, true, false},
    {L"--warehouse", L"warehouse", L"", L"data/microstructure.duckdb", L"", L"1", false, true, false},
    {L"--cache-root", L"cache_root", L"", L"data/archive-cache", L"", L"1", false, true, false},
    {L"--output", L"output", L"", L"", L"", L"1", false, true, false},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"auto", L"", L"1", false, true, false},
    {L"--memory-limit", L"memory_limit", L"", L"8GB", L"", L"1", false, true, false},
    {L"--threads", L"threads", L"", L"8", L"", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_microstructure_refit[] = {
    {L"--input", L"input", L"", L"data/microstructure-model.json", L"", L"1", false, true, false},
    {L"--output", L"output", L"", L"", L"", L"1", false, true, false},
    {L"--warehouse", L"warehouse", L"", L"data/microstructure.duckdb", L"", L"1", false, true, false},
    {L"--cache-root", L"cache_root", L"", L"data/archive-cache", L"", L"1", false, true, false},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"auto", L"", L"1", false, true, false},
    {L"--memory-limit", L"memory_limit", L"", L"8GB", L"", L"1", false, true, false},
    {L"--threads", L"threads", L"", L"8", L"", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_microstructure_shadow[] = {
    {L"--input", L"input", L"", L"data/microstructure-model.json", L"", L"1", false, true, false},
    {L"--output", L"output", L"", L"", L"", L"1", false, true, false},
    {L"--seconds", L"seconds", L"", L"25260.0", L"public-feed capture duration; promotion requires feature warmup plus six complete evaluated hours", L"1", false, true, false},
    {L"--output-root", L"output_root", L"", L"data/microstructure-shadow/captures", L"", L"1", false, true, false},
    {L"--report", L"report", L"", L"data/microstructure-shadow/report.json", L"", L"1", false, true, false},
    {L"--trades", L"trades", L"", L"data/microstructure-shadow/trades.csv", L"", L"1", false, true, false},
    {L"--db", L"db", L"", L"data/market_data.sqlite", L"", L"1", false, true, false},
    {L"--timeout", L"timeout", L"", L"10.0", L"", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_microstructure_train[] = {
    {L"--symbol", L"symbol", L"", L"BTCUSDT", L"", L"1", false, true, false},
    {L"--warehouse", L"warehouse", L"", L"data/microstructure.duckdb", L"", L"1", false, true, false},
    {L"--cache-root", L"cache_root", L"", L"data/archive-cache", L"", L"1", false, true, false},
    {L"--output", L"output", L"", L"data/microstructure-model.json", L"", L"1", false, true, false},
    {L"--horizon-seconds", L"horizon_seconds", L"", L"900", L"", L"1", false, true, false},
    {L"--decision-cadence-seconds", L"decision_cadence_seconds", L"", L"5", L"evaluate one decision candidate every N seconds while retaining 1s features", L"1", false, true, false},
    {L"--total-latency-ms", L"total_latency_ms", L"", L"750", L"", L"1", false, true, false},
    {L"--taker-fee-bps", L"taker_fee_bps", L"", L"5.0", L"", L"1", false, true, false},
    {L"--additional-slippage-bps-per-side", L"additional_slippage_bps_per_side", L"", L"1.0", L"adverse execution stress charged on both entry and exit notionals in addition to taker fees (default: 1 bps per side)", L"1", false, true, false},
    {L"--max-quote-age-ms", L"max_quote_age_ms", L"", L"1000", L"", L"1", false, true, false},
    {L"--reference-order-notional-quote", L"reference_order_notional_quote", L"", L"1000.0", L"reference quote-currency order size used for L1 executability labels", L"1", false, true, false},
    {L"--max-l1-participation", L"max_l1_participation", L"", L"", L"maximum share of displayed top-of-book quantity; defaults by risk profile", L"1", false, true, false},
    {L"--stop-loss-bps", L"stop_loss_bps", L"", L"", L"", L"1", false, true, false},
    {L"--take-profit-bps", L"take_profit_bps", L"", L"", L"", L"1", false, true, false},
    {L"--trigger-slippage-bps, --stop-slippage-bps", L"trigger_slippage_bps", L"", L"1.0", L"adverse exit-price adjustment after a stop/take trigger (default: 1 bps)", L"1", false, true, false},
    {L"--risk-level", L"risk_level", L"conservative, regular, aggressive", L"conservative", L"", L"1", false, true, false},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"auto", L"", L"1", false, true, false},
    {L"--minimum-promotion-days", L"minimum_promotion_days", L"", L"240", L"minimum observed UTC days for exact-BBO promotion; default 240 within Binance's 320-day official BBO history", L"1", false, true, false},
    {L"--deployment-calibration-days", L"deployment_calibration_days", L"", L"14", L"recent purged tail used only to calibrate the post-validation deployment refit", L"1", false, true, false},
    {L"--maximum-model-age-seconds", L"maximum_model_age_seconds", L"", L"86400", L"hard live-inference expiry measured from the latest labeled refit row", L"1", false, true, false},
    {L"--evaluate-terminal", L"evaluate_terminal", L"", L"false", L"disabled compatibility flag; use hash-bound microstructure-promote", L"0", false, false, false},
    {L"--candidate-only", L"evaluate_terminal", L"", L"false", L"emit a selection-stage candidate without consuming the terminal holdout (default)", L"0", false, false, false},
    {L"--memory-limit", L"memory_limit", L"", L"8GB", L"", L"1", false, true, false},
    {L"--threads", L"threads", L"", L"8", L"", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_model_blueprint[] = {
    {L"--risk-level", L"risk_level", L"conservative, regular, aggressive, default, balanced, risky", L"", L"filter the roadmap to one risk level", L"1", false, true, false},
    {L"--implemented-only", L"implemented_only", L"", L"false", L"hide research-only, blocked, and sandbox model families", L"0", false, false, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_model_lab[] = {
    {L"--output-dir", L"output_dir", L"", L"data/model_lab", L"", L"1", false, true, false},
    {L"--starting-cash", L"starting_cash", L"", L"1000.0", L"", L"1", false, true, false},
    {L"--objective", L"objective", L"", L"", L"objective/risk level to run; repeatable", L"1", false, true, true},
    {L"--max-symbols", L"max_symbols", L"", L"6", L"", L"1", false, true, false},
    {L"--max-scan", L"max_scan", L"", L"250", L"", L"1", false, true, false},
    {L"--limit", L"limit", L"", L"1000", L"candles per selected symbol", L"1", false, true, false},
    {L"--quote-asset", L"quote_asset", L"", L"", L"override runtime quote asset for this lab run", L"1", false, true, false},
    {L"--interval", L"interval", L"", L"", L"override runtime interval for this lab run", L"1", false, true, false},
    {L"--full-history", L"full_history", L"", L"false", L"page klines backward for each selected symbol until no older closed candles are returned", L"0", false, false, false},
    {L"--market-db", L"market_db", L"", L"", L"SQLite market-data database to train from instead of exchange API klines", L"1", false, true, false},
    {L"--require-db-data", L"require_db_data", L"", L"false", L"force model-lab to train from SQLite market data; defaults to data/market_data.sqlite when --market-db is omitted", L"0", false, false, false},
    {L"--market", L"market", L"spot, futures", L"", L"override runtime market type for this lab run", L"1", false, true, false},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"", L"1", false, true, false},
    {L"--batch-size", L"batch_size", L"", L"8192", L"", L"1", false, true, false},
    {L"--score-batch-size", L"score_batch_size", L"", L"", L"", L"1", false, true, false},
    {L"--max-candidates", L"max_candidates", L"", L"", L"smoke/research cap per objective; default evaluates the full grid", L"1", false, true, false},
    {L"--learning-feedback", L"learning_feedback", L"", L"", L"optional learning_feedback.json artifact; default uses data/autonomous/learning_feedback.json when present", L"1", false, true, false},
};

inline constexpr CommandOptionSpec kOptions_positions[] = {
    {L"--stats", L"stats", L"", L"false", L"also print realized + unrealized stats", L"0", false, false, false},
    {L"--learning", L"learning", L"", L"false", L"also print bounded post-trade learning feedback", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_prepare[] = {
    {L"--historical", L"historical", L"", L"data/historical_market.json", L"", L"1", false, true, false},
    {L"--model", L"model", L"", L"data/model.json", L"", L"1", false, true, false},
    {L"--limit", L"limit", L"", L"500", L"", L"1", false, true, false},
    {L"--batch-size", L"batch_size", L"", L"1000", L"klines per fetch request (spot max 1000, futures max 1500)", L"1", false, true, false},
    {L"--preset", L"preset", L"balanced, custom, quick, thorough", L"balanced", L"", L"1", false, true, false},
    {L"--epochs", L"epochs", L"", L"", L"override preset training epochs", L"1", false, true, false},
    {L"--learning-rate", L"learning_rate", L"", L"0.05", L"", L"1", false, true, false},
    {L"--l2-penalty", L"l2_penalty", L"", L"0.0001", L"", L"1", false, true, false},
    {L"--seed", L"seed", L"", L"7", L"", L"1", false, true, false},
    {L"--start-cash", L"start_cash", L"", L"1000.0", L"", L"1", false, true, false},
    {L"--walk-forward", L"walk_forward", L"", L"", L"force walk-forward validation", L"0", false, false, false},
    {L"--no-walk-forward", L"walk_forward", L"", L"", L"skip walk-forward validation", L"0", false, false, false},
    {L"--walk-forward-train", L"walk_forward_train", L"", L"", L"override walk-forward training window", L"1", false, true, false},
    {L"--walk-forward-test", L"walk_forward_test", L"", L"", L"override walk-forward test window", L"1", false, true, false},
    {L"--walk-forward-step", L"walk_forward_step", L"", L"", L"override walk-forward step", L"1", false, true, false},
    {L"--calibrate-threshold", L"calibrate_threshold", L"", L"", L"force threshold calibration", L"0", false, false, false},
    {L"--no-calibrate-threshold", L"calibrate_threshold", L"", L"", L"skip threshold calibration", L"0", false, false, false},
    {L"--online-doctor", L"online_doctor", L"", L"false", L"include exchange connectivity in final readiness checks", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_reconcile[] = {
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
    {L"--output", L"output", L"", L"data/autonomous/reconciliation.json", L"", L"1", false, true, false},
    {L"--quantity-tolerance", L"quantity_tolerance", L"", L"1e-08", L"", L"1", false, true, false},
};

inline constexpr CommandOptionSpec kOptions_report[] = {
    {L"--account", L"account", L"", L"false", L"include authenticated account state", L"0", false, false, false},
    {L"--doctor", L"doctor", L"", L"true", L"include readiness checks", L"0", false, false, false},
    {L"--no-doctor", L"doctor", L"", L"true", L"omit readiness checks", L"0", false, false, false},
    {L"--online", L"online", L"", L"false", L"include exchange connectivity in readiness checks", L"0", false, false, false},
    {L"--input", L"input", L"", L"data/historical_market.json", L"", L"1", false, true, false},
    {L"--model", L"model", L"", L"data/model.json", L"", L"1", false, true, false},
};

inline constexpr CommandOptionSpec kOptions_risk[] = {
    {L"--model", L"model", L"", L"data/model.json", L"", L"1", false, true, false},
    {L"--paper", L"paper", L"", L"false", L"assess paper/dry-run execution", L"0", false, false, false},
    {L"--live", L"live", L"", L"false", L"assess authenticated testnet/demo execution", L"0", false, false, false},
    {L"--leverage", L"leverage", L"", L"", L"optional futures leverage override", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_signals[] = {
    {L"--model", L"model", L"", L"data/model.json", L"model path used to derive default cache location", L"1", false, true, false},
    {L"--cache", L"cache", L"", L"", L"signal cache path (default: model-adjacent data/signals)", L"1", false, true, false},
    {L"--ttl", L"ttl", L"", L"300", L"cache TTL seconds", L"1", false, true, false},
    {L"--timeout", L"timeout", L"", L"3.0", L"per-provider timeout seconds", L"1", false, true, false},
    {L"--max-adjustment", L"max_adjustment", L"", L"0.04", L"maximum model score adjustment", L"1", false, true, false},
    {L"--min-providers", L"min_providers", L"", L"2", L"minimum usable providers for positive boosts", L"1", false, true, false},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"optional backend for news keyword scoring", L"1", false, true, false},
    {L"--short-reaction-refresh", L"short_reaction_refresh", L"", L"30", L"seconds after which cached short-horizon reaction news must refresh", L"1", false, true, false},
    {L"--news-provider-limit", L"news_provider_limit", L"", L"", L"maximum RSS/news providers to poll", L"1", false, true, false},
    {L"--news-items-per-provider", L"news_items_per_provider", L"", L"", L"feed items scored per news provider", L"1", false, true, false},
    {L"--provider-parallelism", L"provider_parallelism", L"", L"", L"maximum simultaneous news provider requests", L"1", false, true, false},
    {L"--provider-jitter", L"provider_jitter", L"", L"", L"random per-provider delay ceiling in seconds", L"1", false, true, false},
    {L"--ollama-news", L"ollama_news", L"", L"", L"enable Ollama AI headline evaluation", L"0", false, false, false},
    {L"--no-ollama-news", L"ollama_news", L"", L"true", L"disable Ollama AI headline evaluation", L"0", false, false, false},
    {L"--ollama-model", L"ollama_model", L"", L"", L"", L"1", false, true, false},
    {L"--ollama-url", L"ollama_url", L"", L"", L"", L"1", false, true, false},
    {L"--ollama-timeout", L"ollama_timeout", L"", L"", L"", L"1", false, true, false},
    {L"--telemetry-db", L"telemetry_db", L"", L"", L"SQLite raw telemetry DB path", L"1", false, true, false},
    {L"--source-grade-max-age-hours", L"source_grade_max_age_hours", L"", L"", L"ignore source grades older than this; 0 disables the age cap", L"1", false, true, false},
    {L"--no-telemetry", L"no_telemetry", L"", L"false", L"do not journal raw provider/model payloads", L"0", false, false, false},
    {L"--loop", L"loop", L"", L"false", L"poll repeatedly with jitter instead of one collection", L"0", false, false, false},
    {L"--iterations", L"iterations", L"", L"0", L"loop iterations; 0 means until interrupted", L"1", false, true, false},
    {L"--sleep", L"sleep", L"", L"", L"base loop interval seconds", L"1", false, true, false},
    {L"--jitter", L"jitter", L"", L"", L"random loop delay ceiling in seconds", L"1", false, true, false},
    {L"--refresh", L"refresh", L"", L"false", L"ignore cache and fetch every provider", L"0", false, false, false},
    {L"--json", L"json", L"", L"false", L"print machine-readable report", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_signals_benchmark[] = {
    {L"--provider-limit", L"provider_limit", L"", L"", L"", L"1", false, true, true},
    {L"--parallelism", L"parallelism", L"", L"", L"", L"1", false, true, true},
    {L"--iterations", L"iterations", L"", L"1", L"", L"1", false, true, false},
    {L"--timeout", L"timeout", L"", L"3.0", L"", L"1", false, true, false},
    {L"--provider-jitter", L"provider_jitter", L"", L"0.0", L"", L"1", false, true, false},
    {L"--ollama-news", L"ollama_news", L"", L"", L"", L"0", false, false, false},
    {L"--no-ollama-news", L"ollama_news", L"", L"true", L"", L"0", false, false, false},
    {L"--ollama-model", L"ollama_model", L"", L"", L"", L"1", false, true, false},
    {L"--ollama-url", L"ollama_url", L"", L"", L"", L"1", false, true, false},
    {L"--ollama-timeout", L"ollama_timeout", L"", L"", L"", L"1", false, true, false},
    {L"--cache", L"cache", L"", L"data/signals/benchmark_external_signals.json", L"", L"1", false, true, false},
    {L"--no-telemetry", L"no_telemetry", L"", L"false", L"", L"0", false, false, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_source_grades[] = {
    {L"--db", L"db", L"", L"", L"SQLite raw telemetry DB path", L"1", false, true, false},
    {L"--window-hours", L"window_hours", L"", L"", L"", L"1", false, true, false},
    {L"--ollama", L"ollama", L"", L"", L"enable Ollama grading", L"0", false, false, false},
    {L"--no-ollama", L"ollama", L"", L"true", L"disable Ollama grading", L"0", false, false, false},
    {L"--ollama-model", L"ollama_model", L"", L"", L"", L"1", false, true, false},
    {L"--ollama-url", L"ollama_url", L"", L"", L"", L"1", false, true, false},
    {L"--ollama-timeout", L"ollama_timeout", L"", L"", L"", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_spot_roundtrip[] = {
    {L"--quantity", L"quantity", L"", L"8e-05", L"base-asset quantity to test", L"1", false, true, false},
    {L"--mode", L"mode", L"auto, buy-sell, sell-buy", L"auto", L"order sequence; auto buys first when quote balance is available, otherwise sells first when base balance is available", L"1", false, true, false},
    {L"--yes", L"yes", L"", L"false", L"confirm signed testnet/demo order placement", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_status[] = {
    {L"--compact", L"compact", L"", L"false", L"print one secret-free operator status line", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_strategy[] = {
    {L"--profile", L"profile", L"active, aggressive, balanced, conservative, custom, regular", L"custom", L"", L"1", false, true, false},
    {L"--risk-level", L"risk_level", L"conservative, regular, aggressive", L"", L"", L"1", false, true, false},
    {L"--reinvest-profits", L"reinvest_profits", L"", L"", L"", L"0", false, false, false},
    {L"--no-reinvest-profits", L"no_reinvest_profits", L"", L"", L"", L"0", false, false, false},
    {L"--leverage", L"leverage", L"", L"", L"", L"1", false, true, false},
    {L"--risk", L"risk", L"", L"", L"", L"1", false, true, false},
    {L"--max-position", L"max_position", L"", L"", L"", L"1", false, true, false},
    {L"--stop", L"stop", L"", L"", L"", L"1", false, true, false},
    {L"--take", L"take", L"", L"", L"", L"1", false, true, false},
    {L"--cooldown", L"cooldown", L"", L"", L"", L"1", false, true, false},
    {L"--min-position-hold-bars", L"min_position_hold_bars", L"", L"", L"", L"1", false, true, false},
    {L"--flat-signal-exit-grace-bars", L"flat_signal_exit_grace_bars", L"", L"", L"", L"1", false, true, false},
    {L"--max-position-hold-bars", L"max_position_hold_bars", L"", L"", L"", L"1", false, true, false},
    {L"--max-open", L"max_open", L"", L"", L"", L"1", false, true, false},
    {L"--min-diversified-assets", L"min_diversified_assets", L"", L"", L"", L"1", false, true, false},
    {L"--max-asset-allocation", L"max_asset_allocation", L"", L"", L"", L"1", false, true, false},
    {L"--max-portfolio-risk", L"max_portfolio_risk", L"", L"", L"", L"1", false, true, false},
    {L"--min-quote-volume-usdc", L"min_quote_volume_usdc", L"", L"", L"", L"1", false, true, false},
    {L"--min-trade-count-24h", L"min_trade_count_24h", L"", L"", L"", L"1", false, true, false},
    {L"--max-spread-bps", L"max_spread_bps", L"", L"", L"", L"1", false, true, false},
    {L"--min-liquidity-score", L"min_liquidity_score", L"", L"", L"", L"1", false, true, false},
    {L"--unpredictability-cooldown", L"unpredictability_cooldown", L"", L"", L"", L"1", false, true, false},
    {L"--max-regime-unpredictability", L"max_regime_unpredictability", L"", L"", L"", L"1", false, true, false},
    {L"--max-prediction-entropy", L"max_prediction_entropy", L"", L"", L"", L"1", false, true, false},
    {L"--min-model-confidence", L"min_model_confidence", L"", L"", L"", L"1", false, true, false},
    {L"--max-trades-per-day", L"max_trades_per_day", L"", L"", L"", L"1", false, true, false},
    {L"--signal-threshold", L"signal_threshold", L"", L"", L"", L"1", false, true, false},
    {L"--max-drawdown", L"max_drawdown", L"", L"", L"", L"1", false, true, false},
    {L"--max-daily-loss", L"max_daily_loss", L"", L"", L"", L"1", false, true, false},
    {L"--max-session-loss", L"max_session_loss", L"", L"", L"", L"1", false, true, false},
    {L"--max-consecutive-losses", L"max_consecutive_losses", L"", L"", L"", L"1", false, true, false},
    {L"--max-network-errors", L"max_network_errors", L"", L"", L"", L"1", false, true, false},
    {L"--recovery-cooldown-seconds", L"recovery_cooldown_seconds", L"", L"", L"", L"1", false, true, false},
    {L"--taker-fee-bps", L"taker_fee_bps", L"", L"", L"", L"1", false, true, false},
    {L"--slippage-bps", L"slippage_bps", L"", L"", L"", L"1", false, true, false},
    {L"--label-threshold", L"label_threshold", L"", L"", L"", L"1", false, true, false},
    {L"--model-lookback", L"model_lookback", L"", L"", L"", L"1", false, true, false},
    {L"--training-epochs", L"training_epochs", L"", L"", L"", L"1", false, true, false},
    {L"--confidence-beta", L"confidence_beta", L"", L"", L"", L"1", false, true, false},
    {L"--feature-window-short", L"feature_window_short", L"", L"", L"", L"1", false, true, false},
    {L"--feature-window-long", L"feature_window_long", L"", L"", L"", L"1", false, true, false},
    {L"--set-features", L"set_features", L"", L"", L"comma-separated ordered feature list for retraining", L"1", false, true, false},
    {L"--enable-feature", L"enable_feature", L"", L"", L"enable a feature by name", L"1", false, true, true},
    {L"--disable-feature", L"disable_feature", L"", L"", L"disable a feature by name", L"1", false, true, true},
    {L"--external-signals", L"external_signals", L"", L"", L"enable live free external signals", L"0", false, false, false},
    {L"--no-external-signals", L"external_signals", L"", L"true", L"disable live free external signals", L"0", false, false, false},
    {L"--external-signal-max-adjustment", L"external_signal_max_adjustment", L"", L"", L"", L"1", false, true, false},
    {L"--external-signal-min-providers", L"external_signal_min_providers", L"", L"", L"", L"1", false, true, false},
    {L"--external-signal-ttl", L"external_signal_ttl", L"", L"", L"", L"1", false, true, false},
    {L"--external-signal-timeout", L"external_signal_timeout", L"", L"", L"", L"1", false, true, false},
    {L"--external-news-ai", L"external_news_ai", L"", L"", L"", L"0", false, false, false},
    {L"--no-external-news-ai", L"external_news_ai", L"", L"true", L"", L"0", false, false, false},
    {L"--external-news-ai-model", L"external_news_ai_model", L"", L"", L"", L"1", false, true, false},
    {L"--external-news-ai-url", L"external_news_ai_url", L"", L"", L"", L"1", false, true, false},
    {L"--external-news-ai-timeout", L"external_news_ai_timeout", L"", L"", L"", L"1", false, true, false},
    {L"--external-news-provider-limit", L"external_news_provider_limit", L"", L"", L"", L"1", false, true, false},
    {L"--external-provider-parallelism", L"external_provider_parallelism", L"", L"", L"", L"1", false, true, false},
    {L"--external-provider-jitter", L"external_provider_jitter", L"", L"", L"", L"1", false, true, false},
    {L"--external-poll-jitter", L"external_poll_jitter", L"", L"", L"", L"1", false, true, false},
    {L"--telemetry-db", L"telemetry_db", L"", L"", L"", L"1", false, true, false},
    {L"--no-telemetry", L"no_telemetry", L"", L"", L"", L"0", false, false, false},
    {L"--source-grading", L"source_grading", L"", L"", L"", L"0", false, false, false},
    {L"--no-source-grading", L"source_grading", L"", L"true", L"", L"0", false, false, false},
    {L"--source-grading-interval", L"source_grading_interval", L"", L"", L"", L"1", false, true, false},
    {L"--source-grading-window-hours", L"source_grading_window_hours", L"", L"", L"", L"1", false, true, false},
    {L"--source-grade-max-age-hours", L"source_grade_max_age_hours", L"", L"", L"", L"1", false, true, false},
};

inline constexpr CommandOptionSpec kOptions_tape_depth_confirm[] = {
    {L"--selection", L"selection", L"", L"", L"", L"1", true, true, false},
    {L"--report", L"report", L"", L"", L"", L"1", true, true, false},
    {L"--output", L"output", L"", L"data/tape-depth-confirmation.json", L"", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_tape_depth_design[] = {
    {L"--risk-level", L"risk_level", L"conservative, regular, aggressive", L"conservative", L"", L"1", false, true, false},
    {L"--sampled-count", L"sampled_count", L"", L"24", L"", L"1", false, true, false},
    {L"--seed", L"seed", L"", L"20260710", L"", L"1", false, true, false},
    {L"--output", L"output", L"", L"data/tape-depth-experiment-design.json", L"", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_tape_depth_execution_confirm[] = {
    {L"--design", L"design", L"", L"docs/model-research/tape-depth/confirmation-design.json", L"", L"1", false, true, false},
    {L"--availability", L"availability", L"", L"docs/microstructure/availability.json", L"", L"1", false, true, false},
    {L"--warehouse", L"warehouse", L"", L"data/microstructure.duckdb", L"", L"1", false, true, false},
    {L"--cache-root", L"cache_root", L"", L"data/archive-cache", L"", L"1", false, true, false},
    {L"--output-dir", L"output_dir", L"", L"data/tape-depth-execution-confirmation", L"", L"1", false, true, false},
    {L"--memory-limit", L"memory_limit", L"", L"8GB", L"", L"1", false, true, false},
    {L"--threads", L"threads", L"", L"8", L"", L"1", false, true, false},
    {L"--resume", L"resume", L"", L"false", L"", L"0", false, false, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_tape_depth_prequential[] = {
    {L"--symbols", L"symbols", L"", L"BTCUSDT,ETHUSDT,SOLUSDT", L"", L"1", false, true, false},
    {L"--warehouse", L"warehouse", L"", L"data/microstructure.duckdb", L"", L"1", false, true, false},
    {L"--cache-root", L"cache_root", L"", L"data/archive-cache", L"", L"1", false, true, false},
    {L"--output-dir", L"output_dir", L"", L"data/tape-depth-prequential", L"", L"1", false, true, false},
    {L"--training-window-days", L"training_window_days", L"", L"730", L"", L"1", false, true, false},
    {L"--tuning-window-days", L"tuning_window_days", L"", L"30", L"", L"1", false, true, false},
    {L"--calibration-window-days", L"calibration_window_days", L"", L"30", L"", L"1", false, true, false},
    {L"--evaluation-window-days", L"evaluation_window_days", L"", L"90", L"", L"1", false, true, false},
    {L"--horizon-seconds", L"horizon_seconds", L"", L"", L"default 60; sealed confirmation derives the frozen winner", L"1", false, true, false},
    {L"--total-latency-ms", L"total_latency_ms", L"", L"750", L"", L"1", false, true, false},
    {L"--decision-cadence-seconds", L"decision_cadence_seconds", L"", L"", L"default 20; sealed confirmation derives the frozen winner", L"1", false, true, false},
    {L"--maximum-depth-age-ms", L"maximum_depth_age_ms", L"", L"", L"default 60000; sealed confirmation derives the frozen winner", L"1", false, true, false},
    {L"--maximum-rows", L"maximum_rows", L"", L"5000000", L"", L"1", false, true, false},
    {L"--maximum-cached-rows", L"maximum_cached_rows", L"", L"15000000", L"", L"1", false, true, false},
    {L"--no-dataset-cache", L"dataset_cache", L"", L"true", L"disable the verified DuckDB derived-dataset cache", L"0", false, false, false},
    {L"--study-stage", L"study_stage", L"development, screening, confirmation", L"development", L"", L"1", false, true, false},
    {L"--selection-lock", L"selection_lock", L"", L"", L"winner lock required for sealed confirmation", L"1", false, true, false},
    {L"--max-folds", L"max_folds", L"", L"0", L"", L"1", false, true, false},
    {L"--risk-level", L"risk_level", L"conservative, regular, aggressive", L"conservative", L"", L"1", false, true, false},
    {L"--model-profile", L"model_profile", L"regularized, balanced, expressive", L"", L"default regularized; confirmation derives the frozen winner", L"1", false, true, false},
    {L"--feature-set", L"feature_set", L"core, tape_derived, cross_asset, full", L"", L"default full; confirmation derives the frozen winner", L"1", false, true, false},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"auto", L"", L"1", false, true, false},
    {L"--minimum-segment-rows", L"minimum_segment_rows", L"", L"10000", L"", L"1", false, true, false},
    {L"--memory-limit", L"memory_limit", L"", L"8GB", L"", L"1", false, true, false},
    {L"--threads", L"threads", L"", L"8", L"", L"1", false, true, false},
    {L"--plan-only", L"plan_only", L"", L"false", L"", L"0", false, false, false},
    {L"--resume", L"resume", L"", L"false", L"", L"0", false, false, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_tape_depth_select[] = {
    {L"--report", L"report", L"", L"", L"screening report path; repeat for every declared trial", L"1", true, true, true},
    {L"--design", L"design", L"", L"", L"precommitted multi-fidelity experiment design JSON", L"1", true, true, false},
    {L"--output", L"output", L"", L"data/tape-depth-selection.json", L"", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_tape_depth_study[] = {
    {L"--symbols", L"symbols", L"", L"BTCUSDT,ETHUSDT,SOLUSDT", L"", L"1", false, true, false},
    {L"--design", L"design", L"", L"", L"", L"1", true, true, false},
    {L"--warehouse", L"warehouse", L"", L"data/microstructure.duckdb", L"", L"1", false, true, false},
    {L"--cache-root", L"cache_root", L"", L"data/archive-cache", L"", L"1", false, true, false},
    {L"--output-dir", L"output_dir", L"", L"data/tape-depth-study", L"", L"1", false, true, false},
    {L"--training-window-days", L"training_window_days", L"", L"730", L"", L"1", false, true, false},
    {L"--tuning-window-days", L"tuning_window_days", L"", L"30", L"", L"1", false, true, false},
    {L"--calibration-window-days", L"calibration_window_days", L"", L"30", L"", L"1", false, true, false},
    {L"--evaluation-window-days", L"evaluation_window_days", L"", L"90", L"", L"1", false, true, false},
    {L"--total-latency-ms", L"total_latency_ms", L"", L"750", L"", L"1", false, true, false},
    {L"--maximum-rows", L"maximum_rows", L"", L"5000000", L"", L"1", false, true, false},
    {L"--maximum-cached-rows", L"maximum_cached_rows", L"", L"15000000", L"", L"1", false, true, false},
    {L"--no-dataset-cache", L"dataset_cache", L"", L"true", L"", L"0", false, false, false},
    {L"--max-folds", L"max_folds", L"4, 6, 8, 10", L"4", L"", L"1", false, true, false},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"auto", L"", L"1", false, true, false},
    {L"--minimum-segment-rows", L"minimum_segment_rows", L"", L"10000", L"", L"1", false, true, false},
    {L"--memory-limit", L"memory_limit", L"", L"8GB", L"", L"1", false, true, false},
    {L"--threads", L"threads", L"", L"8", L"", L"1", false, true, false},
    {L"--resume", L"resume", L"", L"false", L"", L"0", false, false, false},
    {L"--plan-only", L"plan_only", L"", L"false", L"", L"0", false, false, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_tape_depth_train[] = {
    {L"--symbol", L"symbol", L"", L"BTCUSDT", L"", L"1", false, true, false},
    {L"--warehouse", L"warehouse", L"", L"data/microstructure.duckdb", L"", L"1", false, true, false},
    {L"--cache-root", L"cache_root", L"", L"data/archive-cache", L"", L"1", false, true, false},
    {L"--output", L"output", L"", L"data/tape-depth-model.json", L"", L"1", false, true, false},
    {L"--window-days", L"window_days", L"", L"180", L"", L"1", false, true, false},
    {L"--end-date", L"end_date", L"", L"", L"optional inclusive UTC evaluation date; defaults to latest covered target", L"1", false, true, false},
    {L"--horizon-seconds", L"horizon_seconds", L"", L"60", L"", L"1", false, true, false},
    {L"--total-latency-ms", L"total_latency_ms", L"", L"750", L"", L"1", false, true, false},
    {L"--decision-cadence-seconds", L"decision_cadence_seconds", L"", L"5", L"", L"1", false, true, false},
    {L"--maximum-depth-age-ms", L"maximum_depth_age_ms", L"", L"60000", L"", L"1", false, true, false},
    {L"--risk-level", L"risk_level", L"conservative, regular, aggressive", L"conservative", L"", L"1", false, true, false},
    {L"--model-profile", L"model_profile", L"regularized, balanced, expressive", L"regularized", L"", L"1", false, true, false},
    {L"--feature-set", L"feature_set", L"core, tape_derived, cross_asset, full", L"full", L"", L"1", false, true, false},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"auto", L"", L"1", false, true, false},
    {L"--minimum-segment-rows", L"minimum_segment_rows", L"", L"2000", L"", L"1", false, true, false},
    {L"--maximum-rows", L"maximum_rows", L"", L"5000000", L"", L"1", false, true, false},
    {L"--memory-limit", L"memory_limit", L"", L"8GB", L"", L"1", false, true, false},
    {L"--threads", L"threads", L"", L"8", L"", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_tick_archive_sync[] = {
    {L"--symbols", L"symbols", L"", L"", L"comma-separated BTC/ETH/SOL futures symbols; defaults to runtime symbols", L"1", false, true, false},
    {L"--data-types", L"data_types", L"", L"bookTicker,trades", L"comma-separated official products: bookTicker,trades,bookDepth", L"1", false, true, false},
    {L"--start-date", L"start_date", L"", L"", L"inclusive UTC date, YYYY-MM-DD", L"1", false, true, false},
    {L"--end-date", L"end_date", L"", L"", L"inclusive UTC date, YYYY-MM-DD", L"1", false, true, false},
    {L"--full-history", L"full_history", L"", L"false", L"discover and select every official file independently for each symbol/data type", L"0", false, false, false},
    {L"--available-only", L"available_only", L"", L"false", L"record but do not fail on unavailable symbol/data-type dates", L"0", false, false, false},
    {L"--plan-only", L"plan_only", L"", L"false", L"report official file coverage and compressed bytes without downloading", L"0", false, false, false},
    {L"--plan-output", L"plan_output", L"", L"", L"optional atomic JSON path for the compact official coverage plan", L"1", false, true, false},
    {L"--max-planned-gb", L"max_planned_gb", L"", L"500.0", L"block downloads above this official compressed-byte plan; use 0 to disable", L"1", false, true, false},
    {L"--warehouse", L"warehouse", L"", L"data/microstructure.duckdb", L"", L"1", false, true, false},
    {L"--cache-root", L"cache_root", L"", L"data/archive-cache", L"", L"1", false, true, false},
    {L"--memory-limit", L"memory_limit", L"", L"8GB", L"", L"1", false, true, false},
    {L"--threads", L"threads", L"", L"8", L"", L"1", false, true, false},
    {L"--timeout", L"timeout", L"", L"240.0", L"", L"1", false, true, false},
    {L"--no-retain-archive", L"no_retain_archive", L"", L"false", L"", L"0", false, false, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_tick_corpus_audit[] = {
    {L"--symbols", L"symbols", L"", L"BTCUSDT,ETHUSDT,SOLUSDT", L"comma-separated BTC/ETH/SOL futures symbols", L"1", false, true, false},
    {L"--data-types", L"data_types", L"", L"bookTicker,trades,bookDepth", L"comma-separated official products: bookTicker,trades,bookDepth", L"1", false, true, false},
    {L"--start-date", L"start_date", L"", L"", L"", L"1", false, true, false},
    {L"--end-date", L"end_date", L"", L"", L"", L"1", false, true, false},
    {L"--strict-book-depth-calendar", L"allow_provider_book_depth_gaps", L"", L"true", L"reject dates absent from Binance's official bookDepth listing; by default those provider-proven absences are reported but permitted", L"0", false, false, false},
    {L"--warehouse", L"warehouse", L"", L"data/microstructure.duckdb", L"", L"1", false, true, false},
    {L"--cache-root", L"cache_root", L"", L"data/archive-cache", L"", L"1", false, true, false},
    {L"--memory-limit", L"memory_limit", L"", L"8GB", L"", L"1", false, true, false},
    {L"--threads", L"threads", L"", L"8", L"", L"1", false, true, false},
    {L"--output", L"output", L"", L"", L"", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_train[] = {
    {L"--input", L"input", L"", L"data/historical_market.json", L"", L"1", false, true, false},
    {L"--output", L"output", L"", L"data/model.json", L"", L"1", false, true, false},
    {L"--source", L"source", L"auto, file, db", L"auto", L"", L"1", false, true, false},
    {L"--db", L"db", L"", L"data/market_data.sqlite", L"", L"1", false, true, false},
    {L"--interval", L"interval", L"", L"", L"", L"1", false, true, false},
    {L"--market", L"market", L"spot, futures", L"", L"", L"1", false, true, false},
    {L"--min-rows", L"min_rows", L"", L"120", L"", L"1", false, true, false},
    {L"--download-missing", L"download_missing", L"", L"false", L"", L"0", false, false, false},
    {L"--preset", L"preset", L"balanced, custom, quick, thorough", L"custom", L"", L"1", false, true, false},
    {L"--epochs", L"epochs", L"", L"250", L"", L"1", false, true, false},
    {L"--learning-rate", L"learning_rate", L"", L"0.05", L"", L"1", false, true, false},
    {L"--l2-penalty", L"l2_penalty", L"", L"0.0001", L"", L"1", false, true, false},
    {L"--seed", L"seed", L"", L"7", L"", L"1", false, true, false},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"training backend override; default uses saved runtime compute_backend", L"1", false, true, false},
    {L"--batch-size", L"batch_size", L"", L"8192", L"mini-batch size for GPU training", L"1", false, true, false},
    {L"--walk-forward", L"walk_forward", L"", L"false", L"run walk-forward validation before final training", L"0", false, false, false},
    {L"--walk-forward-train", L"walk_forward_train", L"", L"300", L"", L"1", false, true, false},
    {L"--walk-forward-test", L"walk_forward_test", L"", L"60", L"", L"1", false, true, false},
    {L"--walk-forward-step", L"walk_forward_step", L"", L"30", L"", L"1", false, true, false},
    {L"--calibrate-threshold", L"calibrate_threshold", L"", L"false", L"optimize a probability threshold on validation split", L"0", false, false, false},
};

inline constexpr CommandOptionSpec kOptions_train_suite[] = {
    {L"--input", L"input", L"", L"data/historical_market.json", L"", L"1", false, true, false},
    {L"--output-dir", L"output_dir", L"", L"data", L"", L"1", false, true, false},
    {L"--symbol", L"symbol", L"", L"", L"explicit asset identity for durable terminal governance; omission is research-only", L"1", false, true, false},
    {L"--starting-cash", L"starting_cash", L"", L"1000.0", L"", L"1", false, true, false},
    {L"--objective", L"objective", L"", L"", L"restrict suite to named objective(s); repeat to list multiple.", L"1", false, true, true},
    {L"--max-workers", L"max_workers", L"", L"", L"parallel candidate workers; defaults to available CPU cores", L"1", false, true, false},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"training backend override; GPU backends run candidates sequentially to protect VRAM", L"1", false, true, false},
    {L"--batch-size", L"batch_size", L"", L"8192", L"mini-batch size for GPU training", L"1", false, true, false},
    {L"--max-candidates", L"max_candidates", L"", L"", L"smoke/research cap per objective; default evaluates the full grid", L"1", false, true, false},
};

inline constexpr CommandOptionSpec kOptions_tune[] = {
    {L"--input", L"input", L"", L"data/historical_market.json", L"", L"1", false, true, false},
    {L"--save-best", L"save_best", L"", L"false", L"", L"0", false, false, false},
    {L"--min-risk", L"min_risk", L"", L"0.002", L"", L"1", false, true, false},
    {L"--max-risk", L"max_risk", L"", L"0.02", L"", L"1", false, true, false},
    {L"--steps", L"steps", L"", L"5", L"", L"1", false, true, false},
    {L"--min-leverage", L"min_leverage", L"", L"1.0", L"", L"1", false, true, false},
    {L"--max-leverage", L"max_leverage", L"", L"20.0", L"", L"1", false, true, false},
    {L"--min-threshold", L"min_threshold", L"", L"0.52", L"", L"1", false, true, false},
    {L"--max-threshold", L"max_threshold", L"", L"0.88", L"", L"1", false, true, false},
    {L"--min-take", L"min_take", L"", L"0.01", L"", L"1", false, true, false},
    {L"--max-take", L"max_take", L"", L"0.06", L"", L"1", false, true, false},
    {L"--min-stop", L"min_stop", L"", L"0.008", L"", L"1", false, true, false},
    {L"--max-stop", L"max_stop", L"", L"0.04", L"", L"1", false, true, false},
    {L"--compute-backend", L"compute_backend", L"cpu, cuda, rocm, directml, mps, auto", L"", L"", L"1", false, true, false},
    {L"--batch-size", L"batch_size", L"", L"8192", L"mini-batch size for accelerated tuning", L"1", false, true, false},
    {L"--lookback-days", L"lookback_days", L"", L"", L"use only the most recent N days of candles for tuning", L"1", false, true, false},
    {L"--from-date", L"from_date", L"", L"", L"inclusive start date for tuning window (YYYY-MM-DD)", L"1", false, true, false},
    {L"--to-date", L"to_date", L"", L"", L"inclusive end date for tuning window (YYYY-MM-DD)", L"1", false, true, false},
};

inline constexpr CommandOptionSpec kOptions_universe[] = {
    {L"--symbols", L"symbols", L"", L"", L"comma-separated symbols; default uses runtime.symbols", L"1", false, true, false},
    {L"--json", L"json", L"", L"false", L"", L"0", false, false, false},
};

inline constexpr CommandSpec kCommands[] = {
    {L"ai", L"usage: simple-ai-trading ai [-h] [--enable] [--disable] [--provider PROVIDER]                             [--model MODEL] [--require-gpu] [--no-require-gpu]                             [--min-free-vram-gb MIN_FREE_VRAM_GB]                             [--min-free-ram-gb MIN_FREE_RAM_GB]                             [--min-model-parameters-b MIN_MODEL_PARAMETERS_B]                             [--allow-paper-fallback] [--no-paper-fallback]                             [--json]", kOptions_ai, 12},
    {L"ai-benchmark", L"usage: simple-ai-trading ai-benchmark [-h] [--models MODELS] [--url URL]                                       [--timeout TIMEOUT]                                       [--minimum-score MINIMUM_SCORE]                                       [--output OUTPUT] [--json]", kOptions_ai_benchmark, 6},
    {L"ai-forecast-benchmark", L"usage: simple-ai-trading ai-forecast-benchmark [-h] [--database DATABASE]                                                [--model-size {small,base}]                                                [--backend {cpu,cuda,rocm,directml,mps,auto}]                                                [--source-cache SOURCE_CACHE]                                                [--bootstrap-source]                                                [--repair-source] [--allow-cpu]                                                [--start START]                                                [--end-exclusive END_EXCLUSIVE]                                                [--samples-per-symbol SAMPLES_PER_SYMBOL]                                                [--lookback-bars LOOKBACK_BARS]                                                [--prediction-bars PREDICTION_BARS]                                                [--batch-size BATCH_SIZE]                                                [--inference-samples INFERENCE_SAMPLES]                                                [--temperature TEMPERATURE]                                                [--top-k TOP_K] [--top-p TOP_P]                                                [--include-volume]                                                [--seed SEED]                                                [--bootstrap-samples BOOTSTRAP_SAMPLES]                                                [--worker-timeout WORKER_TIMEOUT]                                                [--max-worker-restarts MAX_WORKER_RESTARTS]                                                [--worker-rotation-batches WORKER_ROTATION_BATCHES]                                                [--observations OBSERVATIONS]                                                [--output OUTPUT]                                                [--chart CHART] [--json]", kOptions_ai_forecast_benchmark, 27},
    {L"ai-review", L"usage: simple-ai-trading ai-review [-h] [--report REPORT] [--output OUTPUT]                                    [--model MODEL] [--url URL]                                    [--timeout TIMEOUT] [--json]", kOptions_ai_review, 6},
    {L"api-budget", L"usage: simple-ai-trading api-budget [-h] [--db DB] [--market {spot,futures}]                                     [--refresh] [--cached-only]                                     [--max-age-seconds MAX_AGE_SECONDS]                                     [--compact] [--json]", kOptions_api_budget, 7},
    {L"archive-sync", L"usage: simple-ai-trading archive-sync [-h] [--db DB] [--symbol SYMBOL]                                       [--symbols SYMBOLS]                                       [--top-symbols TOP_SYMBOLS]                                       [--quote-asset QUOTE_ASSET]                                       [--max-scan MAX_SCAN]                                       [--min-history-months MIN_HISTORY_MONTHS]                                       [--interval INTERVAL]                                       [--market {spot,futures}]                                       [--cadence {monthly,daily}]                                       [--data-type {klines,aggTrades}]                                       [--max-files MAX_FILES]                                       [--start-period START_PERIOD]                                       [--end-period END_PERIOD] [--plan-only]                                       [--max-planned-gb MAX_PLANNED_GB]                                       [--timeout TIMEOUT] [--force]                                       [--aggregate-only]                                       [--no-verify-checksum]                                       [--require-checksum] [--json]", kOptions_archive_sync, 22},
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
    {L"microstructure-capture", L"usage: simple-ai-trading microstructure-capture [-h] [--symbols SYMBOLS]                                                 [--seconds SECONDS]                                                 [--output-root OUTPUT_ROOT]                                                 [--db DB] [--timeout TIMEOUT]                                                 [--no-convert] [--json]", kOptions_microstructure_capture, 7},
    {L"microstructure-prequential", L"usage: simple-ai-trading microstructure-prequential [-h] [--input INPUT]                                                     [--warehouse WAREHOUSE]                                                     [--cache-root CACHE_ROOT]                                                     [--output OUTPUT]                                                     [--predictions PREDICTIONS]                                                     [--chart CHART]                                                     [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                                                     [--training-window-days TRAINING_WINDOW_DAYS]                                                     [--minimum-training-days MINIMUM_TRAINING_DAYS]                                                     [--calibration-days CALIBRATION_DAYS]                                                     [--policy-days POLICY_DAYS]                                                     [--evaluation-block-days EVALUATION_BLOCK_DAYS]                                                     [--minimum-segment-rows MINIMUM_SEGMENT_ROWS]                                                     [--minimum-class-rows MINIMUM_CLASS_ROWS]                                                     [--bootstrap-samples BOOTSTRAP_SAMPLES]                                                     [--max-folds MAX_FOLDS]                                                     [--memory-limit MEMORY_LIMIT]                                                     [--threads THREADS]                                                     [--json]", kOptions_microstructure_prequential, 19},
    {L"microstructure-promote", L"usage: simple-ai-trading microstructure-promote [-h] [--input INPUT]                                                 [--prequential-report PREQUENTIAL_REPORT]                                                 [--prequential-predictions PREQUENTIAL_PREDICTIONS]                                                 [--prequential-chart PREQUENTIAL_CHART]                                                 [--warehouse WAREHOUSE]                                                 [--cache-root CACHE_ROOT]                                                 [--output OUTPUT]                                                 [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                                                 [--memory-limit MEMORY_LIMIT]                                                 [--threads THREADS] [--json]", kOptions_microstructure_promote, 11},
    {L"microstructure-refit", L"usage: simple-ai-trading microstructure-refit [-h] [--input INPUT]                                               [--output OUTPUT]                                               [--warehouse WAREHOUSE]                                               [--cache-root CACHE_ROOT]                                               [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                                               [--memory-limit MEMORY_LIMIT]                                               [--threads THREADS] [--json]", kOptions_microstructure_refit, 8},
    {L"microstructure-shadow", L"usage: simple-ai-trading microstructure-shadow [-h] [--input INPUT]                                                [--output OUTPUT]                                                [--seconds SECONDS]                                                [--output-root OUTPUT_ROOT]                                                [--report REPORT]                                                [--trades TRADES] [--db DB]                                                [--timeout TIMEOUT] [--json]", kOptions_microstructure_shadow, 9},
    {L"microstructure-train", L"usage: simple-ai-trading microstructure-train [-h] [--symbol SYMBOL]                                               [--warehouse WAREHOUSE]                                               [--cache-root CACHE_ROOT]                                               [--output OUTPUT]                                               [--horizon-seconds HORIZON_SECONDS]                                               [--decision-cadence-seconds DECISION_CADENCE_SECONDS]                                               [--total-latency-ms TOTAL_LATENCY_MS]                                               [--taker-fee-bps TAKER_FEE_BPS]                                               [--additional-slippage-bps-per-side ADDITIONAL_SLIPPAGE_BPS_PER_SIDE]                                               [--max-quote-age-ms MAX_QUOTE_AGE_MS]                                               [--reference-order-notional-quote REFERENCE_ORDER_NOTIONAL_QUOTE]                                               [--max-l1-participation MAX_L1_PARTICIPATION]                                               [--stop-loss-bps STOP_LOSS_BPS]                                               [--take-profit-bps TAKE_PROFIT_BPS]                                               [--trigger-slippage-bps TRIGGER_SLIPPAGE_BPS]                                               [--risk-level {conservative,regular,aggressive}]                                               [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                                               [--minimum-promotion-days MINIMUM_PROMOTION_DAYS]                                               [--deployment-calibration-days DEPLOYMENT_CALIBRATION_DAYS]                                               [--maximum-model-age-seconds MAXIMUM_MODEL_AGE_SECONDS]                                               [--evaluate-terminal | --candidate-only]                                               [--memory-limit MEMORY_LIMIT]                                               [--threads THREADS] [--json]", kOptions_microstructure_train, 25},
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
    {L"status", L"usage: simple-ai-trading status [-h] [--compact]", kOptions_status, 1},
    {L"strategy", L"usage: simple-ai-trading strategy [-h]                                   [--profile {active,aggressive,balanced,conservative,custom,regular}]                                   [--risk-level {conservative,regular,aggressive}]                                   [--reinvest-profits] [--no-reinvest-profits]                                   [--leverage LEVERAGE] [--risk RISK]                                   [--max-position MAX_POSITION] [--stop STOP]                                   [--take TAKE] [--cooldown COOLDOWN]                                   [--min-position-hold-bars MIN_POSITION_HOLD_BARS]                                   [--flat-signal-exit-grace-bars FLAT_SIGNAL_EXIT_GRACE_BARS]                                   [--max-position-hold-bars MAX_POSITION_HOLD_BARS]                                   [--max-open MAX_OPEN]                                   [--min-diversified-assets MIN_DIVERSIFIED_ASSETS]                                   [--max-asset-allocation MAX_ASSET_ALLOCATION]                                   [--max-portfolio-risk MAX_PORTFOLIO_RISK]                                   [--min-quote-volume-usdc MIN_QUOTE_VOLUME_USDC]                                   [--min-trade-count-24h MIN_TRADE_COUNT_24H]                                   [--max-spread-bps MAX_SPREAD_BPS]                                   [--min-liquidity-score MIN_LIQUIDITY_SCORE]                                   [--unpredictability-cooldown UNPREDICTABILITY_COOLDOWN]                                   [--max-regime-unpredictability MAX_REGIME_UNPREDICTABILITY]                                   [--max-prediction-entropy MAX_PREDICTION_ENTROPY]                                   [--min-model-confidence MIN_MODEL_CONFIDENCE]                                   [--max-trades-per-day MAX_TRADES_PER_DAY]                                   [--signal-threshold SIGNAL_THRESHOLD]                                   [--max-drawdown MAX_DRAWDOWN]                                   [--max-daily-loss MAX_DAILY_LOSS]                                   [--max-session-loss MAX_SESSION_LOSS]                                   [--max-consecutive-losses MAX_CONSECUTIVE_LOSSES]                                   [--max-network-errors MAX_NETWORK_ERRORS]                                   [--recovery-cooldown-seconds RECOVERY_COOLDOWN_SECONDS]                                   [--taker-fee-bps TAKER_FEE_BPS]                                   [--slippage-bps SLIPPAGE_BPS]                                   [--label-threshold LABEL_THRESHOLD]                                   [--model-lookback MODEL_LOOKBACK]                                   [--training-epochs TRAINING_EPOCHS]                                   [--confidence-beta CONFIDENCE_BETA]                                   [--feature-window-short FEATURE_WINDOW_SHORT]                                   [--feature-window-long FEATURE_WINDOW_LONG]                                   [--set-features SET_FEATURES]                                   [--enable-feature ENABLE_FEATURE]                                   [--disable-feature DISABLE_FEATURE]                                   [--external-signals] [--no-external-signals]                                   [--external-signal-max-adjustment EXTERNAL_SIGNAL_MAX_ADJUSTMENT]                                   [--external-signal-min-providers EXTERNAL_SIGNAL_MIN_PROVIDERS]                                   [--external-signal-ttl EXTERNAL_SIGNAL_TTL]                                   [--external-signal-timeout EXTERNAL_SIGNAL_TIMEOUT]                                   [--external-news-ai] [--no-external-news-ai]                                   [--external-news-ai-model EXTERNAL_NEWS_AI_MODEL]                                   [--external-news-ai-url EXTERNAL_NEWS_AI_URL]                                   [--external-news-ai-timeout EXTERNAL_NEWS_AI_TIMEOUT]                                   [--external-news-provider-limit EXTERNAL_NEWS_PROVIDER_LIMIT]                                   [--external-provider-parallelism EXTERNAL_PROVIDER_PARALLELISM]                                   [--external-provider-jitter EXTERNAL_PROVIDER_JITTER]                                   [--external-poll-jitter EXTERNAL_POLL_JITTER]                                   [--telemetry-db TELEMETRY_DB]                                   [--no-telemetry] [--source-grading]                                   [--no-source-grading]                                   [--source-grading-interval SOURCE_GRADING_INTERVAL]                                   [--source-grading-window-hours SOURCE_GRADING_WINDOW_HOURS]                                   [--source-grade-max-age-hours SOURCE_GRADE_MAX_AGE_HOURS]", kOptions_strategy, 66},
    {L"tape-depth-confirm", L"usage: simple-ai-trading tape-depth-confirm [-h] --selection SELECTION                                             --report REPORT [--output OUTPUT]                                             [--json]", kOptions_tape_depth_confirm, 4},
    {L"tape-depth-design", L"usage: simple-ai-trading tape-depth-design [-h]                                            [--risk-level {conservative,regular,aggressive}]                                            [--sampled-count SAMPLED_COUNT]                                            [--seed SEED] [--output OUTPUT]                                            [--json]", kOptions_tape_depth_design, 5},
    {L"tape-depth-execution-confirm", L"usage: simple-ai-trading tape-depth-execution-confirm [-h] [--design DESIGN]                                                       [--availability AVAILABILITY]                                                       [--warehouse WAREHOUSE]                                                       [--cache-root CACHE_ROOT]                                                       [--output-dir OUTPUT_DIR]                                                       [--memory-limit MEMORY_LIMIT]                                                       [--threads THREADS]                                                       [--resume] [--json]", kOptions_tape_depth_execution_confirm, 9},
    {L"tape-depth-prequential", L"usage: simple-ai-trading tape-depth-prequential [-h] [--symbols SYMBOLS]                                                 [--warehouse WAREHOUSE]                                                 [--cache-root CACHE_ROOT]                                                 [--output-dir OUTPUT_DIR]                                                 [--training-window-days TRAINING_WINDOW_DAYS]                                                 [--tuning-window-days TUNING_WINDOW_DAYS]                                                 [--calibration-window-days CALIBRATION_WINDOW_DAYS]                                                 [--evaluation-window-days EVALUATION_WINDOW_DAYS]                                                 [--horizon-seconds HORIZON_SECONDS]                                                 [--total-latency-ms TOTAL_LATENCY_MS]                                                 [--decision-cadence-seconds DECISION_CADENCE_SECONDS]                                                 [--maximum-depth-age-ms MAXIMUM_DEPTH_AGE_MS]                                                 [--maximum-rows MAXIMUM_ROWS]                                                 [--maximum-cached-rows MAXIMUM_CACHED_ROWS]                                                 [--no-dataset-cache]                                                 [--study-stage {development,screening,confirmation}]                                                 [--selection-lock SELECTION_LOCK]                                                 [--max-folds MAX_FOLDS]                                                 [--risk-level {conservative,regular,aggressive}]                                                 [--model-profile {regularized,balanced,expressive}]                                                 [--feature-set {core,tape_derived,cross_asset,full}]                                                 [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                                                 [--minimum-segment-rows MINIMUM_SEGMENT_ROWS]                                                 [--memory-limit MEMORY_LIMIT]                                                 [--threads THREADS]                                                 [--plan-only] [--resume]                                                 [--json]", kOptions_tape_depth_prequential, 28},
    {L"tape-depth-select", L"usage: simple-ai-trading tape-depth-select [-h] --report REPORT --design                                            DESIGN [--output OUTPUT] [--json]", kOptions_tape_depth_select, 4},
    {L"tape-depth-study", L"usage: simple-ai-trading tape-depth-study [-h] [--symbols SYMBOLS] --design                                           DESIGN [--warehouse WAREHOUSE]                                           [--cache-root CACHE_ROOT]                                           [--output-dir OUTPUT_DIR]                                           [--training-window-days TRAINING_WINDOW_DAYS]                                           [--tuning-window-days TUNING_WINDOW_DAYS]                                           [--calibration-window-days CALIBRATION_WINDOW_DAYS]                                           [--evaluation-window-days EVALUATION_WINDOW_DAYS]                                           [--total-latency-ms TOTAL_LATENCY_MS]                                           [--maximum-rows MAXIMUM_ROWS]                                           [--maximum-cached-rows MAXIMUM_CACHED_ROWS]                                           [--no-dataset-cache]                                           [--max-folds {4,6,8,10}]                                           [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                                           [--minimum-segment-rows MINIMUM_SEGMENT_ROWS]                                           [--memory-limit MEMORY_LIMIT]                                           [--threads THREADS] [--resume]                                           [--plan-only] [--json]", kOptions_tape_depth_study, 21},
    {L"tape-depth-train", L"usage: simple-ai-trading tape-depth-train [-h] [--symbol SYMBOL]                                           [--warehouse WAREHOUSE]                                           [--cache-root CACHE_ROOT]                                           [--output OUTPUT]                                           [--window-days WINDOW_DAYS]                                           [--end-date END_DATE]                                           [--horizon-seconds HORIZON_SECONDS]                                           [--total-latency-ms TOTAL_LATENCY_MS]                                           [--decision-cadence-seconds DECISION_CADENCE_SECONDS]                                           [--maximum-depth-age-ms MAXIMUM_DEPTH_AGE_MS]                                           [--risk-level {conservative,regular,aggressive}]                                           [--model-profile {regularized,balanced,expressive}]                                           [--feature-set {core,tape_derived,cross_asset,full}]                                           [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                                           [--minimum-segment-rows MINIMUM_SEGMENT_ROWS]                                           [--maximum-rows MAXIMUM_ROWS]                                           [--memory-limit MEMORY_LIMIT]                                           [--threads THREADS] [--json]", kOptions_tape_depth_train, 19},
    {L"tick-archive-sync", L"usage: simple-ai-trading tick-archive-sync [-h] [--symbols SYMBOLS]                                            [--data-types DATA_TYPES]                                            [--start-date START_DATE]                                            [--end-date END_DATE]                                            [--full-history] [--available-only]                                            [--plan-only]                                            [--plan-output PLAN_OUTPUT]                                            [--max-planned-gb MAX_PLANNED_GB]                                            [--warehouse WAREHOUSE]                                            [--cache-root CACHE_ROOT]                                            [--memory-limit MEMORY_LIMIT]                                            [--threads THREADS]                                            [--timeout TIMEOUT]                                            [--no-retain-archive] [--json]", kOptions_tick_archive_sync, 16},
    {L"tick-corpus-audit", L"usage: simple-ai-trading tick-corpus-audit [-h] [--symbols SYMBOLS]                                            [--data-types DATA_TYPES]                                            [--start-date START_DATE]                                            [--end-date END_DATE]                                            [--strict-book-depth-calendar]                                            [--warehouse WAREHOUSE]                                            [--cache-root CACHE_ROOT]                                            [--memory-limit MEMORY_LIMIT]                                            [--threads THREADS]                                            [--output OUTPUT] [--json]", kOptions_tick_corpus_audit, 11},
    {L"train", L"usage: simple-ai-trading train [-h] [--input INPUT] [--output OUTPUT]                                [--source {auto,file,db}] [--db DB]                                [--interval INTERVAL] [--market {spot,futures}]                                [--min-rows MIN_ROWS] [--download-missing]                                [--preset {balanced,custom,quick,thorough}]                                [--epochs EPOCHS]                                [--learning-rate LEARNING_RATE]                                [--l2-penalty L2_PENALTY] [--seed SEED]                                [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                                [--batch-size BATCH_SIZE] [--walk-forward]                                [--walk-forward-train WALK_FORWARD_TRAIN]                                [--walk-forward-test WALK_FORWARD_TEST]                                [--walk-forward-step WALK_FORWARD_STEP]                                [--calibrate-threshold]", kOptions_train, 20},
    {L"train-suite", L"usage: simple-ai-trading train-suite [-h] [--input INPUT]                                      [--output-dir OUTPUT_DIR]                                      [--symbol SYMBOL]                                      [--starting-cash STARTING_CASH]                                      [--objective OBJECTIVE]                                      [--max-workers MAX_WORKERS]                                      [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                                      [--batch-size BATCH_SIZE]                                      [--max-candidates MAX_CANDIDATES]", kOptions_train_suite, 9},
    {L"tune", L"usage: simple-ai-trading tune [-h] [--input INPUT] [--save-best]                               [--min-risk MIN_RISK] [--max-risk MAX_RISK]                               [--steps STEPS] [--min-leverage MIN_LEVERAGE]                               [--max-leverage MAX_LEVERAGE]                               [--min-threshold MIN_THRESHOLD]                               [--max-threshold MAX_THRESHOLD]                               [--min-take MIN_TAKE] [--max-take MAX_TAKE]                               [--min-stop MIN_STOP] [--max-stop MAX_STOP]                               [--compute-backend {cpu,cuda,rocm,directml,mps,auto}]                               [--batch-size BATCH_SIZE]                               [--lookback-days LOOKBACK_DAYS]                               [--from-date FROM_DATE] [--to-date TO_DATE]", kOptions_tune, 18},
    {L"universe", L"usage: simple-ai-trading universe [-h] [--symbols SYMBOLS] [--json]", kOptions_universe, 2},
};
inline constexpr int kCommandCount = static_cast<int>(sizeof(kCommands) / sizeof(kCommands[0]));

} // namespace simple_ai_trading::native_contract
