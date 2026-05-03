# Model and Signal Validation

Last local validation: 2026-05-03 on Windows 11 with DirectML GPU backend.

## Market Data

- Command: `fetch --limit 5000 --batch-size 1000`
- Result: 3,100 public BTCUSDC 15m candles saved locally
- Audit: 3,100 clean candles, 0 duplicates, 0 time gaps, 3,058 feature rows

## Model Runs

Baseline model on expanded data:

- Backtest: 35 trades, 65.71% win rate, +1.19 realized PnL, 0.06% max drawdown
- DirectML scoring backend: `privateuseone:0`

DirectML retrain candidate:

- Command: `train --preset thorough --compute-backend directml --batch-size 512 --walk-forward --calibrate-threshold`
- Training backend: DirectML `privateuseone:0`
- Backtest: 42 trades, 40.48% win rate, +1.45 realized PnL, 0.07% max drawdown
- Audit: model artifact accepted, threshold 0.290, validation rows 458

Conservative objective candidate:

- Source: `train-suite --objective conservative --compute-backend directml`
- Evaluation quality: `ok`, 0.599 validation accuracy, 0.501 F1
- Backtest: 9 trades, -0.14 realized PnL, 0.02% max drawdown

The DirectML retrain improved realized PnL over the baseline on the local expanded dataset, but it still trails buy-and-hold during a strongly rising sample and keeps the model-quality warning. Treat it as a safer incremental candidate, not proof of production profitability.

## Signal Harvest

Live signal run:

- Command: `signals --news-provider-limit 30 --news-items-per-provider 3 --provider-parallelism 12 --ollama-news --ollama-model gemma4:e4b --compute-backend directml --refresh`
- Result: 45 provider components, 42 fresh, telemetry written
- Horizons: short, medium, and long scores populated
- Slow providers: CryptoCompare, GDELT, and the dedicated Ollama news component reported warnings without blocking the harvest

Source grading:

- Command: `source-grades --window-hours 24 --ollama --ollama-model gemma4:e4b`
- Result: 274 source-grade rows written
- Ollama grading timed out at the configured 3s budget and the command completed with heuristic grading instead of blocking.
