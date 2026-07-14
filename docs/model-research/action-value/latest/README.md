# Round 55: Stop-Bounded Payoff Models

> **Rejected development evidence.** No profitability, untouched-confirmation, AI-uplift, leverage, testnet, live-trading, or promotion claim is made.

Round 55 trained `18` OpenCL LightGBM artifacts (`36` side models) on BTCUSDT, ETHUSDT, and SOLUSDT. Targets used real one-minute Binance futures paths, exact gap-through stops, settled funding, and a `16 bps` round-trip stress charge. Every position stopped or timed out within `60 minutes`; notional used fixed initial capital with no reinvestment and no leverage.

| Treatment | Period | Trades | Stress return | Max drawdown | Profit factor |
|---|---:|---:|---:|---:|---:|
| Baseline | Jul-Aug 2024 | 27 | +0.2447% | 0.1806% | 1.733 |
| 8B AI factors | Jul-Aug 2024 | 14 | +0.3659% | 0.0253% | 11.515 |
| Baseline | Sep 2024 | 7 | +0.0900% | 0.1645% | 1.547 |
| 8B AI factors | Sep 2024 | 5 | +0.0559% | 0.1645% | 1.340 |

Both treatments failed six frozen gates: development and September trade/day counts, September P&L concentration, and the familywise block-bootstrap lower bound. The seven Fino1/Qwen3 factor programs improved July-August descriptively but reduced September stress return by `-0.0341%`; the paired lower bound was `-0.01424 bps/hour`. AI uplift therefore failed.

The run read `24,096` hourly timestamps and `72,288` symbol paths through September 2024. It generated no synthetic rows and did not load the `6,551` excluded October 2024-June 2025 timestamps. A future interval remains untouched, but Round 55 authorized no access to it.

## Frozen Attrition Diagnostic

The separately frozen diagnostic exactly reproduced the control without refitting a model or trying a threshold. The baseline had `201` July-August symbol-hours where at least one view voted, `70` where two views agreed, and only `31` where all three agreed; market-state gates then removed just `4`. Relaxed baseline controllers lost money in both consumed periods. The pooled-nine AI diagnostic returned +0.2056% over 26 July-August trades and +0.1100% over 8 September trades, while its matched baselines returned -0.2094% and -0.1134%. Those are post-hoc, sparse, non-monotonic score diagnostics on consumed data, not AI-uplift or profitability evidence.

## Evidence

| View | Graph | Source |
|---|---|---|
| Model skill | [SVG](charts/model-skill.svg) | [CSV](models.csv) |
| Path economics | [SVG](charts/economics.svg) | [CSV](treatments.csv) |
| Trading activity | [SVG](charts/activity.svg) | [CSV](treatments.csv) |
| Matched AI effect | [SVG](charts/ai-uplift.svg) | [CSV](treatments.csv) |
| September equity | [SVG](charts/september-equity.svg) | [CSV](equity.csv) |
| Controller attrition | [SVG](charts/controller-attrition.svg) | [CSV](controller-attrition.csv) |
| Diagnostic economics | [SVG](charts/controller-economics.svg) | [CSV](controller-economics.csv) |
| Score calibration | [SVG](charts/controller-score-calibration.svg) | [CSV](controller-score-calibration.csv) |
| Round progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`trades.csv`, `hourly-ledger.csv`, `monthly-economics.csv`, `predictive-rank.csv`, `ai-factors.csv`, `gates.csv`, `controller-symbol-economics.csv`, `controller-overlap.csv`, `controller-vote-patterns.csv`, `controller-trades.csv`, `screen.json`, and `controller-diagnostic-report.json` preserve the underlying evidence. Every chart is regenerated from tracked tabular data.
