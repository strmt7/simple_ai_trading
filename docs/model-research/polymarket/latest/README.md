# Polymarket model status

> **Beta research software. No paper or live trading authority exists.**

![Held-out predictive metrics](charts/round14-held-out-metrics.svg)

Round 14's subsequent one-hour, after-cost BTC five-minute shadow **failed**:
12 events produced `-9.87720` quote net PnL, `0.455725` profit factor, and
`11.69847` maximum drawdown. This rejects the model for paper or live
promotion. The canonical
[economic evaluation](../evidence/round-014-btc-5m-shadow-hour-evaluation-v1.json)
overrides the earlier predictive-only result below.

![Round 14 cumulative after-cost shadow P&L](shadow/cumulative-pnl.svg)

The plotted values are preserved in the
[exact event-outcome table](shadow/event-outcomes.csv).

Round 14 tested a frozen BTC five-minute direction model on all 287 eligible
conditions from 2026-06-22 UTC. The shallow Binance-flow LightGBM challenger
recorded log loss `0.644667` versus
`0.691317` for the best control, a
`6.75%` relative skill.
Balanced accuracy was `0.6208` and the
paired 95% block-bootstrap improvement interval was
`[0.03544,
0.05865]`.

This is **predictive evidence only**. Polymarket spread, queue position, fills,
latency, fees, settlement, redemption, inventory risk, and PnL were not tested,
so it is not a profitability or execution claim.

Round 21 is a separate 30-day, target-blind BTC five-minute campaign. No model
or profitability result exists while that capture is active. Its deterministic
program evaluates five settlement-probability candidates and 81 after-cost
profile/scenario ledgers. The local AI comparison is also finite: exactly
`qwen3:8b`, `fin-r1:8b`, and `fino1:8b` must use the same matched development
population and deterministic baseline. A model is nominated for the one-use
sealed test only if all 81 ledgers pass; otherwise the auditable result is no
challenger. Nomination is not model promotion or trading authority.

Round 16 is a separate, preregistered BTC fifteen-minute comparison. Its
historical/live one-second feature transform is bit-identical, but no model
result exists yet. Before held-out access, its pretest artifact must freeze
train-only feature-support bounds and label-blind tune-only settlement-anomaly
thresholds. A future prospective scorer can activate only from caller-pinned
artifacts that pass every predictive gate; it has no execution authority.

The resumable workflow is intentionally phase-separated:
`python tools/run_polymarket_round16_screen.py status`, then `identities`,
`features`, `development-targets`, and `fit`. The one-use `test-targets` phase
requires `--acknowledge-one-use-test-access`; only then may `evaluate` and
`export` run. No command grants trading authority.

## Audit

- [Evaluation artifact](round-014-evaluation.json)
- [Sealed pretest artifact](round-014-pretest.json)
- [Candidate metrics](tables/round14-candidates.csv)
- [Every held-out decision](tables/round14-decisions.csv)
- [UTC condition series](tables/round14-conditions.csv)
- [Decision-offset metrics](tables/round14-decision-offsets.csv)
- [Cross-round progression](tables/optimization-progress.csv)
- [AI risk-model rejection record](ai-risk-models-rejected.json)
- [Round 21 AI candidate-selection contract](../round-021-ai-candidate-selection-design-v1.json)
- [Publication integrity](publication-integrity.json)

Regenerate from the closed local evidence database with
`python tools/publish_polymarket_round14_historical.py`.
