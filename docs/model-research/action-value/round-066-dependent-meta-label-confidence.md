# Round 66: Dependence-aware meta-label confidence

**Status:** implementation and contract validation only. No new economic backtest was run, so this round makes no profitability, ROI, drawdown, AI-uplift, testnet, or live-trading claim.

## Change

Positive average return and aggregate P&L are not enough to size a trade bucket when outcomes are serially dependent. Each selected take bucket, and each proposed downsize bucket, now requires at least 30 outcomes and receives a deterministic 95% moving-block-bootstrap interval over its time-ordered after-cost returns using 2,000 resamples. A bucket is executable only when its lower mean-return bound is strictly positive. Otherwise the policy remains observe-only or collapses the downsize threshold into the take threshold so weaker proposals are skipped. Thirty observations are a conservative governance floor, not a guarantee that an interval is perfectly calibrated.

The evidence is persisted as `meta-label-after-cost-v2`, carried through liquidity/session resizing, and bound into the shared autonomous decision. AI review requires the exact v2 bucket's sample floor, precision floor, positive mean return, positive aggregate P&L, resampling settings, and positive lower bound before local-model inference. Invalid evidence is vetoed without spending provider tokens. The deterministic executor also distinguishes a missing or user-disabled policy from an explicit observe-only result: observe-only skips the entry instead of silently reverting to full-size primary-model execution. Live readiness blocks observe-only, legacy v1, and malformed enabled policies, requiring a new provenance-bound training and promotion run. This rule cannot create an entry, relax a deterministic risk control, or authorize live trading.

The resampling implementation is shared with the AI-uplift gate to prevent statistical drift between model and AI governance. Its extraction preserved the prior AI-uplift algorithm and deterministic seed behavior. Meta-label resampling is bound to ordered timestamps, sides, and percentage returns, so changing quote-currency capital cannot alter a return interval. Threshold construction was also corrected: rounded score buckets now retain the lowest exact observed score, preventing upward decimal rounding from accidentally excluding every member of an equal-score bucket while retaining bounded candidate deduplication.

The threshold is selected on development evidence before its selected bucket is resampled. The interval therefore reduces dependence-blind false confidence but is not an untouched confirmation result and does not remove threshold-selection risk. Promotion still requires the existing sealed terminal and provenance gates.

## Verification

- 70 focused meta-label, shared-resampling, AI-assist, and model-readiness tests passed together.
- 128 model, hybrid-model, and autonomous-loop tests passed.
- 5 autonomous decision-construction tests and 2 focused backtest integration tests passed.
- The sealed-terminal training regression and generated native Windows/CLI contract parity test passed.
- Positive aggregate expectancy with a non-positive block-bootstrap lower bound remains observe-only.
- Both non-positive expectancy and non-positive bootstrap evidence are rejected before provider inference.
- Legacy and observe-only policies fail closed in both execution and model readiness.
- Ruff and `git diff --check` passed on the changed implementation before publication.

Existing performance graphs remain unchanged because no new market experiment was run.
