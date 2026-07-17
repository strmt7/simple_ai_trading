# Round 67: Purged chronological meta-label validation

**Status:** implementation and contract validation only. No new market backtest was run, so this round makes no profitability, ROI, drawdown, AI-uplift, testnet, or live-trading claim. Existing performance tables and graphs are unchanged.

## Problem

Round 66 measured dependence-aware confidence, but threshold selection and confidence evaluation still used the same development outcomes. That can reward selection luck. A profitable selected bucket is not later-period evidence merely because its serial dependence was resampled.

## Change

`meta-label-after-cost-v3` creates a chronological policy experiment inside the development trade log:

1. Target the earliest 60% for calibration and the latest 40% for validation, with at least 30 outcomes in each partition.
2. Keep every trade sharing the boundary opening timestamp in validation.
3. Purge any earlier trade whose close timestamp reaches or crosses the first validation opening timestamp.
4. Search and rank take thresholds only on calibration outcomes.
5. Apply the frozen threshold to validation outcomes and require the validation take bucket to clear support, objective precision, positive mean after-cost return, positive aggregate after-cost P&L, and a strictly positive 95% moving-block-bootstrap lower mean-return bound from 2,000 resamples.
6. Evaluate the optional downsize bucket only on that same later validation partition. Without its own positive evidence it collapses into skip.

The policy stores source, calibration, purge, and validation counts; boundary timestamps; deterministic SHA-256 bindings for every partition; and separate calibration-selection and validation-action metrics. An executable policy is rejected if take/downsize counts exceed or cannot coexist inside its validation partition. V2 artifacts require retraining and fail closed in execution and model readiness.

This design follows the causal ordering of time-series validation and the financial-ML principle that outcomes spanning an evaluation boundary must not remain in the selection set. It also addresses the strategy-selection risk described by Bailey et al.'s [Probability of Backtest Overfitting](https://papers.ssrn.com/sol3/Papers.cfm?abstract_id=2326253). Scikit-learn's [time-series split contract](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) independently documents why ordinary shuffled validation is inappropriate for time-ordered observations and exposes an explicit train/test gap. These sources motivate the control; they do not validate this strategy or prove profitability.

The split remains internal development evidence. The underlying predictor may already have used other development rows, and the later policy partition is not a substitute for the separately fingerprinted, one-shot terminal replay. AI can review only an action emitted by the validated v3 deterministic policy, and a rejected policy consumes no provider tokens. AI still cannot create entries, weaken deterministic risk controls, or authorize live trading.

## Verification

- A calibration-profitable threshold with later negative expectancy remains observe-only.
- Positions overlapping the validation boundary are removed from calibration and recorded as purged.
- Equal boundary opening timestamps cannot be divided across partitions.
- Internally impossible take/downsize support counts fail closed.
- V2 and malformed policies fail closed; v3 action evidence reaches backtest, autonomous, AI-review, and model-readiness consumers.
- 127 focused meta-label, backtest, AI-assist, and model-readiness tests pass together.
- 128 shared model, hybrid-model, and autonomous-loop tests pass.
- All 72 training-suite tests pass, including the sealed-terminal reservation regression.
- Seven autonomous CLI decision-construction tests and the generated Windows/CLI contract parity test pass.
- Ruff and `git diff --check` pass on the changed implementation.

No economic graph was regenerated because doing so without a new provenance-bound market experiment would misrepresent implementation tests as financial evidence.
