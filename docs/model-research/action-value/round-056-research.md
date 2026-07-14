# Round 56 research: paired action value and calibration

**Status:** frozen research rationale, development-only. This document does not claim profitability, AI uplift, testnet readiness, leverage readiness, or live-trading readiness.

## Why another architecture is justified

Round 55 rejected both treatments. Its post-run diagnostic reproduced the controller exactly and found that liquidity and volatility gates removed only 4 of 31 unanimous baseline votes in July-August 2024. The dominant attrition came from requiring every side-specific view to agree. Relaxing that rule after observing outcomes increased activity but did not establish an edge. High AI-score quintiles also inverted in September, and the selected actions were overwhelmingly long.

The next experiment therefore fixes three mechanisms without changing the source rows, payoff, costs, risk budget, or development period:

| Defect | Round 56 control |
|---|---|
| Separate long and short towers can learn unrelated scales and priors | One shared model receives paired, action-aligned long and short rows |
| One January-March 2024 early-stop window can select a transient regime | Twelve expanding rolling-origin folds determine model length and out-of-fold diagnostics |
| Raw tree scores were treated as comparable across views | Point forecasts receive held-forward isotonic calibration; lower-tail forecasts receive held-forward coverage adjustment |

## Frozen modeling decision

Each decision state produces two rows: `action_sign=+1` for long and `action_sign=-1` for short. Signed market features are expressed from the candidate action's perspective. For example, positive momentum is favorable to a long row and unfavorable to the paired short row. Downside and upside semivolatility become action-favorable and action-adverse semivolatility. Invariant liquidity, volatility, path-efficiency, calendar, and symbol features are unchanged.

This representation is not an instruction to trade both sides. It makes the comparison identifiable inside one fitted function and removes independent score scales. `action_sign` remains available to the numerical model so genuine residual long/short asymmetry can be learned, but it is hidden from language-model factor generation to prevent a factor from encoding a naked side preference.

Two views are retained: exact after-stress-cost payoff in basis points and the same payoff divided by the causal stop width. Each view fits:

- an L2 conditional-mean LightGBM model;
- a 20th-percentile LightGBM quantile model;
- three fixed seeds using the existing OpenCL-compatible tree budget.

L2 replaces Round 55's L1 objective because an L1 regression estimates a conditional median, while order eligibility depends on positive **expected** after-cost payoff. The quantile model is separate and can only reduce size; it cannot create an entry.

## Chronology and calibration

Out-of-fold predictions cover July 2023 through June 2024. For each outer month, training ends two calendar months earlier, the immediately preceding month is used only for early stopping, and a 61-minute label embargo separates roles. The rounded median best iteration across all twelve folds is then used for a final January 2022-June 2024 refit.

July-December 2023 out-of-fold predictions fit the point and quantile calibration objects. January-June 2024 evaluates those objects without refitting. If all predictive gates pass, calibration is refit on all twelve out-of-fold months before the unchanged July-August 2024 policy-development and September 2024 consumed-holdout replay.

The quantile adjustment is explicitly empirical. No exchangeability-based conformal guarantee is claimed for serially dependent, non-stationary market data. Rolling-origin evaluation and a label embargo are used because multiple test origins are more informative than one split, while dependent observations require separation between fit and evaluation roles.

## Decision and risk logic

For each action, seed predictions are reduced by their median, calibrated view forecasts are averaged with equal weights, and the action with the larger calibrated expected stress payoff is eligible only when:

- its calibrated expected payoff is strictly positive after the frozen 16 bps round-trip stress charge;
- it exceeds the opposite action's score;
- at least four of six raw point forecasts are positive;
- the unchanged causal liquidity and volatility-shock gates pass.

There is no optimized score threshold and no minimum trade quota. The conservative controller remains fixed at 1x, fixed initial capital, no profit reinvestment, 0.10% stop risk per position, and 0.15% aggregate open stop risk. Leverage cannot manufacture predictive edge and remains prohibited until a later, separately frozen unlevered confirmation passes.

## Predictive falsification

The architecture is rejected before economic interpretation unless held-forward January-June 2024 evidence beats causal constant baselines on point MSE and quantile pinball loss, has positive pooled rank association, orders the top score quintile above the bottom quintile, and achieves the frozen lower-tail coverage band. Side-specific diagnostics must pass too. A side-preference above 90% is reported as a structural warning, never repaired by forcing trades.

Development and September economic gates remain at least as strict as Round 55. The baseline and AI-augmented treatments use identical rows, seeds, costs, calibration, controller, and replay. AI uplift requires both treatments to pass independently plus a familywise paired block-bootstrap improvement. Narrative review cannot override a failed gate.

## AI boundary

The governed AI candidates are `qwen3.5:9b`, `fin-r1:8b`, and `fino1:8b`. Each receives only feature names, semantics, allowed operators, and an action-conditioned output schema. It never sees market values, timestamps, labels, returns, or trade outcomes. Every syntactically valid, non-duplicate program is included without return-based selection; invalid output fails closed.

This uses language models for auditable factor-program hypotheses, not numerical forecasting or order authority. Recent financial benchmarks still report material limitations in numerical and trading-signal reasoning, so an LLM result is not presumed to improve the baseline. The matched ablation must demonstrate that improvement.

## Primary research basis

- [LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html): L2 and quantile objectives, compact bins, and GPU/OpenCL controls.
- [LightGBM GPU device targeting](https://lightgbm.readthedocs.io/en/latest/GPU-Targets.html): explicit platform/device verification is required so an OpenCL CPU target cannot be mistaken for a GPU.
- [Tashman, out-of-sample forecasting tests](https://doi.org/10.1016/S0169-2070(00)00065-0): motivates rolling origins, coefficient updates, and multiple test periods.
- [Racine, hv-block cross-validation](https://doi.org/10.1016/S0304-4076(00)00030-0): motivates separation around dependent validation observations; Round 56 uses a stricter chronological embargo rather than claiming iid folds.
- [Niculescu-Mizil and Caruana, probability calibration](https://doi.org/10.1145/1102351.1102430): boosted-tree outputs can require held-out monotone calibration.
- [Gneiting and Raftery, proper scoring rules](https://doi.org/10.1198/016214506000001437): supports evaluating probabilistic forecasts with proper losses rather than accuracy narratives.
- [Conformalized quantile regression](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html) and [adaptive conformal inference under shift](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html): motivate lower-tail calibration while also making clear why ordinary exchangeability guarantees are not asserted here.
- [Bates and Granger, forecast combination](https://doi.org/10.1057/jors.1969.103): supports combining distinct forecasts; equal weights avoid estimating another policy from the consumed development period.
- [Probability of Backtest Overfitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253): requires complete trial accounting and prohibits converting repeated development results into confirmation claims.
- [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B) and [official Ollama package](https://ollama.com/library/qwen3.5): identify the locally feasible general reasoning candidate and runtime artifact.
- [FinTradeBench](https://arxiv.org/abs/2603.19225): reports persistent weaknesses in numerical and time-series trading-signal reasoning, supporting the strict non-authority boundary for language models.

## Resolution boundary

Round 56 remains an hourly decision experiment over complete one-minute execution paths. It does not pretend that the available multi-year data contains continuous second-level order books. If this architecture survives, a later frozen stage may lower the decision interval using multi-year minute bars and a separately measured tick-level execution lane. Mixing that resolution change into this round would make the cause of any result unidentifiable.
