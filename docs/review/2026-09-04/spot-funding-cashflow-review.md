# Spot/perpetual carry: actual cash-flow weighting and cost headroom

One zero-network accounting audit reused all 17 original pairs and 51 original
training/validation/test roles. The [result](spot-funding-cashflow-review.json)
binds the original result/contract, every funding response, the
[audit plan](spot-funding-cashflow-plan.json), and the new implementation.
The durable journal records one completed offline audit. None of the original
results, implementations, holding periods or selection gates was changed.

This is a review of the existing terminal family
`binance_broad_crypto_current_liquidity_selected_spot_perpetual_funding_carry_preflight`,
not a new accepted hypothesis. Its canonical failure remains valid under its
frozen contract. Reused history is exploratory, not fresh out-of-sample evidence.

## What was corrected in the analysis

The old funding-only screen sums rates and subtracts a 32-bp execution stress
plus 10% annual opportunity cost on each of two reference capital legs.
That capital hurdle is a scenario, not paid exchange commission or measured
account financing. Failing it does not alone establish a negative cash profit.

For a fixed one-base-unit short, settlement funding is instead
`sum(rate_t * mark_t)` in quote units. This audit normalizes that cash by the
immediately preceding retained funding mark before each original role, then
deducts the unchanged 32-bp stress. That reference is **not an executable spot
entry**. It is only a consistent scaling unit. The original role duration and
two-leg capital stress are retained separately.

For after-execution normalized funding `A` bps, duration `d` days and two
reference capital legs, the annual cost per leg that exactly exhausts this
limited cash-flow headroom is `A * 365 / (2*d)` bps. This is neither an expected
APR nor a financing quote. Missing entry/exit basis and operational risks can
erase positive funding. The original bootstrap, regime and concentration
gates were not recalculated or treated as passed.

## What the retained numbers say

Rounded bps after mark weighting and the 32-bp execution sensitivity, before
capital cost and omitted economic terms:

| Pair | Original training period | Original validation period | Original test period |
| --- | ---: | ---: | ---: |
| BTCUSDT | -20.31 | 20.36 | 34.45 |
| ETHUSDT | -15.74 | -0.88 | 24.50 |
| SOLUSDT | -90.46 | -15.58 | 18.18 |

Thus the majors' lack of cross-period consistency is not merely an artifact
of the 20%-of-reference-notional annual capital stress. Every major still has
a negative role after execution stress alone. Conversely, calling the BTC
test role a funding cash loss would be inaccurate: its funding cash and
after-execution subtotal are positive, although its stressed net is negative.

Only BNBUSDT, PYTHUSDT and SUIUSDT have positive after-execution subtotals in
all three original roles. Their **weakest-role** annual break-even capital
costs per reference leg are approximately 45.58, 49.21 and 21.36 bps
(0.456%, 0.492%, 0.214%). These are descriptive extrema over consumed history,
not prospective symbol selection, statistical lower bounds or a qualified
carry portfolio. The complete 17-pair/51-role table remains in the JSON;
no losing pair or period was dropped.

## Decision and next work

This evidence does not justify a new price request, threshold optimization,
larger model, GPU training run or promotion. The current-liquidity-selected
population is already consumed, economically incomplete and selection-biased.
All exact retry gates remain unchanged. No accepted-edge count changed.

Future eligible carry work must use integrated settlement cash flows and
measured feasible alternative capital returns, with explicit stress scenarios
kept separate. Where coin collateral or portfolio offsets change capital,
compare equal risk/liquidity budgets and include every incremental cost; do
not simply shrink a denominator. A causal abstention rule needs a separately
frozen independent confirmation population and complete basis/execution data,
not filtering the losing periods shown here.

Scope reviewed: `_observations`, `_split_roles`, `_role_metrics`,
`_bootstrap_lower_bound`, related regime calculations and transport portions
of the original broad-crypto runner; its entire CLI/collection tail was not
semantically reviewed. The new offline evaluator and tests were read fully.
Fifteen focused tests and Ruff pass. No broad CI, bootstrap sweep, new dataset
download, account access or benchmark was needed.

## Published-source reconstruction

The pre-push check found that the original 17 funding-response files were
local-only under ignored `data/`. Their exact 1,037,711 bytes are now copied to
`spot-funding-sources/`, preserving each original basename and SHA-256; no
original file was modified. The audit's original source paths and hashes stay
unchanged. Only this explicit public-source prefix maps to the publication
sidecars; there is no fallback to local cache or the network.

Run `uv run --locked python -m tools.verify_spot_funding_cashflow_publication`
to reconstruct all 51 published role calculations. This deterministic
integrity check is not a new statistical test or replay of the consumed
acceptance experiment. The source-only publication exception is documented
in the data provenance policy for this review session.
