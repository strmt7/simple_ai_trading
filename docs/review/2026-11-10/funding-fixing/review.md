# Partial funding information and total hedged cash

This source/conditional-mathematics review changes the next model question:
predict incremental executable hedged cash, rather than treating a nearly fixed
funding coupon as profit. The [evidence](evidence.json) binds the source phases
and [constructed controls](controls.json), reproduced by [this script](controls.py).
No new market population, model fit or qualified edge follows.

## Source findings

The retained [Clearing Procedures](https://bin.bnbstatic.com/static/cms/cg08ou2ak0tn7mcplvfg/file/53197b612332da02c20b5b7d19b81ff53ee5f4938c6330c72a30a1ca4f91049f.pdf)
was read through its existing complete native extraction, without downloading
again. Visual review of PDF pages 25–27 confirms five-second premiums, increasing
sample weights above one hour, equal weights at one hour, a cap, mark-based
payments and operational tolerance up to 15 seconds. Caps and intervals can
change. That tolerance does not establish this bot's eligibility cutoff.

The formula labels `N` in hours, while its table defines `N` in seconds. Inner
clamp endpoints are printed positive-first, unlike the outer ordered clamp.
Weighting prose describes remaining time while explicit coefficients increase
from the first sample. These distinctions are visible, not extraction errors.
Effective version, current product/account scope, interest basis and native
implementation remain unqualified. Do not silently normalize these into an
adapter or rewrite old settled-rate results.

A distinct formula-change notice was frozen and read once. Native navigation
reported HTTP 202, success and one Markdown character. The retained 4,471-byte
native result admits no source facts; its size is not document wire bytes.
There was no alias/follow-up request. The earlier FAQ citation lacked a retained
full source body and was not recaptured. URL inventory covered 2,567 files and
skipped 33 oversized published artifacts; no universal novelty claim follows.

## Exact conditional result

With a complete fixed schedule, observed raw weighted contribution `A`, remaining
weight `W` and valid remaining-premium lower bound `p_min`, the final average is
at least `A + W*p_min`. Declare the mathematical law explicitly:

`F(P) = clamp((P + clamp(I-P, -d, d))*hours/8, -cap, cap)`.

The premium-to-rate map increases outside the damper and is constant inside it.
Positive scaling and ordered outer clamping preserve monotonicity. Applying it
to the average lower bound therefore bounds the rate. This does not reconcile
the document's notation or units. A rolling estimate alone is not `A`.

Under the printed premium-price expression, positive index and nonnegative impact
prices imply `P >= -1`: its negative numerator term cannot exceed the index.
Raw coverage, weights, support and clock applicability still need qualification.

For a constructed eight-hour, 5,760-sample increasing-weight schedule with zero
known raw premiums, `I=0.01%`, `d=0.05%` and supplied cap `0.75%`, one unknown
final sample leaves weight `2/5761` and rate floor `0.01%`. Two unknown samples
allow a negative floor. This shows useful conditional information, not a current
signal, feasible last-second entry or minimum cash: mark, quantity, entitlement
and rounding remain necessary. Unknown entitlement still encloses zero payment.
In a separate constructed four-hour case, literal seconds substitution changes
the capped rate from `0.025%` to `0.75%`; unit reconciliation is consequential.

## Model consequence

For matched fixed-base long spot/short linear perpetual positions in one quote
currency, define `B = perpetual price - spot price`. Total hedged cash before
remaining costs is `q*(B_entry-B_exit) + funding_cash`. Equal base quantities
remove the common price component, not basis changes. Execution prices,
commissions, financing, reserves, margin and opportunity cost remain necessary;
do not charge price slippage twice.

The existing exact cash core supplies a constructed funding credit of `0.01`
quote units for one base unit. With spot fixed at 100 and perpetual entry 100,
exits at 100, 100.01 and 100.02 give pre-cost cash `+0.01`, `0` and `-0.01`.
A supplied 0.02 total cost makes all three negative. This rejects coupon-only
profit inference, not the entire family's expected value.

For existing inventory compare continuing with executable closing now at equal
initial wealth; shared sunk costs cancel. For fresh positions include complete
entry/exit cash. A future funding-aware model must improve incremental net
utility with causal raw information, chronological evaluation and capital/tail
constraints. Coupon sign accuracy alone is not that target.

No production predictor was added before input qualification. Thirteen exact
constructed checks and Ruff pass; no unchanged application suite, fit or GPU
benchmark was repeated. Protected partials, original results, financial census
and consumed market windows remain unchanged. Next require applicable formula
units, raw sample/clock coverage and total paired execution cash before selecting
or fitting a new economic population.

The capture, rendering and control helpers terminated. Automatic approval review
rejected removal of three owned cache files and empty scratch directories with
only the reason `blocked by policy`. They remain inert; no alternate deletion
was attempted. Original raw evidence and the rejection receipt remain private.
