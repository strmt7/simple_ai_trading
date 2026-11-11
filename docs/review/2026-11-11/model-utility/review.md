# Fixed-base exit impact before further model expansion

Baseline `d04b2880434c766b8a52aee69a26d3ed3800d0bd` held a fixed base
quantity through each simulated trade. Its backtest close and payoff-label
helper nevertheless sized exit impact using the original entry quote notional.
When the exit reference changed, the fill model priced a different order size
from the quantity actually used in cash accounting. The existing parity test
could not detect this because both implementations shared the same error.

The new independent controls failed all eight long/short and rising/falling
price cases against that baseline. Their supplied-law oracle computes:

`exit reference quote notional = abs(held base quantity) * exit reference price`

The entry fill determines the label's held quantity. Spread/impact then changes
the exit fill price, without resizing that base quantity. Realized PnL and exit
fees use the same quantity. This is an accounting/participation consistency
rule, not a statement about actual Binance impact or executable capacity.

For the direct constructed close, ten base units at an exit reference of 50
represent 500 quote units. At an exit reference of 200 they represent 2,000.
With the supplied one-million quote ADV and 100-bp impact coefficient, the
modeled impacts are about 2.236 and 4.472 bp. The baseline used about 3.162 bp
in both cases. It could overstate or understate costs; no universal performance
improvement or price-direction edge follows from the repair.

## Implementation and evidence

Backtests and payoff labels now use one finite positive quote-notional helper.
Signed quantities retain their absolute base magnitude. Zero, Boolean,
non-numeric, non-finite, overflowing and underflow-to-zero values reject. An
unrepresentable label is unavailable rather than a zero-cost training target.
The obsolete entry-notional argument was removed from the private close helper
and all three callers. Entry sizing, fee accounting, statistical guards,
features and model-selection objectives were not replaced.

New payoff-expert metadata uses `causal-adv-square-root-fixed-base-v3`.
No historical model or result was rewritten, retagged or retrained. Old weights
are not qualified for this changed execution law by being readable or carrying
a new filename. The version marker is provenance, not an implemented universal
artifact-admission gate; complete model/runtime compatibility enforcement remains
work. It must preserve independent owned-position closing when a model is invalid.

The [sealed evidence](evidence.json) binds exact source phases, the independent
control intent, code-scope comparison and retained logs. The final affected
backtest/execution/hybrid/training lane passed **247 cases with 10 skips**.
That lane includes the 24 focused cases; the counts are not additive. The skips
are the DirectML cases whose backend is unavailable in the selected environment.
No GPU/backend equivalence or performance benchmark is claimed. Formatting was
limited to changed definitions; targeted Ruff checks passed.

The source-bound semantic index was refreshed once with its existing storage
and model memoization before navigation. Actual query receipts and live-source
checks were retained. It describes the inspected pre-edit source phase, not a
current index after this patch. No other repository registration was changed.

## Financial and product boundaries

These are synthetic cost-model controls, not market observations, a new fit or
qualified net utility. The current generic payoff labels still model independent
entry/lifecycle paths; the separately supplied inventory-cash helpers are not a
fully qualified continuous hedged trainer. Native funding, financing, actual
fees, liquidity, paired quantities, partial execution, capital constraints and
feasible alternatives still need source-complete qualification. Existing ADV,
impact and missing-volume proxies remain conditional assumptions.

Further training requires causal net targets, full search provenance, untouched
chronological comparison and dependence/tail evidence. A better-sized execution
proxy does not supply those missing inputs. No production model fit, market or
account request, order, credential, protected-data read, service, recurring task
or release occurred. Financial counts remain **206 observations / 65 hypotheses
/ 37 historical scopes / 0 qualified stable edges**. The complete goal remains
active; this is not whole-repository or stable-product qualification.
