# Incumbent inventory cash versus immediate flattening

The [frozen offline question](inventory-action-intent.json) asks whether an
existing position can be held, resized or reversed without treating every hour
as a newly opened round trip. The implementation reuses the shared funding and
inventory ledger, rather than introducing another funding-payment law.
The [generated exact controls](inventory-action-controls.json) bind source bytes,
state, quantities, clocks, costs and funding. They are synthetic algebra, not
observed market returns, training evidence or a qualified edge.

## Implemented behavior

`InventoryCashPath` optionally declares initial signed base inventory and explicit
target quantities. Existing callers still start flat and retain their fixed-base
holding behavior. A transition trades `abs(target - incumbent)` units at the
supplied entry price. Held inventory does not rebalance implicitly when prices
change. Every terminal position liquidates at the final supplied price; sunk
pre-path cash is excluded.

`evaluate_inventory_cash_actions` evaluates a single-interval action set including
a mandatory immediate-flatten comparator. It retains each action's own cash:

```text
surplus lower = action cash lower - immediate-flatten cash upper
surplus upper = action cash upper - immediate-flatten cash lower
```

The comparator against itself is exactly zero. Independent uncertainty bounds
are conservative enclosures, not jointly attainable extrema. Side-specific
debits and common costs are not removed through a signed contrast. Basis points
use the explicit reference quote budget, not inferred incumbent notional or
compounded equity.

The supplied boundary model is a monotone net quantity transition. Funding is
linear in quantity, so old/new endpoint payments enclose partial quantities.
Same-sign resizing does not pass through zero. An unchanged incumbent is held
at entry, unlike an uncertain fresh opening; terminal liquidation still encloses
held-to-flat entitlement. This does not prove actual fills or native entitlement.

State time must be no later than decision, and decision strictly precedes entry.
A state-source digest is mandatory. These declarations do not prove native
origin or causal feature availability. Malformed quantities, conflicting signs,
duplicate actions, missing flatten comparators, future state and nonpositive
costs reject. No inference/order path consumes realized action outcomes.

## Exact controls and verification

At unchanged price 100, quantity one and modeled one-way cost 6 bps, holding an
existing unit costs 0.06 quote units at terminal liquidation. Flattening now also
costs 0.06: holding surplus is zero. Opening a new unit costs 0.12 across entry
and exit. This is an accounting distinction, not a profitable strategy.

When supplied price changes from 100 to 200, an incumbent unit remains one unit.
Holding surplus is 99.94 quote units. Resizing to half gives 49.97; reversing to
minus one trades two units at entry and gives -100.18. These constructed
directional changes are not neutral hedges, forecasts or observed opportunities.
A positive entry-funding control preserves incumbent debits and the flatten
comparator's ordering uncertainty; same-sign resizing cannot invent zero cash.

378 affected checks passed, including 117 new action/quantity checks. They cover
75 incumbent/target/settlement-clock combinations with partial quantities,
costs, both signs, malformed inputs, bindings, legacy-trainer rejection and
existing stateful/funding routes. Three additional artifact checks verify source
hashes, reproduction, exclusive creation and overwrite rejection. Ruff passed.
No full trading suite, account integration, fit, GPU benchmark, network capture
or skill-efficacy test is claimed. One documentation patch failed its context
check without applying changes; it was corrected without rerunning controls.

Reproduce only when source changes invalidate the receipt, with a new output:

```powershell
uv run python -m tools.reproduce_inventory_cash_actions --output <new-controls.json>
uv run python -m pytest tests/test_inventory_cash_actions.py tests/test_inventory_action_control_artifact.py tests/test_stateful_inventory_cash.py tests/test_stateful_fixed_base_cash_counterexample.py tests/test_stateful_cash_labels.py tests/test_stateful_turnover_model.py tests/test_funding_cash.py tests/test_funding_cash_labels.py tests/test_funding_cash_remaining_routes.py --tb=short -x
```

## Still required for the real objective

This is a transition-aware horizon-liquidation primitive, not a continuous
Bellman objective, qualified dataset, forecasting policy or neutral hedge. The
legacy signed trainer rejects this table before model-directory creation.
Independent hourly labels and historical results are unchanged. Origin/coverage,
paired hedge quantities, complete fee/spread/latency/orphan costs, basis,
financing, margin and causal roles must qualify before fits.

Next qualify a complete paired hedge/cash population and connect state-conditional
two-sided surplus forecasts to an explicit continuation/abstention objective.
Do not fit on constructed controls or rerun known losing capture families to
manufacture progress. Counts stay 202 observations / 65 hypotheses / 37 scoped
mechanisms / zero qualified stable profitable edges. The overall goal is active.

Fresh GitHub contributor inspection still exposes old shared AI-identity
violations, including `agent@local.invalid` and hyphenated `AI-agent`. They are
surfaced, not rewritten. This checkpoint must use `AI agent <>` for both author
and committer; no shared-history rewrite is authorized.
