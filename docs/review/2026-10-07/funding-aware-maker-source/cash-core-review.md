# Funding cash core: implemented amount law, pending label integration

`funding_cash.py` implements exact rational, fixed-base USD-M funding cash for
supplied BTC/ETH/SOL USDT events. It parses hash-matched retained funding-history
bytes, preserves decimal rates/settlement marks, rejects ambiguous schemas and
reconciles the exact ordered timestamp/rate population against a separately
certified list. No network, warehouse mutation, credentials or Torch dependency.
[Structured source/verification binding](cash-core-review.json).

This is a real cash-law implementation, not the complete label repair. No
existing builder calls it yet. Hashes prove byte identity, not API origin;
alignment proves equality to the supplied expected list, not that list's
coverage. Empty pages are not proof of no payments. Unknown extra response
fields/types fail closed rather than silently adopt changed contract semantics.
Only regular or unreported rate types are parsed; absence is retained as
unreported, not asserted Regular. Special/dividend and inverse products are
unsupported. Parser limits are permanent documented endpoint/resource bounds,
not temporary campaign logic.

For known holding, signed cash is `-base_quantity * settlement_mark * rate`.
Entry-relative bips use absolute base quantity times entry price, not margin or
a constant-notional rate sum. With unknown entitlement, each event encloses zero
and its payment: an uncertain debit remains possible while an uncertain credit
is not guaranteed. Opposite rates cannot cancel away entitlement uncertainty.
The bound is conservative, not necessarily jointly attainable. Zero-rate cash
can be exact while entitlement is still unknown. Supplied holding flags do not
prove execution timing. This does not include venue rounding, fees, financing,
FX, missing events or all-account cash reconciliation.

## Official rule/schema check

Accessed October 7, 2026; official primary sources, not third-party mirrors:

- [Binance funding FAQ](https://www.binance.com/en/support/faq/detail/360033525031)
  directly supports position-value-times-rate and mark-based position value.
  Its current wording ties payments to open positions at specified times. That
  does not reconstruct historical fills, intrabar exit order or all prior
  jurisdiction/date-specific settlement implementations.
- [USD-M funding-history API](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#get-funding-rate-history)
  directly defines `fundingTime`, decimal rate and charge-associated `markPrice`,
  ascending order and a 1,000-row page maximum. Current `rateType` distinguishes
  Regular and Special. The old official route redirects to this new catalog;
  this was observed, not a guessed market API retry.

These web document reads are not new market observations or retained one-use
price captures. No API page, real mark population or authoritative old-period
settlement-clock support was acquired in this phase. Source-byte captures and
origin/coverage qualification remain prerequisites for actual label admission.

## Expanded consumer scope

In addition to the original hurdle and barrier finding, source inspection found:

- `stop_time_payoff_data._funding_in_holding_window` also sums raw rates and
  assigns stop exits to minute end.
- `second_flow_execution_model._funding_bps` sums raw rates but excludes the exit
  timestamp, unlike the other helpers. That convention is not proof of actual
  entitlement. Its timing-label price return also uses entry-price normalization.
- `stateful_turnover_model.build_stateful_hourly_dataset` calls the hurdle
  helper for hourly targets consumed by its replay. Repair must qualify actual
  quantity/rebalancing semantics, not mechanically apply independent hourly
  funding returns to a held-position ledger.

These ranges were inspected, not all lines of those model files. All five legacy
consumers and warehouse/state sources remain numerically unchanged. Their old
results are preserved as proxies; new core tests do not qualify those results.

## Focused verification and next work

66 new core cases plus nine existing barrier/stop-time/AI-veto cases: **75 passed**
in 2.63 seconds in the locked lightweight environment. Covers signed quantity,
signed rates, mark scaling and near-zero sign, exact decimal units, all supported
symbols, unknown/absent entitlement, cancellation uncertainty, malformed and
tampered responses, parser ceilings and exact population alignment failures.
Opposite rates at unequal settlement marks do not falsely cancel fixed-base cash.
Synthetic cases are unit controls, not market performance. Ruff passed. No
full-suite/coverage/GPU benchmark, fit, AI request or uplift claim.

Next: validate complete mark-page provenance against the certified rate window,
bind supported settlement clocks and intrabar exit bounds, integrate all five
consumers with distinct cash-label/provenance identities, and assert missing or
contradictory evidence blocks admission. Preserve original sources/results;
then freeze the smallest chronological economic comparison before retraining.
Do not repeat these checks without changed behavior or source bindings.
No research count, acceptance gate or protected-capture boundary changed.
