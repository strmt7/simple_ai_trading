# Inverse collateral: accounting lead, not a qualified edge

This is a conditional mathematical review, not an exchange-mechanics
certification or a rerun of the failed linear/inverse funding screen.
Three exact official FAQ requests redirected to a regional landing page.
The [source plan](inverse-collateral-source-plan.json) and complete
[tool extraction](inverse-collateral-source-extraction.json) retain that
failure. Extraction is not raw origin HTTP bytes. Search snippets are discovery
only. No locale retry, market data, protected capture or account request followed.

## What the conditional identity establishes

Let `N` be fixed quote notional (integer contract count times contract unit),
`F0` the executed inverse short entry price, `M` its current mark, `S` the
executable coin-to-quote conversion price, `q` the initial coin collateral and
`C` net coin cash flows after that initial balance. Under the hypothesized
inverse short payoff, coin equity is

`E = q + N*(1/M - 1/F0) + C = N/M + (q - N/F0 + C)`.

Thus quote equity is `N*S/M + S*(q - N/F0 + C)`. Along the **conditional**
common-price path `S=M`, it is `N + S*r`, where `r=q-N/F0+C`.

- With exactly `q=N/F0` and `C=0`, equity stays `N`. The coin collateral is
  already the spot inventory; adding its value a second time is double counting.
- A negative coin debit leaves negative residual direction exposure. A small
  fee alone defeats an unconditional all-price solvency claim if it is not
  funded separately. Under illustrative proportional maintenance `m*N/M`,
  surplus is `(1-m)*N/M+r`; when `r<0`, it reaches zero at
  `M=(1-m)*N/(-r)`. This is **not Binance's liquidation formula**.
- Positive funding retained in coin leaves positive residual exposure. Cash
  credited at funding mark `Mt` under an assumed `N*rate/Mt` coin-payment rule
  has later quote value `N*rate*S/Mt`, not generally `N*rate`.
- Converting a positive receipt immediately at exactly `S=Mt` would realize
  `N*rate` before conversion costs and remove that residual coin exposure.
  Negative payments require separate replenishment. This operation is not
  modeled as free, executable, atomic, or authorized.
- Discrete contracts, coin precision, extra collateral, entry fees and other
  positions leave residual exposure. `S/M` retains spot/mark basis risk even
  when the residual is zero. Exact sizing is a target, not a guaranteed fill.

For a dated inverse contract with settlement `Fd`, matched initial coin
purchase costs `(N/F0)*S0`, and its eventual liquidation value is `N*Sexit/Fd`
before fees. Only under the additional exact-convergence condition
`Sexit=Fd` does gross carry equal `N*(1-S0/F0)`. Coin acquisition, futures
execution, delivery fees, conversion, settlement delay and capital costs all
remain. Perpetual funding is variable and cannot be substituted for locked
dated carry.

## What changes in our research

`tools/review_inverse_collateral.py` implements this small, immutable-input,
Decimal accounting model **outside production execution code**. It never
selects a strategy or reports eligibility. Its 12 synthetic scenarios contrast
matched collateral, coin fees, retained funding and converted funding.
Tests cover extreme common prices, basis divergence, fee insolvency, discrete
sizing, direct-ledger reconciliation and invalid inputs. These are arithmetic
checks, not market validation, training examples or benchmarks.

The [generated artifact](inverse-collateral-scenarios.json) binds exact source
and implementation bytes. No historical result or registry acceptance changed.
This supports reviewing the existing dated-carry and funding families with
correct collateral/cash-flow models; it does not establish a new accepted
family or demonstrate current capital savings. Separately margined linear
hedges and portfolio-margin offsets must be compared on the **same risk and
liquidity budget**, not just a smaller denominator. Reduced nominal margin
alone is leverage, not profit.

## Source and confirmation gates

Before a venue-specific implementation, obtain an independently available
primary contract (not an alias retry) covering reciprocal PnL, actual integer
unit and precision, delivery/mark basis, liquidation/ADL, maintenance tiers,
collateral recognition and fee debits. Account-specific qualification stays
unproved without separate read-only authority. No collector is justified yet.
The original rank-14 account-evidence trigger and the terminal linear/inverse
funding test remain unchanged. See the [next-round plan](economic-rd-rounds.md).

Checkpoint verification: 25 focused tests cover scenario arithmetic and both
generated/retained artifact hashes; Ruff passes. Full CI and GPU timing were
not rerun for this isolated offline tool. Read completely for this increment:
the new evaluator/tests, `quarterly_carry.py`, and the new review/round plan.
Existing funding contracts were field-projected, not exhaustively reviewed.
