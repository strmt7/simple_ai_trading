# Receipt-native Spot realized cash PnL

The baseline `86edabe7` reproduced a false break-even: buy 1 BTC for 100 USDC,
pay 0.001 BTC commission, then sell the received 0.999 BTC for 99.9 USDC.
The ledger reported zero; actual quote-cash loss is 0.10. The failing regression
was run before implementation. [Canonical evidence](spot-native-cash.json)
binds the repaired source and verification scope.

New autonomous Spot lots now retain selected, validated entry receipt fields.
The completed trade retains the closing receipt too. Realized PnL subtracts
allocated original entry cash, including quote fees, from actual net sale cash.
Base entry fees are already reflected in the smaller received inventory; they
are not charged twice. Partial allocations use exact rational arithmetic against
the original received quantity, not repeatedly rounded remaining float costs.
The existing PnL/fee/percentage fields are float projections reconstructed from
the receipts before publication and on load. Fee totals value entry base fees
at that execution's VWAP; this is cost attribution, not a new conversion trade.

Receipt identity, gross/net quantity, execution price, native fee rows, closing
status and projected numbers must agree. Unknown response fields are excluded.
Older absent-receipt records and durable request bytes remain compatible; no
historical lot, dataset or result was retroactively given native evidence.

Nonzero third-asset fees have no invented quote valuation. SELL base fees require
separate inventory reconciliation. Missing fee rows, unvalued fees and conflicting
receipts cannot complete native accounting: after submission the durable close
stays UNKNOWN, preserving the existing block on repeated sells/new exposure.
This does not cancel a submitted order or assert that the exchange lot is still
open. Exact-order recovery must reconcile it before rearm. A query response with
only average price is not substituted for complete commission evidence.

## Verification and limits

730 distinct affected checks pass across staged runs, including 46 new cases.
The 464-case affected suite initially had one incomplete identity fixture;
its corrected focused check passes. A later 265-case native/compatibility run
passes, plus one injected-interruption/restart test. These are not one full-suite
or hosted-CI result. Ruff passes on seven touched Python files. The local skill
workflow kept regression reproduction, restart boundaries and artifact checks
separate; it did not justify another market-data capture.

Native unrealized PnL, quote-aware portfolio/risk aggregation, third-asset
valuation, SELL fee ownership, account qualification, incremental terminal
recovery, explicit rearm, direct CLI integration and independent supervision
remain unfinished. Open `entry_fees` still carries its modeled planning estimate;
the native realized path does not use it as cash truth. Legacy and Futures
accounting are preserved, not certified. This is a demonstrated capital-accounting
repair, not a profitability result or enterprise-readiness claim.

No exchange requests, credentials, orders, protected data or unrelated workstation
processes were used. Research totals are unchanged from the last bound registry:
37 accepted mechanism scopes, 65 hypotheses, 198 terminal observations, zero
qualified stable profitable edges. October 7 is the actual continuation date;
September timing instructions are historical, not automatic repeat authority.
After this checkpoint, re-evaluate current eligible research triggers using the
complete capture boundaries before any new public market request.
