# Native realized-cash quote boundary after the last close

Closing inventory must not erase its cash denomination. A synthetic native
BTCUSDC round trip loses 1 USDC in entry commission; an unrelated legacy
ETHUSDT round trip reports +2 USDT. With no open positions, the previous
admission guard offset those amounts and reported zero daily/session loss.
The new regression failed before repair. No FX basis justified that offset.

The [source-bound record](native-flat-quote-guard.json) retains the reproduction,
implementation bindings and verification scope. Native receipt history now keeps
instrument/product and quote qualification active after the last close. Flat,
same-quote cash needs no current market price. Open inventory still requires
its exact instrument/product mark. Unknown denomination blocks new exposure;
it does not authorize an uncertain forced close. Guard zero fields on rejection
are uncalculated placeholders, not a zero-loss claim. No order path changed.

378 affected Binance risk/reconciliation checks pass, including 13 new cases:
mixed quotes, missing/invalid context, same-quote known loss, persisted restart,
entry-gate integration, day rollover and unchanged legacy-only behavior. Ruff
passes. Fixture expectations were corrected for the existing daily loss limit
and cooldown; neither production limit was weakened. No full suite, hosted CI,
benchmark, GPU workload or model fit was run. The old base-commission cash
defect was already fixed and was not repaired again.

This closes the no-open-position native-history admission gap identified in the
[October 7 review](../2026-10-07/spot-cash-mark.md), not general portfolio FX,
native UI reporting or legacy/native cash comparability. Same quote does not
prove complete execution costs. Native hedge coverage, capital/margin costs,
source-qualified labels and the stateful two-sided objective still precede
training. Independent trading supervision and final whole-repo review remain
open. Historical results and protected captures were untouched. No market or
account request, credential use, order or automation occurred. Financial counts
remain 202 observations / 65 hypotheses / 37 scoped mechanisms / 0 stable edges.
