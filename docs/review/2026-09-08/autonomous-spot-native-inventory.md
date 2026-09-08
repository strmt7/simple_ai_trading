# Autonomous Spot gross/net inventory

Baseline: `317c1a317fef0d7afecd1a5e310bf224b001f7f7`. The
[source-bound record](autonomous-spot-native-inventory.json) is authoritative.

An offline execution of the baseline opening projection reproduced 1 BTC of
sellable inventory after buying 1 BTC with a 0.001 BTC commission. Only 0.999 BTC
was received. This is the autonomous counterpart of the preserved
[CLI resale defect](binance-spot-resale-ownership.md).

New fully filled Spot lots now retain original gross entry quantity and base
commission as exact decimal strings; active quantity is net received base.
The original opening intent still binds gross executed quantity exactly to
the eight-decimal transmitted order. Net quantity cannot disguise an underfill
or overfill. Complete native fee rows are mandatory; futures and paper behavior
is preserved without manufacturing native Spot evidence.

Closing carries that receipt metadata forward and submits remaining net base.
Native partial closes use exact subtraction under a private decimal context;
a one-satoshi remainder is not deleted by the old relative float tolerance.
Quantities the existing float/eight-decimal adapter cannot represent exactly
remain unresolved, rather than silently rounded or dropped.

Empty new fields are omitted from canonical opening and nested closing request
templates. Older pending intents remain byte-compatible and old lots are not
assigned invented commissions. No actual user ledger was migrated.

## Verification and limits

744 distinct affected checks passed across stages, including 60 new cases.
The initial retained JUnit runs contain three subsequently corrected fixture
failures: two unfilled-order fixtures retained inconsistent fills, and one
execution-scope fixture omitted native fee rows. Focused reruns passed. These
are not a claim of one all-green 744-case run. Tests include all six supported
Spot pairs, base/quote/BNB/zero fees, gross/net mismatch, restart, interruption,
one-satoshi remainder, wire rejection and unchanged paper/futures boundaries.

This is not native cash PnL. Entry fees remain modeled on gross traded notional;
exit commissions and third-asset valuation remain separate unfinished work.
Missing, nonterminal or unrepresentable Spot evidence preserves UNKNOWN and
does not create guessed inventory. Retained terminal recovery movements are
not automatically applied by this change. Account qualification, explicit
rearm, CLI durable recovery and independent supervision remain required.

No market/account requests, orders, credentials or protected data were used.
Financial totals remain 37 accepted mechanism scopes, 65 hypotheses, 197
terminal observations and zero qualified stable profitable edges. Historical
results and exact capture retry rules are unchanged. The source map records
only reviewed boundaries, not completion of the whole-repository review.
