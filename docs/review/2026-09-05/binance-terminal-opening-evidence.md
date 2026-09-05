# Terminal opening evidence: observation stage

This checkpoint implements durable, exact-order execution observations for
unresolved Binance openings. It does **not** complete recovery, apply inventory,
clear UNKNOWN, rearm trading, or qualify an edge. The source bindings and exact
verification scope are in [the JSON record](binance-terminal-opening-evidence.json).

## Implemented behavior

`collect_opening_recovery` reads the existing scope-bound intent, queries its
exact client ID, checks terminal identity before requesting any trades, then
requests at most one exact-order trade page. The unchanged bounded GET retry
policy still applies; this is not a promise of exactly two HTTP transmissions.
An exact zero-execution terminal order needs no trade query. Not-found, missing
fields, mismatched scope or incomplete fills never mean zero exposure.

Trade IDs must be unique. Decimal strings remain exact through quantity, quote
and native commission sums. Spot long and one-way USD-M long/short MARKET
openings are supported. Futures closing evidence, other order types and hedge
mode are not silently interpreted as compatible openings. A price-times-quantity
discrepancy requires instrument-precision adjudication; no tolerance is guessed.
Native fees and rebates are retained by asset, not converted to an assumed quote
value or treated as zero. Gross filled quantity is not net Spot inventory.

The collector saves a validated field allowlist, exact request binding and
normalized evidence in an additive table inside the existing v2 SQLite journal.
It checks the pending intent again inside the write transaction. Identical
concurrent observations do not duplicate rows; conflicting observations cannot
overwrite evidence. Reopening retained evidence revalidates it without another
venue query. Corruption rejects rather than triggering a repair/refetch.
Legacy journals are not automatically migrated or attributed to current keys.

## Verification and reasoning

82 new cases are included in the 371 passing affected-domain checks. These cover
Spot/Futures and long/short, canceled partial fills, multiple native fee assets,
invalid/missing/duplicate execution data, scope rotation, storage failure,
concurrent observations, corruption and an abrupt child-process exit with code
71. The restarted process loads the committed observation with UNKNOWN intact.
Ruff and diff checks pass. The initial red was an absent implementation import,
not proof of a previously accepted bad fill. No benchmark claim follows from
test duration. No hosted/full-repository verification is claimed.

Semantic review also caught an overly broad product-field allowlist during
implementation; the final code retains only fields validated for that product.
The Python and regression skills guided the exact-decimal core, paired failure
paths and process-exit check; the documentation skill keeps this stage distinct
from full recovery and preserves every earlier result.

## Source basis and remaining work

The Spot account-trade endpoint exposes per-trade quantities and native fees;
the order query alone is not a fee ledger. See the official
[Spot REST reference](https://raw.githubusercontent.com/binance/binance-spot-api-docs/master/rest-api.md).
The official USD-M reference documents a 1,000-row maximum and a recent-seven-day
default when times are absent. Older or larger fill histories may therefore
fail this bounded collector; a short page is never completeness proof. See
[Account Trade List](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/trade#account-trade-list).
These sources were inspected through browsing; no retained raw documentation
artifact is claimed. No signed venue request or actual credential was used.

The next substantive integration is an owned inventory/native-fee ledger,
atomic recovery application, then explicit rearm against current account,
policy and process-generation evidence. CLI/Windows exposure must share that
contract. This collector is not automatically called by the trading loop and
does not itself supply the frozen authenticated-operation workflow. Failed
attempt response capture, receipt-time provenance, legacy/key-rotation migration,
whole-store deletion/rollback protection and independent process supervision
remain unfinished. A validated terminal observation is not current-account
flatness, complete inventory ownership, profitability or enterprise readiness.
