# Binance order responses: bind the original request through recovery

September 8, 2026. [Source-bound review](binance-order-response-binding.json).
Base: `019db70c869a81b95c1c536b754e9d5bf1f0a576`.

The baseline shared client accepted a mocked ETH SELL response to a BTC BUY
submission, despite different client ID, exchange order ID and quantity. An
exact-order GET also returned the foreign payload. No actual exchange, account,
credential, user ledger or protected capture was used to reproduce it.

## Implemented boundary

An immutable MARKET request binding now precedes actual transmission. It records
the exact outgoing symbol, side, eight-decimal quantity, product, client ID and
reduction intent. Responses must match symbol, numeric order identity, every
supplied client identity alias, side, MARKET type and original quantity. Futures
responses must also match one-way BOTH and exact boolean reduceOnly; contradictory
optional closePosition or origType fields reject. A quantity rounded to zero or
a nonboolean reduction flag fails before order/leverage transmission.

When a caller omits a client ID, the API creates a 36-character random ID before
transmission. Explicit IDs are preserved. This establishes a selector for the
current call; it does **not** create a durable intent or permanent idempotence.
[Binance documents](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/trade)
client-ID uniqueness among open orders, not an eternal deduplication guarantee.

Exact-order queries require every supplied selector to match. Ambiguous numeric
IDs such as booleans, floats or whitespace-padded strings fail before access.
Autonomous open/close and direct CLI response-loss paths carry the original
request semantics through the query. The CLI pending-fill query retains the
semantics of its first API-validated submission response. This closes the bypass
where a rejected POST response could be replaced by a query answer checked only
for a matching ID. Missing pending-response semantics fail without another query.

The [official futures query schema](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/trade)
separately exposes symbol, IDs, original quantity, side, positionSide and reduceOnly.
It describes order IDs as per-symbol. Documentation was inspected through web-tool
extractions, not captured as raw HTTP bytes or treated as economic evidence; the
two old futures documentation URLs redirect to one consolidated source.

## Verification and limits

660 distinct offline checks passed across stages: 595 affected execution-domain,
61 CLI, and four additional normal response-loss recovery cases. The new boundary
has 105 cases, including missing/mismatched selectors, precise quantity mismatch,
futures mode, invalid input before transmission, generated versus explicit IDs,
and actual client-path open/close response-loss recovery. Conflicting queries
preserve the owned lot and UNKNOWN; matching Spot/Futures opening and closing
queries still complete normally. Transport is stubbed; no virtual or real order
was sent. Ruff, targeted formatting and whitespace checks passed.

The new response module was read completely; the review JSON identifies exact
existing function ranges and normalized source hashes. No unchanged execution
suite was rerun just for the documentation update. The final four added cases
were run separately instead of repeating the preceding 656 passing checks.

This does not prove terminal trade completeness, native-fee cash/inventory,
current balances or margin, account UID, bot ownership, a supervisor or rearm
policy. Direct CLI durable-intent and account-scope integration remain incomplete.
Generic ID-only reads cannot establish an original plan that was not supplied.
The existing RuntimeConfig symbol normalization is unchanged; CLI tests use its
verified effective symbol. No CLI option/native parser contract was changed.

Financial truth remains 37 accepted mechanism scopes, 65 hypotheses, 196 terminal
observations and zero qualified stable profitable edges. No market screen was
repeated, no training/benchmark ran, and this repair is not a profitability claim.
Next priorities remain an eligible informative financial study, native-fee
incremental accounting and durable direct-CLI recovery, then explicit rearm and
independent process controls. Broader code review and final bug hunting are pending.

## Efficiency correction

The workflow now states Ruff's exact line-range syntax: `START-END`, not
`START:END`. Valid ranges also triggered a formatter panic on the inspected older
API tests. Rather than retain unrelated formatting churn, the corrected workflow
decodes BOM-bearing files with UTF-8-sig, formats in memory, verifies AST equality
and applies only edited function blocks. Existing file encoding and unrelated
code were preserved; no toolchain upgrade was required.
