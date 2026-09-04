# NYC September 5: small gross discounts, negative after configured fees

One prospectively fixed public Gamma GET returned event 958036, all eleven
temperature bins, eleven usable YES prices and seven usable NO-side proxies.
The 57,128 response bytes, request journal, contract and exact implementation
hashes are retained here. The other four NO sides remain incomplete; no
missing-side request, book, fee endpoint, account, order or protected capture
was accessed.

Selection was the next full NYC calendar day at the verified September 4 UTC
clock, using the retained daily slug pattern before any current price-bearing
page or response. The response confirmed that exact event and eleven-market
population. All markets have one identical description and resolution URL:
NOAA LaGuardia hourly data, with the stated Weather Underground and lowest-bin
fallbacks. This is not a forecast of the temperature and is not a claim of
independently verified on-chain settlement completeness.

## Economic result

The existing complete long-only basis calculations produced 17 price-complete
frontier rows: the all-YES set, available binary straddles, and the cheapest
metadata/stressed NO-cardinality frontiers. Four rows had positive gross
headroom; two are different orderings of the same seven-NO set, not independent
opportunities. None passed the frozen fee-and-one-adverse-tick gate.

The gross-positive rows hit a one-pUSD per-leg price boundary under tick
stress, so their stressed net is **unavailable**, not a numerical loss or zero
fee. The strongest finite stressed row was a binary straddle losing 0.02150
pUSD at five shares. Do not confuse that finite row with the gross leader.

A separately recorded **post-capture, zero-network rejection-only** audit then
applied the retained configured taker fees at the unchanged original prices,
with no adverse ticks. It did not change the frozen test or authorize capture.

| Gross-positive row | Gross headroom, pUSD | Configured fees, pUSD | Net before any adverse tick, pUSD |
| --- | ---: | ---: | ---: |
| Six NO, metadata ordering | 0.05000 | 0.15344 | -0.10344 |
| Six NO, stressed ordering | 0.01000 | 0.15146 | -0.14146 |
| Seven NO, either ordering | 0.06000 | 0.15394 | -0.09394 |

Every gross-positive row includes a single leg with a 0.06160-pUSD configured
fee, already larger than its entire gross headroom. Thus the rejection is not
solely due to a conservative tick allowance. All 17 fee-audit rows are
nonpositive without tick stress. These are source-based cost diagnostics,
not executed PnL, account fee receipts, capacity or fill guarantees.

## Capture and publication changes

The v2 bridge preserves the old frontier economics and replaces only its
unbounded transport: a two-MiB ceiling plus one overflow-detection byte,
exclusive intent journal, durable raw/partial/error retention, no redirects
or retries, and bounded socket reads. The between-chunk elapsed check is not
a hard wall-clock cancellation guarantee. Seven network-free transport tests
cover successful capture, replay refusal, size overflow, partial timeout,
HTTP redirect/error retention and wrong-method rejection.

Twelve focused checks pass in total, including the exact-event fee and hash
reconstruction and the central profitability/durability ledger checks. Ruff
passes on the changed Python files; no broad CI or timing benchmark was run.

The exact event is now terminal in rank 31. Canonical counts are 37 accepted
scopes, 65 hypotheses, 187 terminal observations, zero stable current
account-qualified after-all-cost edges. The preceding registry and durability
ledger bytes are retained as `registry-before.json` and
`durability-before.json`; no historical experiment result was rewritten.
The fee-only audit is supporting evidence, not a second terminal observation.

Do not refresh, reprice, request missing NO sides or books for this event, or
relabel the three tiny gross discounts as profitable edges. Future research
still needs a distinct eligible population or the exact material-change
trigger. No new model or larger training dataset is justified by this result.
