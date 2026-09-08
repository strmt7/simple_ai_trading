# September 9 NYC: complete long-only price rejection

September 8, 2026. [Frozen contract](contract.json), [canonical result](result.json),
[retained primary response](raw/event.json), [durable request journal](request-journal.jsonl).

One public unauthenticated Gamma GET returned event 981253 and all 11 expected
markets. The next full New York calendar day was selected before prices at
12:34:55 UTC; request intent began at 12:36:32.160 UTC. The response was 56,225
bytes, HTTP 200, with no redirect or retry. The unchanged bounded collector and
all nine implementation bindings passed their preflight; the exact retained
September 6 loader was exercised before the new contract was frozen.

Every YES and NO side was price-complete. The entire frozen primitive long-only
basis was screened: all-YES, every binary YES-plus-NO straddle, and every optimal
k-NO cardinality frontier under gross and fee/tick costs. All 28 retained rows
failed the gross gate; none passed configured fees plus one adverse tick per leg.
Rows include alternate orderings, not 28 disjoint trading opportunities.

| Package | Gross cost per share | Payout floor per share | Gross floor at five shares | Fee/tick floor at five shares |
| --- | ---: | ---: | ---: | ---: |
| All 11 YES | 1.093 | 1 | -0.465 | -1.03217 |
| Best gross: 75 F or below binary straddle | 1.001 | 1 | -0.005 | Unavailable: stressed NO reaches 1 |
| Best finite stress: 92-93 F binary straddle | 1.007 | 1 | -0.035 | -0.04773 |

Amounts are pUSD. The last row has 0.00273 pUSD configured stressed taker fees.
An unavailable stressed subtotal is a failed gate, not a zero fee. YES acquisition
uses direct bestAsk; NO uses conservative one minus YES bestBid, never midpoint
outcomePrices. These are rejection-only metadata, not executable order-book quotes.

## Financial interpretation and limits

The common retained description resolves the whole-degree Fahrenheit high for
September 9 at NOAA LaGuardia hourly data. The bins are at most 75, each consecutive
two-degree range from 76-77 through 92-93, and at least 94. They cover the stated
integer domain without gaps or overlap. The common fallback is Weather Underground
when NOAA is unavailable by the next-day deadline, then the lowest bracket if no
data exists. All 11 descriptions and resolutionSource fields match. The metadata
endDate of 12:00 UTC is not substituted for the description's full-day observation.

The one-pUSD complete-set and k-minus-one NO floors are conditional on this common
mutually exclusive resolution model; no on-chain question-count, adapter, conversion,
or exceptional oracle-state proof was obtained. A source-price rejection does not
need to assume those unverified execution facilities work. Buying the complete set
already costs 9.3% more than its one-pUSD model payout before fees. No direction
forecast, model training or GPU benchmark could reverse that fixed payoff arithmetic.

No books, fee endpoint, on-chain request, account, credential, order, funds or
protected capture were used. The exact event is consumed; no missing-side repair,
refresh, reprice, alias or sibling selection is authorized by this result. It is
not a theorem that all weather trading, maker policies or future events have negative
expected value. No stable profitable edge, deployment readiness or acceptance follows.

## Durable integration

[Amendment plan](registry-amendment-plan.json) binds the preceding Git checkpoint
dafb6b6eebf16b60cf255ffe6bbb6ebe491690b2 and both previous mutable self-hashes.
Rank 31 receives the contract/result bindings and one terminal observation;
the durability audit binds the updated registry. Prior entries and frozen results
are preserved. Totals: 37 accepted mechanism scopes, 65 hypotheses, 196 terminal
observations, zero qualified stable profitable edges.

Focused offline verification: 13 checks passed (three new exact-event checks,
three preceding-event checks and seven bounded-transport checks). They reconstruct
the full retained result, source bytes, prospective date, journal, rule bins and
downstream bindings. A separate Git-baseline comparison proved that removing only
the documented additions and restoring timestamp/self-hash fields reproduces both
preceding mutable ledgers exactly as JSON values. Ruff and diff-whitespace checks
passed. No economic request is repeated by verification. Broader enterprise/recovery
work remains unfinished.
