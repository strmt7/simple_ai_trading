# Future NFL total-ladder rejection

The prospectively frozen [contract](contract.json) was used exactly once for a
public unauthenticated Gamma keyset request. The native HTTP-200 response was
1,876,400 bytes, retained in `events.raw`; its SHA256 is
`34c9c9cbb09c878229b455d019b230c579433a5114c94dd6dd2197a9085ab592`.
Intent/completion journals and separate capture/screen results retain the
request, source, implementation and terminal-decision bindings.

Selection used the earliest native full-game start/slug/id before prices. The
selected `nfl-buf-min-2026-11-10` contains 305 native markets and 21 active
full-game totals. All 210 unordered total pairs were price-complete. None had a
strictly sub-floor selected-side cost. The strongest conditional one-unit payout
floor cost 1.04 before fees, using `bestAsk` for Over 27 and `1-bestBid` for Under
29; gross surplus was -0.04. These are acquisition proxies, not synchronized
executable depth. Oracle/collateral identity, fees, clocks, fills and independent
recurrence were not qualified. No edge or profitability claim is admitted.

The reusable pair driver separates a 50-MiB native transport bound from the
128-total/8,128-pair computational bound. Every eligible unordered pair uses the
unchanged complete rule/asset validator; unrelated props do not determine the
payoff budget. This fixes the design for a distinct unobserved population while
preserving the old October 8–9 resource failure, implementation and results.
It does not repair or reprice that consumed population. Native imports and full
pre-network validation passed before freezing/access.

The whole November 10 UTC window and every returned event are consumed:

- `nfl-buf-min-2026-11-10`
- `nfl-buf-min-2026-11-10-player-props`
- `nfl-buf-min-2026-11-10-first-td-scorer`

No refetch, alias, repaired row, alternate sibling, budget increase or book/fee
capture is permitted for this population. There were zero book, fee, account,
credential, order, transaction or funds requests. The floor result rejects this
snapshot only; it proves neither family-wide impossibility nor negative expected
value for every strategy or future state. There is no permission to resample
until success. A future round should ask a different informative question.

The new driver, legacy floor and bounded transport affected suite passed 72
checks on synthetic/local fixtures, without network access. The separate
registry/audit binding checks passed twice out of two. Offline verification of
these retained artifacts is recorded in `verification.json`; it does not rerun
the one-use capture or evaluate a new population. Counts advance once, from 202
to 203 observations, with 65 hypotheses, 37 scoped mechanisms and zero qualified
stable edges. The 210 dependent pairs are not independent observations.

During registry publication, the previous HEAD already contained a stale registry
self-hash: recorded `91cbf72283d724a9f493f28edeb0c73fd6b73d85dbc9f5201427fddd32550bd1`,
actual canonical `6f718560f546b47fb234e79978977893a77402430228b94a4fa1b62b9d54ccfa`.
The current mutable routing registry was rehashed after appending this result,
and the dependent audit binding was updated. Prior committed bytes and historical
economic results are preserved; the mismatch is not presented as a passing
historical integrity check.
