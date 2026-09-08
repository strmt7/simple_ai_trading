# Newly deployed CFB event: deployment proved, ladders rejected

Baseline: `a688afa87a663a69bb4ec0ef9dd5edb807610a44`.
The [deployment contract](contract.json), [retained source receipt](source-result.json),
[deployment result](deployment-result.json), [offline ladder contract](ladder-contract.json)
and [all 553 ladder rows](ladder-result.json) bind this result.

## Information gain

The August 31 catalog for September 8-12 returned zero events at
16:33:29.370 UTC. Its immutable result remains unchanged. A later official
[Florida A&M vs. Miami (FL) event](https://polymarket.com/zh-hant/sports/cfb/cfb-flam-mia-2026-09-10)
was found during two bounded searches that did not establish a new WNBA event.
Search metadata was only a deployment lead, not a price or creation-time input.

Under the user's session-only rule-revision authority, one exact public GET
tested actual deployment before inspecting economic fields. The source-bound
event creation timestamp was August 31 at **22:00:14.747908 UTC**, later than
the empty observation. The event starts September 11 at 00:00 UTC and had
58 embedded active, accepting markets. This proves a changed population, not
an error in the old observation or an excuse to refresh its entire catalog.

The request was journaled at 14:32:41.799 UTC on September 8 and completed at
14:32:42.339 UTC, HTTP 200, 181,864 retained bytes. Raw SHA-256:
`17bfdc911797f5a584ed1f664a8f1309962a19b194cf22eae39a5cdd0405fce1`.
No redirect, retry, credentials, account, book, fee, order or protected capture.

## Economic rejection

A separately frozen zero-network test used only the retained response.
Rule-only preflight identified 23 spreads and 25 totals in two identical-rule
groups. Only threshold phrases were normalized; teams, observation timing,
overtime, cancellation and source differences were preserved. Every compatible
lower-positive/higher-complement pair was enumerated: 253 spread plus 300 total
relations. All 553 had positive side-specific acquisition evidence, and none
cost strictly below its conditional one-pUSD common-rule payout floor.

The cheapest pair was Over 60.5 (market 4240031, ask 0.53) plus Under 61.5
(market 4349890, conservative `1-bestBid` 0.54): cost **1.07 pUSD** for a
one-pUSD floor, before fees. No fee or book escalation is justified. This does
not prove negative expected value for every strategy: the middle state can pay
more, but no probability model or expected-value claim is made here.

The ten other markets were explicitly excluded from this ladder scope; this
is not a complete cross-family graph. Exact synchronized depth, owned fills,
costs, venue/resolution risk and recurrence would still be needed for any edge.
Both contracts are consumed. Do not refetch or reprice the event under them.

## Verification and continuation

35 focused checks pass for deployment identity/time, immutable binding, restart
of an already-consumed adjudication, complete rule grouping, side-specific
prices, missing/zero quotes, duplicate identities, cancellation, altered
observation rules and pre-economic relation ceilings. All transport checks used
the existing bounded collector; no duplicate transport implementation was added.
The final checks reconstruct every retained economic row and verify registry/audit
binding independently. Four changed Python files pass Ruff. A missing placeholder hash field was fixed
before validation/output creation/network access; this was an unconsumed local
preflight error, not a market retry.

A separate publication check caught ambiguous patch context placing the new
references under rank 18 rather than rank 30. This was corrected before commit;
an exact-family membership assertion now supplements the canonical hash check.
Every other hypothesis row and the complete prior terminal prefix are verified
unchanged. Raw evidence and journals receive explicit byte-preserving Git rules.

The registry now has 198 terminal observations, 65 hypotheses and 37 accepted
mechanism scopes; qualified stable profitable edges remain zero. Rank 30 also
now explicitly excludes the already-consumed September 3-9 WNBA window, which
was recorded in its latest status but omitted from its literal retry text.
The session exception applies only to this exact event and does not generally
reopen consumed date ranges. Continue with independently satisfied research
triggers or concrete capital-safety work, preserving all historical outcomes.
