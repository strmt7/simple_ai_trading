# Option/future settlement identity after the matching expiry

Rank 13's literal retry gate permitted a separately frozen prospective contract
after the September 25 matching expiry. The observed research date is November 9.
This round used that gate for the previously unrequested BTC September 25 window;
it did not resume, reorder or repair the terminal August eight-request contract.
The complete rank-13 row, its terminal adjudication and continuation were checked
before access. The scoped duplicate inventory covered 21 settlement/delivery
artifacts and their native request bindings; its scope is recorded in the
[contract](contract.json), not presented as a scan of protected private captures.

The financial reason for the check remains important. A synthetic option position
and opposite quarterly future have a constant terminal payoff only if their
actual settlement values, units and quantities agree. Matching expiry labels or
current price correlation cannot establish that identity. A failed source query
cannot establish that the values disagree either.

## What the two native responses establish

The first public GET returned HTTP 200 and 100 option exercise rows for the exact
BTC September 25 expiry. The pre-frozen parser required native integer expiry
timestamps, exact settlement strings, valid strike/result/side fields, no duplicate
symbols, one common settlement value and same-strike call/put pairs. It found
44 such pairs. The page reaches its 100-row ceiling: it is a bounded observed
sample, not proof of the entire option population. No continuation was requested.

Only after that gate passed did the second public GET request BTC futures
delivery history. It returned HTTP 200 and 18 rows, whose latest UTC delivery
calendar marker is June 26, 2026. None falls on the frozen September 25 date.
The parser stopped before any numeric comparison. The current HTTP response
date does not establish that the history payload covers the elapsed expiry.
No exact futures settlement epoch is inferred from these calendar markers.

Therefore **neither numeric equality nor numeric disagreement is proved**.
This is improved source evidence for one side and a precisely identified missing
record on the other. It does not qualify a fixed-payoff trade, independent
profitability, current/future benchmark identity or executable capacity.

## Controls, efficiency and next decision

The complete byte-to-file path and exact-field parser were checked on synthetic
UTF-8 data before freezing the controller digest. Method, canonical URL and empty
body hash were durably journaled before each request. Both response bodies and
receipts were fsynced before semantic decoding; raw bytes, hashes, journals and
the executed controller snapshot remain private. Both calls completed in the
same bounded, hidden, session-owned process. The controller is now inert.

There were two public unauthenticated GETs, no redirects, retries, pagination,
source aliases, books, credentials, account access, fits, orders or persistent
workers. This exact-API study used native transport; it read no web page. No
protected partial data or old result bytes were changed. The source-bound
[result](result.json) and current rank-13 row preserve the consumed failure.

Do not substitute an older date from the returned futures history or poll this
window. Reopen only on independently new official evidence supplying the missing
exact-date record, or a separately frozen distinct later source-qualified common
expiry after its actual settlement. The original broad post-September trigger
is retained as provenance and narrowed now that this follow-up is consumed.
No depth escalation follows this result.

A possible option-only box alternative was also checked against the existing
registry and live implementation. It is already a terminal screened family;
rank 37's separate RFQ/account gate remains unchanged. No duplicate hypothesis
or quote request was created. The existing carry account/fee gates likewise
remain unchanged. Financial counts stay 203/65/37/0; this adds no evaluated
strategy, accepted edge or new hypothesis. The broader goal remains active.
