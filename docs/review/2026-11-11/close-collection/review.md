# Bounded collection of an owned close

The new programmatic collection boundary queries one original Binance
testnet/Demo closing order under ownerless Stop. It retains verified terminal
status and native fills in the original journal. It does not apply inventory,
resolve the close intent, complete Stop or rearm. The reviewed base is
`e32758f68eeb037a0b9c9ee40e0f420df2d11238`.

`closing_collection_sha256` inspects the exact unresolved request within one
noncreating SQLite snapshot. `collect_stopped_closing_recovery` requires that
digest, the original execution scope, matching explicitly supplied HMAC
credentials, enrolled execution storage and the matching controller. It holds
the existing dispatch and close interlocks through collection, verifies storage
before and afterward, and leaves application to the separate retained-evidence
boundary. An active owner, a rotated key, changed request or missing enrollment
rejects before a worker starts. A digest identifies a local request; it does
not establish authenticated account ownership or current venue availability.

The client allows one exact client-ID order query and, when execution is
positive, one page of at most 1,000 trades for that observed order ID. The
existing terminal validator rejects nonterminal orders, missing fills,
conflicting native cash, identities and commissions. A zero-fill terminal order
needs no trade request. Already retained evidence is validated and reused
without refreshing the account. No pagination, automatic retry, clock correction,
order submission or cancellation is provided.

## Process and resource boundary

Both reads share one monotonic response deadline, 20 seconds by default and
configurable from 1 to 60 seconds. Startup uses at most five seconds from that
same budget. Signing follows startup. The fixed signed receive window is five
seconds; a clock problem fails rather than silently adjusting time or replaying
the request. Each native response is limited to 2 MiB; framed replies also have
a finite bound. Local SQLite admission and worker teardown have their separate
existing bounded waits. The response deadline is not a whole-command latency
benchmark.

The existing session-owned HTTP engine supplies hidden creation, Windows job
ownership, bounded framing and teardown. Its new authenticated read policy
admits GET only, no body and an explicit endpoint set, while permitting the
authentication header. Its credential-free public policy continues rejecting
headers. The recovery client binds the endpoint set to the original product's
order and order-trade endpoints. The API secret stays in the signing process;
the worker receives only the signed request and API key through IPC. Ambient
credentials are excluded from its environment. This is containment, not a
credential-security sandbox or a memory-erasure claim.

The shared engine currently belongs to the `polymarket-live` optional dependency
cohort. Recovery imports it lazily and reports an actionable setup requirement
when absent. It does not change another application, install globally or infer
an available runtime. Credentials never enter command arguments or diagnostics.
Transport and cleanup errors use static messages. Every owned client is closed
on success and failure; no scheduler, service or restart hook is installed.

## Verification and remaining work

The [sealed evidence](evidence.json) records the source-bound affected checks,
native subprocess fixtures, resource assertions and retained failed phases.
There are 156 distinct affected passing cases: 132 unchanged existing cases
from the retained final phase and 24 new cases from the corrected 49.93-second
run. Production source stayed unchanged between those phases. After one warm
lifetime, 12 further native lifetimes met the four-handle, zero-local-lock-growth
and 8 MiB private-memory-growth bounds; every observed child exited. These are
bounded fixture assertions, not a host-independent memory or speed guarantee.
Synthetic workers validate the signed query and reject writes and unlisted
endpoints before reaching the native request method. Faults include stalls in
either read, process exit, explicit rejection and invalid/incomplete responses.
Tests preserve unresolved admission, inventory and the Stop latch.

The initial active-owner test expected the wrong exception family; production
rejected correctly. The first resource fixture retained completed `Popen`
objects and therefore one observation handle per iteration. The corrected
fixture verifies each exit and drops only its own reference, with no forced
garbage collection or production cleanup substitution. Original failed logs and
source archives remain retained. The final receipt distinguishes reused checks
from the corrected run instead of reporting an invented single all-pass run.

CLI/native collection selection, credential provisioning, actual account UID
and symbol/filter qualification, complete operator recovery/rearm and independent
trading supervision remain open. Futures observations retain native PnL; this
does not add futures inventory application or establish local lot cash. A
post-collection error can follow a committed evidence row: inspect the original
journal before another separately authorized read. Never replay its order.

No actual account request, order, market capture or fit occurred. Targeted URL
discovery did not confirm Auto Order Routing API support; this leaves it unknown
and does not reopen SOR or the consumed conversion studies. No new source-content
claim is made. Financial counts remain 206 observations, 65 hypotheses,
37 historical scoped mechanisms and zero qualified stable edges.
