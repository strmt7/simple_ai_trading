# Binance credential transport boundary

Outcome: **fixed in the reviewed BinanceClient path**; see the
[canonical source bindings and verification command](binance-transport-authority.json).
This is capital-safety engineering, not profitable-edge or complete-security proof.

## Cause and repair

Tracing exact-order recovery exposed a prerequisite failure: environment override
or mutable `base_url` -> hostname-only guard -> `Session.request` admitted HTTP
testnet URLs. Separately, constructor API key -> global session headers -> every
prepared request sent the key on unsigned public calls. Three offline pre-fix
reproductions failed: unsigned HTTP, signed HTTP and public API-key transmission.

The shared request boundary now validates a bare HTTPS origin, rejects URL
components and invalid API paths, and uses the same origin snapshot throughout
the operation. Signed calls remain official testnet/Demo only. Authentication is
per-request; unsigned calls explicitly remove stale session API-key headers.
Rejected-origin diagnostics do not echo the supplied URL. Host classification
for metadata stays unchanged. Test transport doubles accept Requests' header
argument; production code contains no test-specific path or weakened checks.

Official HTTPS Spot/Futures testnet and Demo signed calls and public HTTPS
mainnet/custom origins remain supported. Plain HTTP and component-bearing base
overrides are intentionally rejected; a root trailing slash is normalized.
Existing GET retries and no-write-replay/redirect behavior remain covered.

## Critical review and verification

The security-fix skill shaped the work: reproduce the source-to-sink defect,
repair shared enforcement, then challenge bypasses and legitimate compatibility.
Repository rules prohibit subagents, so separate pre-patch compatibility and
post-patch bypass perspectives were performed locally, not independently delegated.
The review distinguished metadata classification from execution authority and
traced the only Requests sink, including later origin mutation, stale mixed-case
headers, userinfo, ports, control characters, URL components and path confusion.
Tests exercise actual Requests preparation behind an offline adapter.

All original failures now pass. The final affected-domain run passed **373 checks**,
including 40 new transport cases and opening-intent, reconciliation, lifecycle,
paper, order-identity and replay regressions. Ruff and diff checks passed. Four
lambda transport doubles initially lacked the new header argument; signatures
were corrected without weakening assertions. Future transport-interface changes
should inspect lambda doubles alongside named functions before domain runs.

No venue requests, real credentials, accounts, orders, protected captures or
unrelated workstation tasks were involved. Functional-test duration supports no
benchmark claim. Historical sources/results and research promotion are unchanged.

## Remaining obligations

Version-1 opening intents lack origin/account binding. They must not inherit
whichever credentials are configured after restart. Bind future intents before
exact-order recovery and explicit rearm; preserve unresolved legacy obligations.
Other callers, close intents, independent supervision and exhaustive final review
remain unfinished. This fix is neither a hostile-code sandbox nor a comprehensive
secret-logging/exception-chain audit. Passing tests do not establish enterprise
readiness or permission to deploy capital.
