# Binance write identity and transport boundary

This forward engineering checkpoint closes three reproduced failure paths; it
does not establish a profitable strategy or complete execution safety. Source
identities and verification totals are in [the evidence record](binance-write-boundary.json).
No venue, account, order or protected capture was accessed for these tests.

The previous submission path silently stripped and truncated caller IDs to 36
characters, while lookup used a differently normalized identity. We now reject
invalid supplied IDs before HTTP, paper output or leverage changes, preserving
valid values exactly. The 36-character ceiling is the existing local contract,
not a new claim about every exchange-supported character.

The shared transport previously retried writes on timeouts, selected HTTP/API
errors and malformed JSON. A response can be lost after acceptance, so transport
retry is not evidence that resubmission is safe. Non-GET calls now make one
attempt. The owning autonomous open workflow's existing exact-ID lookup is
exercised through the real transport with an offline POST-to-GET sequence; an
unknown lookup never causes a second POST. GET retry behavior remains unchanged.

Requests redirects were a separate replay path. The default session now permits
zero redirects. Offline adapters exercise its actual 301/302/303/307/308 handling
and prove that no follow-up route receives a request. This does not qualify
externally replaced sessions or custom retry adapters.

Verification: 270 affected-domain tests plus 9 selected CLI integration tests
pass (279 distinct); changed-file Ruff and whitespace checks pass. The domain
run covers API, identity/replay, autonomous execution, positions, reconciliation
and lifecycle tests. The CLI selection covers order fills, entry notional,
reduce-only forwarding, roundtrip validation and persistent order errors.
These are regression proofs, not the user's final exhaustive review.

The first identity fixture omitted required empty credential constructor
arguments and failed before reaching the behavior under review. After that
fixture correction, the pre-fix behavior failed the intended rejection test.
The write-timeout and redirect regressions also failed before their fixes.

Next we must persist intent before transmission: the autonomous loop currently
records its position only after a successful response. An accepted write plus
crash or inconclusive lookup must leave a durable UNKNOWN obligation. The CLI
roundtrip also needs exact intent IDs. Independent worker containment, response
identity verification and reboot-safe policy integration remain unfinished.
