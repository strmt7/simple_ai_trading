# Ordinary Binance opening policy through SDK dispatch

Base revision: `84315ef5da743ab5c18819c67dc4561ad8a95619`. The
[evidence](evidence.json) binds the exact changed source and retained test phases.
This is a scoped safety integration, not financial qualification or a complete
independent trading gateway. The existing full harness goal stays active.

## Problem and resulting behavior

An earlier strategy loss check could become obsolete before an ordinary owned
order was sent. The durable gateway did not itself require the activated policy
and a current strict open/closed snapshot at its UNKNOWN transaction. SDK
preflight and throttling also occurred after the gateway's earlier checks.

The shared supplied-snapshot evaluator now serves autonomous evaluation and
ordinary durable opening admission. Autonomous evaluation freezes the effective
limits/reference before policy inspection; later mutation cannot relax them.
The gateway uses the current enrolled policy and decodes both position files
inside the same SQLite writer transaction before UNKNOWN. Its private position
copy and original request retain the accepted policy-head digest. The caller's
position remains unchanged, and blank new fields preserve legacy v2 request bytes.

Before transport, the gateway checks that precisely the original UNKNOWN request
remains, with no conflicting unresolved operation. It rechecks policy, cash
state, storage and opening authority. A scoped SDK callback repeats admission
after throttling and before signing/HTTP dispatch for order and futures leverage
writes. Callback denial is outside HTTP error/retry handling; it cannot trigger
order replay or fallback recovery reads.

Owned closes and exact-order reconciliation do not receive the opening callback.
The close tests include a missing policy marker, while retaining ownership and
quantity requirements. After any post-prepare denial, UNKNOWN is preserved.
No policy, database, enrollment, account, lease or rearm is implicitly created.

## Verification and retained failures

The final acceptance is a source-bound composite: **1,070 unchanged passing
non-CLI cases** from the expanded domain phase plus **188 passing final CLI
cases**. Replacing that entire earlier CLI module yields **1,258 distinct passing
cases**, with no failures, errors or skips in the accepted composite. This is not
a single full-suite run. All accepted source hashes were checked against current
bytes; complete source phases, stdout, stderr and JUnit remain private.

The earlier phases are preserved in evidence: 828 cases with 143 failures,
978 with 11, 983 passing, and 1,257 with six CLI setup failures. Their failing
policy setup, callback signatures and policy-stamped equality expectations were
corrected explicitly. The final CLI lane adds a missing-policy denial control;
it does not auto-activate production policy or weaken admission. The initial
missing-marker error was corrected to the gateway's explicit admission error.

The new regression module covers loss arriving after strategy approval or intent
preparation, missing/changed policy, deleted/rewritten UNKNOWN, quote mismatch,
underflow, legacy request bytes, wrong-store transactions, immutable effective
limits and actual SDK admission after throttle. Futures cases distinguish denial
before leverage from denial after leverage but before the order. Synthetic
transports make zero account/order requests. Ruff check and format-check pass
for the 20 changed Python files. Other suites and native operation remain unrun.

## Limits and next work

The supplied entry mark is not independently fresh native NAV. Native account
UID, fee/cash/filter/margin qualification, paired/alternate policy callers,
Polymarket policy, complete source/model/gateway-generation and intent expiry,
full marked/day/external-flow state, reservations, quarantine and controlled
rearm remain open. UUIDs and chains do not defeat coordinated rollback or
malicious in-process code. A leverage write may have completed before a later
order denial; no automatic rollback follows.

SAFE-02 and M2 remain partially complete. Prepare SAFE-01's independent
session-owned supervisor/gateway slice from accepted interfaces; it need not
wait for every advanced field or UI form. Eligible informative financial work
retains priority. No schedules, services, releases or live-money authority are
created. The recorded financial census remains 206 observations, 65 hypotheses,
37 historical scoped mechanisms and **0 qualified stable edges**.
