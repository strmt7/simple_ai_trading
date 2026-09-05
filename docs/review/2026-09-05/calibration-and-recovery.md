# Calibration efficiency and durable recovery checkpoint

This advances the full platform revamp independently of account-qualified edge
research. It does not retrain a model or establish a profitable strategy.
Source hashes and verification counts are in
[the checkpoint record](calibration-and-recovery.json). Original market/model
results and protected capture boundaries are unchanged.

## Implemented and verified

Classification threshold calibration called full model inference once per row
per threshold. The training suite uses 61 thresholds. The new pure-stdlib
`ThresholdCounts` scores deterministic base models once, sorts class scores,
and obtains exact confusion counts by binary search. It preserves grid
arithmetic, first-score tie selection, inversion, special numeric comparisons
and custom prediction fallback. In the 60-row regression, probability calls
fall from 3,660 to 60 with exactly the same selected threshold. This is not a
61-fold total training speed claim: no timing benchmark or GPU experiment ran.
Backtest-profit calibration is a separate path, not the target of this fix.

The runtime heartbeat could renew an expired lease before the next opening
check. Four new assertions reproduced this for expiry and backward clock
movement, with and without Pause. The repair latches durable Stop instead of
renewing, preserving lease ownership and the previous heartbeat. The transition
uses the same interlock as final submission; it cannot retroactively undo an
in-flight order. The existing async control service observes failure and requests
shutdown. Tests cover controller reopen, a separate Python interpreter, clock
correction, boundary age, foreign lease, ordering and simulated disk-write failure.
An unavailable disk cannot promise a persisted latch; the unchanged expired
lease still blocks new exposure at the tested clock. Whole-host clock/reboot
recovery requires the planned boot/generation and monotonic supervision layer.

There are 153 passing model/training tests and one optional dependency skip;
84 distinct runtime/risk/reconciliation checks pass, plus the current publication
reconstruction check. Ruff and diff whitespace checks pass. This is scoped
verification, not the final complete-repository campaign.

## Mandatory remaining architecture and review work

The [capital-protection roadmap](../../CAPITAL_PROTECTION_ARCHITECTURE.md)
records process isolation, persistent policy/loss budgets, recovery and model
quarantine as implementation requirements. An async class named supervisor is
not independent process containment. Neither the new Stop fix nor SQLite alone
proves reboot safety, remote cancellation, flat exposure or capital preservation.

After the main revamp, the user requires a final whole-codebase line-by-line
review/refactor/upgrade, followed by exhaustive bug hunting. Track reviewed
revision and line coverage; later code changes reopen affected coverage. Do not
substitute inventories, syntax checks, CI counts or this checkpoint for that work.

Primary guidance was opened for FINRA algorithmic controls, Microsoft Job
Objects and CME Kill Switch semantics. The source-audit skill kept direct facts
separate from our architecture choices and venue-specific unproved capabilities.
Three selected URLs were opened, followed by two tool-reference line-window
reads because the first output omitted the Microsoft/CME article bodies. The
plan's three-open ceiling did not budget those extra tool calls: retain this
process deviation; do not portray the five tool opens as three calls or as
immutable original HTTP captures. No financial conclusion depends on them.

The GitHub check found zero open PRs and zero open Dependabot alerts. The full
reachable-main identity audit found 656 older AI-identity violations among 2,328
commits; the latest five commits use the required identity. The contributor
response also retains invalid legacy identities. These are shared historical
violations, not merely an assumed stale cache. No shared history was rewritten.
Fourteen remote PR-head refs exist; their identities were not separately audited
in this checkpoint. New commits must use command-scoped `AI agent <>` author
and committer; this audit does not certify historical identity compliance.

The retained `official-change-check/` files predate the broader engineering
resumption. Their no-trigger observation is preserved, not rewritten into a
global claim that useful development is blocked. Avoid repeating those searches.
