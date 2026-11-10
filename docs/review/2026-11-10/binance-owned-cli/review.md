# Durable owned Binance execution in the CLI

The signed `live` loop previously sent native orders before recording their
positions. Entry, normal exit and emergency exit could therefore lose their
recovery selector or inventory update on interruption. Its local deterministic
client ID did not replace a committed obligation. The autonomous path already
had the required write-ahead intent and execution-scope boundary.

Both paths now use the shared `binance_owned_execution` module. It extracts the
existing native validation, Spot receipts, exact-client-ID recovery and durable
opening/closing flow. Typed results return the persisted position or trade and
the raw acknowledgment to the caller. Existing private autonomous import names
remain available. The pure normalization helpers are AST-identical after the
move, and all other autonomous definitions are unchanged.

Signed CLI entry, normal close and emergency close commit UNKNOWN before native
dispatch and bind the same execution scope through transmission and recovery.
Inventory publication precedes release of the obligation. The old internal paper
helper refuses signed dispatch. Only that helper and `command_live` changed
semantically among the CLI's top-level definitions. Native Spot quantities and
receipts are reused; the CLI no longer manufactures its own post-send position.

A close of 99.9% is no longer treated as complete. A real partial residual stays
in both durable inventory and the local loop, while the unresolved close blocks
blind repetition and new exposure. Separately owned close admission remains
available through the existing ledger rules. Account-unproven legacy positions
are not silently assigned to the current account.

## Verification without repeated suites

The [evidence](evidence.json) supports **217 distinct focused cases** across
explicit lanes. An initial 215-case lane passed 210 existing cases; five new
scenarios stopped at an invalid test-fixture stop geometry. After
correcting only that scenario helper, all five passed, including actual CLI
entry/normal/emergency dispatch, a near-full partial close and abrupt child exit
at the native boundary. Two additional exact helper callers were then migrated
and passed separately. Production bytes were unchanged across these lanes; the
remaining test-definition changes are enumerated and checked. This is not one
final 217-case run or a whole-suite claim.

All three entry/close cases fail specifically at the missing pre-native durable
boundary against the original CLI. The child-exit case verifies a reopened exact
pending opening without a manufactured position; its hidden child exits before
the test returns. Existing signed fixtures now declare real scope and native
acknowledgment semantics instead of bypassing them. A stale synthetic replay
schema label was updated to the current v4 contract, without weakening its gate.
Ruff lint passes; production and changed test definitions were formatted.

Raw logs, source snapshots, actual semantic-navigation receipt and phase hashes
remain private. No native venue request, real credential, order, benchmark, model
fit or protected capture was used. No persistent process, service or task was
installed. Financial counts remain **204 / 65 / 37 / 0 qualified stable edges**.

## Remaining execution boundary

The exact named-call inventory still finds two direct signed writes in the Spot
roundtrip diagnostic. That separate two-leg flow needs complete inventory,
interruption and recovery semantics; it is not a template for new callers.
The calibration transport is also distinct. Literal call-site inspection is not
a complete native-write inventory. Low-level API/private primitives remain
callable, so this is not a universal technical write fence.

Independent session supervision, bounded IPC, generation fencing, persistent
capital policy, terminal partial recovery/rearm and verified legacy migration
remain required. Futures fee/accounting adapters and CLI cash displays retain
their stated estimation limits. Native account/margin/cash qualification and the
whole-tree final review are unfinished; this repair does not qualify returns or
institutional readiness.
