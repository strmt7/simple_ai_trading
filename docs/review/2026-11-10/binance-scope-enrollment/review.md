# Fresh credentials cannot inherit legacy live inventory

An isolated reproduction at the parent revision created a live position through
`PositionsStore`, leaving an empty opening journal with no execution scope.
Preparing an unrelated opening with a new key bound that journal to the new
scope. Closing preparation then accepted the old lot under that scope. No venue,
real credentials or user inventory was used. Earlier scope checks protected an
already-bound journal but did not establish safe first enrollment; the previous
checkpoint's broad legacy-ownership wording must be read with this correction.
Historical results and their original source bindings remain unchanged.

## Repaired admission

Opening preparation now participates in the existing paired-position writer
transaction. Before first scope enrollment it rejects existing opening records,
nonempty unscoped auxiliary tables, live or unclassified open/closed history,
and history rejected by the store's strict parser. Every retained row must
explicitly contain boolean `dry_run: true`; dataclass defaults cannot classify
legacy history as paper. Empty storage and parseable explicit paper-only history
can enroll. Original JSON bytes are retained without rewriting either file.

The exact open/closed byte pair, scope, namespace version 3 and UNKNOWN intent
commit together. The unchanged-pair operation shares the existing replacement
record rather than introducing a second inventory authority. Cancellation or
failed intent insertion rolls back enrollment. The existing redo mechanism still
finishes an interrupted committed replacement before participating access.

Version 3 identifies enrollment under this rule. Missing scope or paired metadata
does not recreate a fresh namespace. Every subsequent opening checks the enrolled
pair before dispatch, including changes to a previously absent file. A competing
initializer rechecks namespace version after acquiring the writer; it cannot
downgrade the winning enrollment to version 2. Request payloads still use
`intent_version: 2`; this is a namespace change, not a wire-format change.

Already-bound version-2 journals retain existing scope-bound recovery,
acknowledgement and owned-close paths. New exposure requires separately verified
enrollment migration. No automatic migration based only on current credentials
is supplied. This preserves reduction while refusing to treat an old binding as
proof that its original inventory was safely enrolled.

## Verification and limits

- The first affected-domain lane passed 154 cases. A separate recovery,
  inventory, cash and lifecycle lane passed 470 cases, including one strengthened
  competing-initializer case from the first lane. Counts overlap.
- After the competing-initializer test changed and three direct enrollment-error
  cases were added, all 27 cases in the final enrollment module passed. Its
  exact source and JUnit receipt are retained; unchanged domain tests were not
  needlessly rerun. Production bytes stayed identical across these lanes.
- Three regression cases fail against both original production modules at the
  missing rejection: unscoped live inventory, enrolled-pair drift and nonempty
  auxiliary history. The controller restores exact working bytes in `finally`.
- Checks include explicit/missing/nonboolean classification, both ledger files,
  malformed history, exact paper-byte preservation, SQL failure, cancellation,
  forced simultaneous first connections, lost metadata and legacy owned closing.

The [source-bound evidence](evidence.json) retains log hashes and phase limits.
These checks do not establish whole-suite acceptance, native account identity,
account-key rotation, controlled legacy migration, adversarial database rollback,
whole-database deletion protection, physical power-loss or reboot recovery.
Low-level writers and direct diagnostic paths still need complete gateway fencing;
the independent supervisor and persistent risk policy remain open. No market
population, protected capture, account, order or credential was accessed.
Financial counts remain **204 / 65 / 37 / 0 qualified stable edges**.

The separately admitted [Spot quote-budget source](../spot-quote-budget-source/review.md)
supports further two-leg design; it does not repair or authorize that execution.
