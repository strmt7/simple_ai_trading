# Explicit persistent loss-policy binding

Signed Binance autonomous capital evaluation now requires an explicitly activated
policy matching scope, execution identity, quote unit, reference and effective
loss settings. Canonical revisions use the existing journal and independent
file/interlock enrollment. [Operator instructions](../../../BINANCE_CAPITAL_POLICY.md)
cover inspection, activation and exact prepared recovery.

Baseline: `48c2df41f0d856d55fdd939223fe45eee724fb8e`.
The [evidence](evidence.json) binds source phases and limitations. `SAFE-02`
remains partial.

Activation requires ownerless Stop and an inspected compare-and-swap head. It
creates no account client, order, session, automatic rearm, second database or
host task. Normal reads never bootstrap state. Prepared first activation needs
exact committed recovery; an uncommitted preparation cannot be promoted. Updates
commit history/head together. Capital reference/unit migration is not invented;
prior losses and the retained cash peak survive revisions and process restart.

Entry-policy denial is separate from owned reduction. Known cash-loss breaches
still reach risk-close handling, and qualified owned automatic exits remain
available when only entry policy is unavailable. Direct close/recovery retain
their independent ownership and UNKNOWN controls. The installed parser and
generated native expert contract expose `capital-policy`; diagnostics are
sanitized and post-commit failures preserve activation uncertainty.

The affected-domain phase passes **373 cases**. Formatting three test calls was
then confirmed by all **70 tests in that module**; unaffected scopes were reused.
Counts overlap and are not added. Coverage includes concurrency, stale heads,
scope/unit/reference/settings mismatch, missing/changed resources, bounded
document rejection, rollback, prepared recovery, fresh contained-process
inspection, actual parser/native registration, native cash and operator recovery.
No real account or market was queried.

The first focused phase passed 33 cases. The first broader phase retained 366
passes and four failures: three old entry fixtures lacked explicit activation;
a genuine integration regression also let entry-policy denial interrupt an
owned automatic exit. Fixtures now activate an explicit synthetic policy before
starting their owner, and managed closing is independent of entry-policy
availability. Original source/log phases are preserved. No admission rule was
weakened to restore those entries.

Universal writer/roundtrip and final-intent policy fencing, Polymarket policy,
account qualification, complete marked/day baselines, external flows, reserves,
model/generation identity, quarantine, rearm, independent supervision and full
native/physical-reboot/backup acceptance remain open. Local hashes do not defeat
coordinated authority rollback. Financial observations, old results and edge
census are unchanged. Activation alone grants no trading permission or capital
guarantee.
