# Enrolled Binance loss policy

Signed Binance autonomous capital evaluation requires an explicitly activated loss
policy. Paper mode grants no native account authority and keeps its existing
configuration behavior. Activation is necessary for this admission scope; it is
not account qualification, final native-write permission or rearm.

Policy history, head and storage identity use the existing execution journal.
A small independent enrollment record binds them to its file identity and the
existing controller interlocks. No second risk database, service or scheduler is
created. Normal reads never provision missing state or clear cash/UNKNOWN history.

## Inspect and activate offline

Use the installed CLI or its generated expert entry under Risk → Owned close
recovery. No credentials or venue client are loaded. Runtime and execution storage
must already be enrolled:

```text
simple-ai-trading capital-policy --data-dir "<existing-owned-directory>" --json
```

Inspection returns the execution scope digest/store ID, current policy/head,
revision and enrollment state. Revision zero with a null policy grants no entry
permission. Create a private JSON document with exactly these fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | `binance-capital-policy-v1` |
| `execution_scope_sha256` | Exact inspected original scope digest |
| `execution_store_id` | Exact inspected execution identity |
| `quote_asset` | Exact uppercase quote unit of the supported symbol |
| `reference_cash_quote` | Explicit positive decimal string in quote units |
| `maximum_daily_loss_fraction` | Positive decimal string ≤ 1; effective daily loss limit |
| `maximum_lifetime_loss_fraction` | Positive decimal string ≤ 1; cumulative `max_session_loss_pct` limit |
| `maximum_settled_peak_drawdown_fraction` | Positive decimal string ≤ 1; effective `max_drawdown_limit` |
| `maximum_consecutive_losses` | Exact integer 1–100; effective loss-streak limit |

Choose capital and limits explicitly; the command does not infer balances or
loss tolerance. Fractions divide by approved risk capital. Drawdown uses the
[retained settled-cash definition](review/2026-11-11/cash-peak-risk/review.md),
not complete marked/account NAV. Duplicate/unknown fields and invalid types
reject. Decimal values are canonicalized. Documents are limited to 8,192 bytes;
history is limited to 4,096 revisions, after which a qualified migration is needed.

With ownerless, paused, Stop-latched control:

```text
simple-ai-trading capital-policy --data-dir "<existing-owned-directory>" --action activate --policy "<private-policy.json>" --expected-head-sha256 "<inspected-head>" --json
```

Scope, store ID and head must still match. First activation uses prepared/ready
enrollment around a SQLite transaction. Updates append a hash-linked revision
and replace the head atomically. Repeating the same policy against the current
head is a no-op; stale heads reject. Reference/unit changes require a future
qualified external-cash-flow migration. Explicit limit revisions never delete
cash history or reset its peak. Activation never clears Stop, abandons ownership,
starts a session, places an order or authorizes rearm.

Signed autonomous evaluations must match scope, quote unit, reference and all
effective loss settings. The capital decision records the matching policy hash.
Optional `--runtime-control`, `--runtime-enrollment`, `--execution-enrollment`
and `--policy-enrollment` select existing resources for offline administration.
The current autonomous integration uses its existing data-directory-derived
runtime/storage paths; alternate session-resource wiring remains separate work.

## Interrupted activation

An error after activation entry reports an unknown outcome. Inspect before
another action: a committed policy may exist. Inspection can expose a prepared
enrollment's `policy_store_id` and `enrollment_record_sha256`. Normal admission
rejects prepared state. Finish only an exact committed preparation:

```text
simple-ai-trading capital-policy --data-dir "<existing-owned-directory>" --action recover --expected-store-id "<prepared-policy-id>" --expected-record-sha256 "<prepared-record-digest>" --json
```

Recovery verifies the original storage, committed chain and exact record. An
uncommitted preparation cannot be promoted. Preserve failed/ambiguous originals;
no command deletes a marker, recreates lost history or fabricates zero losses.

Entry-policy denial does not veto qualified owned automatic exits. Known cash-loss
breaches still reach their existing risk-close path. Ownership, quantity, native
receipts, reconciliation and Stop remain controlling for every close.

## Scope limits

This phase binds signed autonomous loss-policy evaluation. Universal direct-writer
and roundtrip fencing, Polymarket policy, final-intent policy binding, account UID,
complete native marked/day baselines, external flows, reservations, price/margin
limits, model/generation identity, quarantine, rearm and independent supervision
remain unfinished. Physical reboot, full backup/restore, alternate resource
wiring and complete native operations still need acceptance. Ordinary resource
loss/replacement and partial chain damage reject; local identities/hashes do not
defeat coordinated rollback/rewrite of all authority material.

The [review](review/2026-11-11/capital-policy/review.md) records actual evidence.
No financial edge, capital guarantee or live-money authority follows.
