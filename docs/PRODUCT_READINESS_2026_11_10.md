# Product assessment and next development round

Assessed development commit: `851e732cf85c814df24cca59340670438ec04088`.
The [evidence](review/2026-11-10/product-assessment/evidence.json) binds the
source-navigation results, current ledger checks, retained failure reproduction
and primary dependency-source review. This is a gap assessment, not a complete
code audit, fresh security scan or release approval.

The product is not yet a qualified stable-release candidate. Important local
mechanisms are implemented and verified, but their end-to-end integration and
several capital-critical failure paths remain open. Financial evidence records
205 terminal observations, 65 hypotheses and 37 scoped mechanisms, with **zero
qualified stable edges**. Neither a completion percentage nor a release date is
defensible from those counts. The user alone decides whether to release; no
release, draft, tag or release workflow is authorized by this plan.

## What is established and what remains

| Area | Established evidence | Required next evidence |
| --- | --- | --- |
| Dependencies | Three compatible upgrades; isolated lock/package/import/CLI validation and 643 focused cases pass | Exact final integrated matrix and supported clean-host profiles; this lane is not the full suite |
| Security status | Captured API inventory lists zero open alerts | Scanner freshness, quota and complete coverage remain unknown; skip explicit exhausted-quota work without retries |
| Persistent economics | Payoff/cash primitives, retained primary sources and negative studies exist | Recurring executable surplus after all costs, financing and total capital, untouched chronological evaluation, capacity and explicit abstention |
| Execution | Durable owned intents, scoped receipts, exact fee accounting and shared Stop/lease machinery exist | Native roundtrip integration, complete write-caller fencing, controlled legacy migration and partial/UNKNOWN recovery |
| Storage and failure recovery | Missing-record repair preserves existing namespace obligations | Whole-database replacement/loss detection, independent enrollment, explicit recovery/rearm and crash/reboot acceptance |
| Supervision | Owned process containment exists for selected development/model tools | Independent session-owned trading supervisor and credential-bearing gateway, bounded IPC/native calls, generation fencing and closing during failures |
| Models and acceleration | Causal/selection controls and risk-reducing LLM mechanisms exist in stated scopes | Qualified causal economic targets, full search provenance and held-out incremental utility; actual compatible backend and useful timing before expansion |
| Operator and distribution | One source-bound operator-view checkpoint and isolated installer exist | Complete backend/UI integration, diagnostics, interaction/accessibility, clean-host packaging and a measured commercial protection boundary |
| Final qualification | Narrow repairs have source-bound regressions | Revision/line-bound whole-repository review, unresolved integration failures, exhaustive final bug hunt and final acceptance matrix |

The November 9 full-suite follow-up retained 86 failures at its source phase.
That is historical evidence, not a current failure count. Preserve its logs and
resolve the underlying source/fixture/production distinctions; do not weaken
financial guards or rewrite old results to obtain a green count.

## Prioritized deliverables

1. **Close the reproduced control-storage loss path.** The retained reproduction
   renamed an owned control database after Stop epoch 1; opening its former path
   created fresh stopped state at epoch 0 and allowed a new lease. Preserve that
   original database and test evidence. Specify independent enrollment and scope
   identity, noncreating normal open, explicit interrupted-enrollment handling,
   existing-schema migration and reconciled rearm before implementing the repair.
   Recovery must preserve owned closing and UNKNOWN obligations. Filesystem loss,
   rollback and malicious alteration are different threats; one digest is not
   proof against all three.
2. **Integrate one complete testnet operation.** Once independent storage and
   dispatch admission are sound, connect the Spot roundtrip to durable intent,
   actual scope/balance/fee-asset admission and Stop authority. Prove first-leg
   uncertainty, partial receipts, restart, second-leg admission and owned
   reduction without duplicate submission. Inventory remaining native writers
   and fence them explicitly; creating an adapter does not upgrade its callers.
3. **Advance an informative structural-economic question.** First inspect the
   complete canonical retry row and exact source contracts. Reuse admitted
   primary evidence; do not resample consumed options, funding panels, daily
   event quotes or promotional pulses until a positive result appears. Prefer
   recurring after-cost cash and executable settlement relationships. Freeze
   the next eligible population, costs, cash/collateral and exceptional-settlement
   assumptions, chronological roles and terminal consequence before access.
   A source-completeness failure is not an economic rejection. No currently
   satisfied new access trigger is asserted by this planning document.
4. **Make model expansion conditional on economic information.** Define the
   objective as incremental net utility against feasible abstention, liquidation
   or an executable hedge, including inventory, transition costs and capital.
   The current hourly labels explicitly describe independently opened fixed-base
   counterfactuals, not a continuous hedged policy. Resolve clock/availability,
   paired quantity/cost and target coverage before more fitting. Record every
   candidate and selection decision; choose a simple economic baseline before
   larger temporal/multimodel experiments. LLMs remain veto/downsize only.
5. **Complete product qualification after integration.** Implement session-only
   supervision and durable loss/model-health policy, connect truthful operator
   states, then perform the required whole-tree semantic review and exhaustive
   bug hunt. Run the final supported matrix once against the exact proposed
   release tree. A validation-only packaging path must be separated from release
   creation: the current release composite creates a draft even when publishing
   is false. Do not dispatch it as a substitute for validation.

Each deliverable has a stop condition and retained evidence; do not serialize
all research behind unavailable scanner quota. Public/offline research and
capital repairs can proceed independently where their own boundaries permit.

## Reuse research before adding machinery

The already hash-verified filelock 4.2.10 wheel provides public native-lock
options `preserve_lock_file` and `on_acquired`. Its Windows implementation checks
reparse status through the opened handle, and the acquisition callback receives
the owned descriptor. These are useful design references for identity-bound
admission. They do not supply an application enrollment authority: normal lock
opening can create a missing file, and retaining a filename after release does
not persist Stop policy or prevent later replacement. Strict soft claims are
not equivalent to the trading system's durable business obligations.

Evaluate public APIs and the existing tested native lock before adding custom
OS machinery. Filelock is currently an optional dependency; silently importing
it into the stdlib-only runtime core would change installation guarantees.
Do not depend on its private process-identity helpers or copy a large third-party
implementation. The parked enrollment prototype remains private, unintegrated
and unqualified. No dependency source was executed to test enrollment authority.

## Resource and host acceptance

Apply the strengthened [host lifecycle rules](AGENT_WORKFLOWS.md#host-lifecycle-and-teardown)
throughout development. Use bounded, hidden, session-owned process trees and
exception-safe resource cleanup; verify repeated/long-lived allocations when
that boundary changes. Keep unrelated tasks, services and GPU contexts intact.
Do not introduce a persistent supervisor service, periodic restart or cleanup
job. Preserve inert evidence and reusable isolated environments.

The candidate Windows Torch build reports neither CUDA nor HIP. The alternate
DirectML profile was not qualified in this round. Package installation is not
GPU acceleration; expand only after identifying the actual backend, useful
workload and correctness/economic baseline. Avoid heavy timing during material
contention, without deferring unrelated functional development.

No further market, account, training, benchmark or security-scanner request was
needed for this assessment. The two required ledger/frontier source checks pass
against the published checkpoint. This plan adds no release authority and no
stable-edge promotion.
