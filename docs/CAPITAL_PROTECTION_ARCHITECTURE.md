# Capital protection and final release verification

Status: implementation roadmap, not a claim of complete protection or live
readiness. Requested September 5, 2026. No live-money authority exists.
Safeguards are core requirements during development, not an afterthought once
edge research ends. They reduce operational risk; they cannot guarantee capital
against gaps, illiquidity, exchange failure, custody loss, or every outage.

## What the current code actually establishes

Reviewed relevant implementations at base commit
`e8e540e0c3237ddd65c89d886a067af59cc5b791`:

| Component | Verified capability | Remaining boundary |
| --- | --- | --- |
| `polymarket_runtime_control.py` | SQLite WAL/FULL durability, integrity checks, lease ownership, ordered submission/Stop interlocks, persistent Pause/Stop | Not an independent OS process supervisor; heartbeat is not proof of useful strategy progress |
| `polymarket_autonomous_runtime.py` | Separate asynchronous safety/model services, decision timeout, gated coordinator, owned-exposure shutdown | Async tasks share a process; a blocked event loop/native call can affect multiple loops |
| `execution_lifecycle.py` | Distinguishes opening from closing; ownership, ledger, reconciliation and endpoint gates | Not proof that every venue write is isolated behind one independent gateway |
| `risk_controls.py` | Deterministic drawdown and consecutive-loss entry rejection | Not a complete cross-process, reboot-tested capital budget or model-decay controller |

The September 5 heartbeat repair prevents a worker from refreshing an expired
or materially future-dated heartbeat into a valid lease. It atomically latches
Stop, retains ownership and old heartbeat, and uses the submission interlock.
This closes a specific recovery defect. It does not implement all rows below.
Neither a controller reopen test nor a normal process exit proves power-loss or
OS-reboot recovery. Historical results remain unchanged.

## Target process and authority boundaries

The subsequent [durable opening checkpoint](review/2026-09-05/binance-durable-opening.md)
now commits UNKNOWN before autonomous Binance submission. Its exact identity,
partial-fill, interruption and entry-only lifecycle checks are implemented;
remaining recovery and caller coverage below are still required. The earlier
after-success-only description refers to the prior write-boundary checkpoint.

The [Binance write-boundary repair](review/2026-09-05/binance-write-boundary.md)
preserves valid client IDs exactly, rejects ambiguous IDs and prevents blind
transport write retries and redirects. Its offline recovery test proves a
POST-to-exact-ID-GET sequence, not durable crash recovery. The autonomous loop
still needs intent persistence before submission and explicit UNKNOWN recovery;
the current after-success position recording is insufficient for that guarantee.

```text
Operator UI / CLI --- same authenticated control contract
                         |
                 supervisor service (user mode)
                   |                 |
           strategy worker jobs   deterministic execution/risk gateway
           research / GPU jobs       |             |
           no order credentials   durable ledger   venue adapters
                                   and policy       |
                                               owned orders/fills
```

Use a user-mode service, not a new privileged kernel driver. A small independent
supervisor must observe progress even if the model process hangs. Training and
strategy workers belong to explicitly owned, terminatable process groups. On
Windows, evaluate Job Objects with kill-on-last-handle-close, no breakaway, and
assignment before allowing worker code to run. Detect unsupported containment;
never silently run an uncontained order-capable worker. Only our child processes
may be controlled; unrelated PC work remains untouched. GPU workers get bounded
resource budgets; saturating hardware is subordinate to safety responsiveness.

The execution gateway, owned-state reconciler and emergency control path must
remain outside the strategy job being terminated. Strategy proposals carry
generation, policy/model identity, sequence and expiry; an old worker cannot
submit after replacement. Credentials belong only to the execution boundary,
with OS access restrictions, not worker IPC, logs, settings or the repository.
Job membership alone is not a credential-security sandbox.

Gate each new exposure on deterministic limits and current reconciled state.
Use one authoritative writer or provably fenced single-writer generation;
bounded queues, deadlines, admission rate limits and backpressure must prevent
order storms. Keep a separately bounded cancel/reduce path available when an
entry model is quarantined. Never let model output override a safety gate.

## Durable policy and state contract

Keep versioned, strictly validated policy separate from transactional mutable
state. Reuse existing ledgers and interlocks through adapters; do not introduce
an unrelated second source of truth. Policy changes require validation, audit
and explicit activation. Unknown fields, missing budgets, bad schema, corrupt
state, missing storage or identity mismatch must leave new exposure disabled.
Do not guess the user's eventual capital allocation or loss tolerance.

Policy must cover account/venue/asset scope, total and per-strategy risk budgets,
gross/net exposure, concentration, leverage, margin/liquidation headroom,
order size/notional/rate, price collars, daily loss and drawdown, loss streaks,
rejection/duplicate limits, market-data freshness, clock tolerance, heartbeat
and useful-progress deadlines, IPC/disk/memory budgets, cancellation/exit policy,
model-health evidence, and explicit restart/rearm authorization. Live defaults
remain disabled. Currency and percentage units must be unambiguous.

Persist at least policy/schema hashes, model version, generation/fencing token,
Stop reason and epoch, owned order/lot and intent IDs, unresolved requests,
balances and reconciled fills, session/day loss state, high-water mark,
reservations and pending exposure, and model quarantine state. Include realized
and valid conservatively marked unrealized losses without double counting;
unknown marks block fresh risk, not manufacture zero loss. Define UTC day
rollover and external cash flows explicitly; reboot cannot reset loss budgets.

Record intent durably before transmission. A timeout after venue acceptance is
UNKNOWN, not rejection: reconcile the exact client/order identity before any
retry. Restarts enter RECOVERY/PAUSED, check storage and owned venue state, and
require explicit rearming; do not blindly replay orders, replace missing state
with an empty ledger, reset a kill latch, or infer flatness from no local process.
Use transactions and durability checks; a hash alone is not protection against
an authorized attacker rewriting both data and hash. Validate backup/restore
and rollback detection without ever resetting actual owned-state obligations.

## Self-diagnosis and model degradation

Liveness must include independently observed completed-work progress, source
sequence/freshness, reconciliation progress and bounded execution latency, not
only an always-running heartbeat thread. Use monotonic deadlines within a boot;
persist boot/generation identity rather than compare monotonic clocks across
reboots. Repeated errors, stale feeds, invalid prices, unknown acknowledgements,
clock discontinuities, storage failure or loss limits latch new-risk shutdown.

Model monitoring is separate from deterministic hard loss stops. Prospectively
define calibration, after-cost decision value, fill/slippage deterioration,
feature/label drift, missing-data and out-of-distribution checks with sample
coverage, delayed-label handling, dependence and repeated-testing controls.
An unfavorable short streak alone does not statistically prove model failure;
it can still trigger a pre-agreed capital limit. Missing evidence is not health.

Quarantine blocks entries while owned-risk management continues. Retraining is
a separate candidate workflow, never automatic permission to resume. Compare
against the existing model and a feasible non-ML baseline on untouched forward
periods after all costs; include abstention, tail loss and capital usage, not
only classification accuracy. Promote only with affirmative evidence and
recorded authorization. Preserve previous models and results for reproduction.

## Delivery sequence and completion evidence

1. Repair demonstrated recovery defects and retain small regression proofs.
   Current: expired-heartbeat resurrection repaired; broader recovery unproved.
2. Inventory every order-capable caller and its actual authority/ownership path;
   define shared policy and gateway interfaces without duplicating ledgers.
3. Implement persistent policy/recovery and independent process containment,
   progress supervision, generation fencing and task-owned termination. Exercise
   actual supported OS behavior locally in isolated tests; no machine reboot,
   service installation or unrelated-process mutation during ordinary R&D.
4. Integrate both venue adapters and CLI/native controls. Verify UNKNOWN states,
   partial fills, owned-only cancellation, scope isolation and restart recovery
   in offline simulation and separately bounded non-live campaigns.
5. Integrate model-health quarantine and causal retraining/selection. Expand
   datasets only when coverage/labels answer a defined question; use the verified
   faster compatible GPU backend for substantial fits, preserving precision and
   CPU parity. Profile real bottlenecks, not repeated synthetic speed grids.
6. After the main revamp is implemented and integrated, perform the user's
   mandatory final whole-codebase, line-by-line review using the requested
   latest model available to the session; record the actual model identity
   rather than claiming a model switch that did not occur. Review, refactor and
   upgrade all code where justified by correctness, architecture, performance,
   maintainability or dependency evidence. No line may remain unchecked. Review
   historical implementations without rewriting frozen results or consumed
   sources. A forced cosmetic rewrite of every line is not this requirement.
7. After those changes, freeze a release candidate and perform the extensive
   final bug hunt below. Both final phases are mandatory and neither is satisfied
   by earlier focused tests or an inventory. Later changes invalidate the
   affected review coverage and require renewed verification before release.

## Mandatory final bug-hunting campaign

Apply the [shared codebase consistency standard](../CONTRIBUTING.md#codebase-consistency-standard)
throughout the final review/refactor. Lead with semantic understanding and
architectural judgment: trace intent, financial invariants, ownership and
failure propagation before deciding what to change. Review comments, naming,
types, logging, errors and interface language as part of each file. Automated
checks corroborate this work; mechanical uniformity and passing tests do not
substitute for it. Record justified differences rather than force unlike
contracts into a common abstraction.

The user's final-pass instruction is explicit: be exhaustive and cut no corners.
Maintain a hash-bound file and line-range coverage ledger for the complete release tree:
code, configuration, build/dependency files, vendored material, scripts, skills,
plans, documentation and interfaces. Distinguish semantic review, generated
verification, historical evidence audit and unreviewed files. Preserve historical
results; corrections are separate, source-bound adjudications. A syntax scan or
passing CI run cannot mark a file semantically reviewed.

Perform independent passes for financial accounting/payoff/fee correctness;
causal data and model validation; all write/ownership and authorization paths;
security and dependencies; concurrency/resource handling; crash/restart and
storage recovery; API/schema changes; and CLI/native rendering/action parity.
Trace end-to-end failure paths and cross-component interactions, not just
isolated functions. Use the applicable repository/security skills when those
phases actually begin; no claim of a completed security scan is made here.

Inject worker/event-loop/native/GPU hangs, worker and supervisor termination,
split-brain/stale generations, interrupted writes, disk-full/corrupt/rolled-back
state, simulated reboot/clock jumps, duplicated/reordered/lost messages, stale
market data, acceptance-before-timeout, partial/rejected cancellations, partial
fills and venue outages. Verify that unknown exposure stays unknown, no foreign
state is altered, no duplicate risk is opened, and restart cannot clear latches.
Actual reboot/power-loss and off-host recovery tests need an isolated environment
and controlled operation; simulations alone must not be reported as equivalent.

Every finding needs severity, exact source/version, reproduction, affected
paths, root cause, fix and regression evidence, or an explicit unresolved
disposition. Re-review fixes for new defects. Re-freeze changed components and
repeat impacted passes; perform the full integration/release verification at
the final candidate, not after every small edit. Any unresolved release-blocking
finding prevents real-money readiness. Zero known defects is not proof of zero
possible defects. Do not declare the whole revamp complete with an uncovered
file or an unsupported safety claim hidden behind aggregate test counts.

## External guidance and limitations

Primary sources accessed September 5, 2026; retained web-tool extractions and
selection receipts are under `review/2026-09-05/supervisor-source-*.json`.
They are rendered tool evidence, not original HTTP bodies. No source below
establishes this platform's compliance, venue eligibility or current profitability.

- [FINRA Notice 15-09](https://www.finra.org/rules-guidance/notices/15-09),
  March 26, 2015: supports change controls, quick disablement, limited pilots,
  independent testing, monitoring and reconciliation. It addresses securities
  firms and explicitly does not promise prevention of every failure. The
  architecture above is our engineering interpretation, not a legal opinion.
- [Microsoft Job Objects](https://learn.microsoft.com/lb-lu/windows/win32/procthread/job-objects),
  page updated July 14, 2025: supports process grouping and termination when the
  last handle closes with the kill-on-close flag. Handle inheritance, assignment
  failure and breakaway behavior require explicit implementation/tests.
- [CME Kill Switch](https://www.cmegroup.com/tools-information/webhelp/globex-credit-controls/Content/Kill-Switch.html),
  publication date not displayed: documents cancellation/blocking scope and
  exceptions, including closed-market periods and mass quotes. It proves no
  equivalent Binance or Polymarket feature. Source-bind each product's native
  protection and exact acknowledgement semantics before relying on it.

Terminating a local process does not cancel exchange orders. A local supervisor
cannot act while the entire PC is off. Eventual deployment therefore also needs
product-qualified venue-native protections and an independently hosted monitor
with carefully scoped authority. Network outages, gaps and venue failure can
still prevent closing or exceed a target loss bound; never describe these
controls as guaranteed capital preservation.
