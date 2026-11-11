# Expired model decisions cannot regain authority

The actual Polymarket supervisor previously kept a timed-out task for a later
call to `_decision`. In the retained synthetic reproduction, that later call
admitted the old result under a different market context and invoked its close
callback once. No new model evaluation occurred between them. This demonstrates
an authority-lifetime bug in the provider interface, not a real trade or a claim
that either built-in profile emits that instruction. The
[evidence](evidence.json) binds the original source and exact verification.

One invocation now owns one monotonic deadline and one result admission. A
timeout or canceled waiter abandons that result permanently. The provider still
occupies the single pending slot until it finishes, so timing out repeatedly
does not start overlapping work. Completion discards abandoned output and
observes late exceptions. Pause and Stop invalidate the local decision generation;
Resume cannot revive an in-flight decision from before the Pause. Existing
operator, reconciliation and deterministic owned-closing paths remain separate.
No additional venue or ledger query was added.

Four regression cases cover a late close at market rollover, Pause/Resume,
Stop, and waiter cancellation with a late provider error. The existing
nonoverlap case still passes. The affected supervisor, live CLI and control
modules pass 104 cases with exact source hashes before and after. Ruff passes.
Synthetic provider releases are bounded; all verification helpers terminated.
No real venue account, order, protected partial, fit or financial observation
was involved.

## Process boundary still required

The [Round 16 follow-up](../../2026-11-11/round16-worker/review.md) now isolates
the promoted fifteen-minute numerical predictor and reaps it before owned
recovery. The paragraphs below describe the source revision verified in
this report.

The live CLI builds Round 21's five-minute public data service and scorer or
Round 16's fifteen-minute flow feed and predictor in the process that owns the
authenticated coordinator and lease. The supervisor's `asyncio.to_thread`
deadline governs delivery of model output; it cannot terminate a frozen thread.
Round 21 also scores in the public data service, so moving only the decision
callback would leave part of the model work in the controller process. The
existing foundation worker is used by the benchmark, not either promoted
runtime stack, and its current IPC cannot be assumed suitable for trading.

Next integrate a session-owned public data and forecast worker into an actual
profile. Send bounded, timestamped, generation-bound forecast packets; keep
portfolio state, proposal construction, credentials, deterministic limits and
native dispatch in the authority boundary. Verify hard worker termination and
Stop/owned reduction with a hung child before claiming independent supervision.
This patch does not qualify cross-process fencing, native hard deadlines, current
portfolio revision at dispatch, account/capital safety or profitability.
