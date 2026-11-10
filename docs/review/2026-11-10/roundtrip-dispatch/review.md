# Write-ahead Spot roundtrip dispatch

The new `SpotRoundtripExecution` gateway connects the existing paired journal
to exact scoped native transports and the enrolled Binance runtime interlock.
The caller supplies an explicit asset allocation, existing control enrollment
and runtime lease. The gateway creates no process, thread, service or schedule,
does not acquire/release authority, and borrows its caller-owned HTTP client.
Existing agent rules require session-owned bounded helpers, exception-safe
resource release and task-owned teardown throughout development.

## Admission and dispatch

Before the first order, both planned fee sides must qualify. The gateway checks
current scoped account trading permission and exact free balances against the
explicit allocation, then reads the exact scoped commission schedule. Account
availability is a point-in-time observation, not ownership or an atomic native
reservation. The received-asset fee route is supported; an eligible external
payment route rejects before the first order. Even an explicit third-asset
allowance does not prove an FX bound or enforce a native fee cap.

The monotonic preflight-age budget and execution scope are rechecked inside the
final dispatch guard, including after durable preparation. An expired decision
after preparation leaves UNKNOWN and sends nothing. Freshness is not clock
accuracy, future fee stability or a hard process-level native-call deadline.

The first leg commits `FIRST_UNKNOWN` before its single native write. The
second leg first validates its exact principal against net first-leg receipts;
larger existing wallet balances cannot increase it. It commits `SECOND_UNKNOWN`
under the final guard before transmission. Both quantities and client IDs match
their journal bindings. Quote BUY uses quote units; SELL uses the existing
adapter's exactly representable eight-decimal base quantity. Native failures,
invalid receipts and cancellation never authorize automatic resubmission.

A receipt-owned second SELL uses the enrolled closing lock and can proceed
after opening Stop or loss of the controller database. Losing that closing lock
rejects without recreating it. A sell-buy second BUY remains an acquisition and
requires the live opening lease. Journal scope/revision checks reject competing
second submissions before another native write.

## Recovery and remaining boundaries

Recovery queries only the UNKNOWN leg's exact client ID, validates the order
binding, then requests one exact-order trade page when execution is nonzero.
Trade order, symbol, side, IDs, cumulative quantity, cash and commissions must
reconcile completely. Missing, foreign, duplicated, truncated or inconsistent
evidence leaves the obligation unchanged. A native not-found response is not
absence proof and never authorizes a replacement order.

Recovery creates no namespace, sends no order, completes no operation and
rearms no runtime. Two observed FILLED orders remain `SECOND_TERMINAL` until
the journal's separate explicit completion. Neither phase establishes restored
initial inventory, flat exposure, realized profit or a stable edge.

The legacy `spot-roundtrip` CLI still contains its two direct writes. This
checkpoint does not upgrade that caller or grant account/order authority.
Its command integration, native instrument/filter capability, partial/canceled
resolution, independent execution-namespace loss/rollback handling and verified
rearm remain open. Independent session supervision, credential isolation,
generation fencing and universal writer admission also remain incomplete.

## Verification

All **131** final cases pass in the gateway, native API and execution-scope lane,
with source snapshots before the contained run. Earlier overlapping lanes
retain their own source phases: 168 journal/quote/runtime cases and the first
127-case API/gateway lane. The latter preceded four additional boundary checks
and alignment of the configured closing timeout with the shared control's
one-second minimum. These are scoped offline checks, not a full-suite result.

The final representative Windows lifecycle check performs 24 interrupted-first-
write, exact-order recovery, second-write and explicit completion cycles. From
the fourth completed cycle through the last, process handles remain **153**,
retained local mutexes **0**, and private committed memory **70,914,048 bytes**.
No forced garbage collection, cache purge, restart or unrelated process change
was used. This small workload supports this path's observed resource lifetime;
it does not prove the absence of all application leaks or qualify a performance
benchmark. All verification child jobs returned terminal; no unattended helper
or persistent registration was installed.

The [evidence](evidence.json) binds the source phases and retained private logs.
No native account, market, order, training or GPU benchmark was accessed.
Financial census and accepted-edge classifications remain unchanged.
