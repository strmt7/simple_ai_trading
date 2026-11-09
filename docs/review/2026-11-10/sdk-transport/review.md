# Owned native SDK transport

The live adapter's configured timeout previously reached its public GET helper,
while native SDK requests used the SDK's module-global HTTP client. The native
response body also bypassed the adapter's response-size check. That mismatch
matters because the coordinator holds its ordered submission interlock during
the native order POST: delayed response work can obstruct control progress.

The pinned SDK's `_get`, `_post` and `_delete` hooks now route through one owned
transport per constructed venue. No package source, global SDK client or another
venue's transport is replaced. The SDK still builds and signs the order; its
serialized UTF-8 body and authentication headers pass through unchanged.
`retry_on_error=False`, `use_server_time=False` and the existing immediate-response
override remain intact. There is no automatic HTTP retry or redirect following.

The transport applies the configured connect, read, write and pool timeouts,
checks an elapsed deadline as response work progresses, and caps retained bodies
at the existing 8 MiB limit. It requests identity encoding and rejects compressed
responses before decoding them. Partial, interrupted, oversized or late responses
have unknown status; a partial 422 cannot become a definitive rejection merely
because its headers arrived. Parsing must also finish within the observed
deadline. Complete bounded HTTP errors preserve the SDK's status and payload
semantics. Network-error diagnostics omit signed URLs; the transport emits no
response-body log. Complete error payloads remain available under the SDK's
existing exception contract.

Construction cancellation closes the new transport and the owned public session.
Venue shutdown attempts public-session cleanup even if SDK cleanup raises. A
second venue retains its independent usable transport after the first closes.
Externally supplied clients keep their existing injection contract; this repair
does not configure or qualify their transports.

## Verification

All **187 focused transport, native-adapter and coordinator checks pass**, without
a skip. Three new regressions fail against the exact pre-repair adapter. The
tests use the actual pinned SDK for signing and serialization with published
synthetic offline fixtures. HTTP responses are mocked; no venue request occurs.
They verify preserved signature headers/body bytes, per-phase budgets, independent
clients, no SDK-global dispatch, no redirect/error replay and cleanup on failure.
Existing adapter tests are unchanged; other adapter definitions are AST-identical.

A real, owned loopback server sends chunks frequently enough to avoid the socket
idle timeout; the observed response deadline still rejects the slow body. Its
listener, serving thread and request handler finish before the test returns.
This checks response behavior, not a performance benchmark. Initial fixture
failure logs remain privately retained alongside the final and old-source logs.
The [evidence](evidence.json) binds current source/tests, original adapter, actual
source-navigation receipt, installed SDK source snapshots and verification logs.
Ruff lint and formatting checks pass.

## Remaining boundary

Socket timeouts and observed deadline checks cannot preempt an arbitrary frozen
native call or thread. They are not a hard process deadline or an independent
supervisor. The application still needs a session-owned process boundary, bounded
IPC, generation fencing, durable pre-transmission uncertainty and exact recovery
after interruption. Preserve separately available owned reduction and never turn
thread cancellation into assumed venue cancellation. The [capital architecture](../../../CAPITAL_PROTECTION_ARCHITECTURE.md)
retains these requirements and the unresolved storage/rearm/model-health work.

The native-write investigation also identified Binance autonomous opening/closing
and a distinct test-environment calibration transport. This is candidate routing,
not a complete caller inventory or proof that all writes use one gateway. No
native market/account request, real credential, order or protected capture was
used. No service, recurring task, startup hook or persistent worker was installed.
Financial counts remain **204 / 65 / 37 / 0 qualified stable edges**.
