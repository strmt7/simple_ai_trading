# Owned SDK HTTP execution

The ordinary Polymarket CLOB SDK transport now executes HTTP work in a
session-owned child. The parent waits on a bounded private socket exchange and
reaps a stalled child. Previously, the [owned HTTP adapter](../../2026-11-10/sdk-transport/review.md)
could observe a deadline after native work returned, but could not interrupt a
native HTTP freeze. The existing native request, body-size, encoding, redirect
and no-retry rules remain inside the child.

One child retains its HTTP connection pool across successful requests. Creation
is lazy; constructing or closing an unused transport starts no helper. Windows
starts it hidden and suspended, assigns it to the owned kill-on-close Job, then
resumes it. The original process and Job handles own cleanup. The POSIX path uses
an explicitly created session/process group, verifies the original live child's
group and signals only that group. This turn verifies real Windows behavior;
POSIX execution and descendant behavior remain unqualified here.

Startup settings are public origin and resource limits. The child inherits a
minimal environment and receives individual signed requests over its private
socket. API secrets and signing keys remain with the existing parent SDK; the
HTTP child necessarily receives each request's authentication headers and body.
This is not a credential-security sandbox or an independent trading gateway.
Request material is not placed in the bootstrap file, command line or logs.
The child releases per-request references after returning a response.

The request deadline includes capacity waiting, lazy startup, sending and reply
receipt. Startup, request and reply frames have byte limits. Strict JSON, exact
message fields, bootstrap digest and request sequence prevent a partial or
misbound response from becoming acceptance or rejection. HTTPX's pinned query
encoder normalizes Decimal, boolean, null and repeated query values before IPC;
their exact query encoding is retained. Signed string bodies and headers pass
through unchanged. No pickle or request replay is involved.

A native hang, child death or protocol fault has no definitive HTTP status.
The old child is reaped before a later explicit request can start a fresh
generation. A later read can therefore reconcile an uncertain write without
replaying it. A complete bounded rejection retains its native status and
payload. Explicit transport close interrupts a blocked request. Parent-side
request serialization, process creation, SDK signing, the venue's separate
public `requests` session and the trading process itself are outside this
containment boundary. Full independent supervision and physical crash/reboot
acceptance remain open; configured deadlines are not protection against every
kernel, hardware or venue failure.

The [source-bound evidence](evidence.json) retains failed and passing phases.
Offline fixtures exercise a real contained child, startup expiry, a native-call
stall, abrupt exit, reply misbinding, capacity expiry, bounded frames, exact
payload/query preservation, explicit close and subsequent read generation.
The actual venue exception classifier and coordinator retain UNKNOWN through
ledger reopen after the injected native HTTP stall, with exactly one POST.
Existing real-SDK serialization tests retain their injected offline HTTP
transport; an owned loopback drip test exercises the ordinary process transport.
No external network or venue account is used by these checks.

An embedded fixture script initially lost a newline escape during formatting.
Its failed phase remains retained. The raw string is now parsed before launch,
and the corrected child cases pass. Current evidence distinguishes those source
phases rather than replacing the failed log. There are no new market captures,
economic observations, orders, credentials from history, recurring services,
startup hooks or scheduled tasks. The financial ledger remains 206 observations,
65 hypotheses, 37 historical scoped mechanisms and zero qualified stable edges.
