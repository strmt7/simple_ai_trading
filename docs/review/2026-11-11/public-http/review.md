# Owned public venue reads

The ordinary V2 venue's public reads now use a session-owned read-only HTTP
child, alongside its separate [SDK HTTP child](../http-process/review.md).
Previously, the public `requests` session ran inside the trading process and
retried GETs. A stalled native call or a long server retry delay could obstruct
preflight or recovery despite the socket timeout. The new default uses bounded
RPC and one HTTP attempt. Explicitly injected sessions retain the existing
offline integration seam and do not inherit the native-containment claim.

Four existing endpoints are admitted: CLOB version and time, geoblock status
and public wallet positions. Their three origins share one public child and
separate native connection pools. The immutable bootstrap binds the origin and
exact endpoint sets. Both the parent reader and child validate scope; the child
also rejects non-GET methods, caller-supplied headers and bodies. It receives no
signing keys or authenticated SDK request material. This read-only policy is
distinct from the SDK child's authority to transmit individually signed requests.
Neither process is a security sandbox against an arbitrary same-user program.

The shared HTTP worker accepts an optional shorter caller deadline. A complete
position query uses one monotonic deadline across all pages, parsing and final
materialization. A timed-out or incomplete query raises; it cannot return a
partial empty inventory as evidence of flatness. Existing page, quantity,
duplicate-token and ownership checks remain. The server-clock midpoint now
brackets only the time request, excluding preceding version-request latency.
This corrects that specific measurement; it does not qualify an independent
clock-offset bound or protect against every wall-clock discontinuity.

The [evidence](evidence.json) binds the current source and retained phases.
All **316 affected checks pass**. Native Windows child fixtures cover the
ordinary preflight factory, all three origins sharing one child, independent
child rejection of unauthorized requests, caller expiry without dispatch,
reaping a stalled read and unchanged signed-SDK/UNKNOWN behavior. Controlled
position tests exhaust one total budget during network delivery and during
materialization. Another test inserts ten seconds of version latency and checks
the corrected clock sample. No performance benchmark is claimed.

The first focused phase retained 52 passes and one failure because its new
preflight fixture omitted two required fake SDK methods. The fixture was
completed; the corrected 54-case phase passed. Final verification also includes
the clock and malformed-scope cases. Earlier logs and full source snapshots
remain separate. Static checks and the affected-domain run supplement semantic
review; they do not establish whole-product or whole-repository qualification.

No external venue requests, account access, orders, model training or protected
research captures occurred. The two designated carry-fee variables were checked
for presence only and were absent; no credential was read from history. No new
primary evidence admitted the consumed negative-risk study. Financial counts
remain 206 observations, 65 hypotheses, 37 historical scoped mechanisms and zero
qualified stable edges.

Windows termination is verified; POSIX execution remains unqualified this turn.
SDK signing, other clients and streams outside this V2 adapter, source admission
at actual remote dispatch, independent credential-bearing gateway supervision
and complete crash/reboot recovery remain open. The existing source-admission
claims concern the local venue-call boundary. No service, recurring task,
startup hook, automatic order replay or release was created.
