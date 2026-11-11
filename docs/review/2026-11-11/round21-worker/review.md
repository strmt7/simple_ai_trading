# Owned five-minute model inference

The ordinary five-minute runtime now compiles and evaluates the frozen Round21
probability hull in a session-owned child. Previously, a stalled native numerical
kernel could retain a scoring thread in the trading process. The parent now owns
the process handle, startup deadline, request deadline and termination. Causal
history, source continuity, risk decisions and trading authority remain in the
parent. The worker receives no credentials or trading authority.

The shared lifecycle also serves the existing fifteen-minute worker. On Windows,
the child starts hidden and suspended, joins the owned kill-on-close Job before
execution and receives a private socket plus a bounded model snapshot. The
parent releases the serialized bootstrap after transfer. Stop, child exit,
startup failure or a reply fault closes the channel and reaps the owned process.
Existing supervisor ordering awaits that teardown before owned-position recovery.
Nothing installs a service, recurring task or startup hook.

Round21 messages contain at most 16 target-free history rows. Length bounds,
strict JSON, request sequence, model identity, population, feature and prediction
digests, candidate hull, availability indices and backend provenance are checked
before accepting a response. Duplicate JSON keys and nonfinite constants fail.
The original numerical predictor supplies the full hull and calibration; the
change introduces no new model mathematics. Backend reporting stays pending
until the child's actual result arrives. A worker fault immediately revokes
forecast admission, including a previously cached forecast.

The [sealed evidence](evidence.json) records source-bound focused verification.
The native CPU fixture compares every probability and support array, including
the prediction digest, against the in-process six-candidate core hull for 1, 8
and 16 rows. Deliberately stalled startup and inference exercise a real persisted
Stop, proving that the child is dead and the blocked caller has returned before
the fixture's owned-recovery callback. Other cases cover startup expiry, stale
request IDs, self-consistent but misbound batches and reply timeouts. A synthetic
environment sentinel is absent in the child. These are synthetic fixtures with
one training epoch; they are not trading results or performance benchmarks.

The first collection attempt failed on an incorrect test import and was fixed.
The earlier 143-case run and subsequent 10-case protocol run remain retained.
A final affected-domain run uses a complete retained source snapshot and a new
private temporary directory. Earlier default pytest temporary-directory use was
a verification mistake; it supplies no host-isolation claim. Formatting findings
are tracked separately from behavioral verification.

There were no market captures, account requests, orders, changes to protected
research partials or new economic observations. Financial qualification remains
206 observations, 65 hypotheses, 37 historical scoped mechanisms and zero stable
qualified edges. This closes the demonstrated numerical-process containment gap
for the ordinary five-minute path on the verified Windows CPU fixture. GPU and
POSIX operation, hard native venue-call containment, a fully independent
credential-bearing gateway and complete crash/reboot recovery remain open.
