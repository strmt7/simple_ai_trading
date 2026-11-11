# Owned numerical inference for the fifteen-minute profile

The promoted fifteen-minute CLI starts a session-owned Round 16 inference
process after authenticated preflight. It warms up the verified candidate
on a synthetic vector and acknowledges the digest of its candidate bytes
before accepting requests. Each request carries one sequence number and a
fixed-width float32 feature vector. Each response carries that sequence and
a finite probability in the open interval from zero to one. Startup and
response deadlines are bounded. A bad response or an expired request
invalidates the worker and triggers supervisor shutdown.

The controller builds and qualifies causal features, applies the promoted
probability age and decision generation, reads the book, constructs the
proposal and enforces risk and dispatch authority. The child receives only
the verified candidate and feature values. On Windows the controller
launches the child hidden and suspended, enrolls it in the owned Job, and
uses a private socket pair. The child has a minimal environment and the
bootstrap file is released after reading. The supervisor closes and reaps
the child before taking over owned-position recovery.

The [evidence](evidence.json) binds the focused checks and the exercised
Windows transport. Ridge and LightGBM reproduced the exact in-process
probabilities on the synthetic fixtures. A mismatched sequence and a
stalled response both invalidated the worker. A real persisted Stop
terminated a hung child and unblocked the waiting call before the
synthetic recovery callback. No venue access, fit on financial data,
timing benchmark or profitability observation followed.

Round 21 numerical inference, Round 16 feature construction and public
book access, and authenticated native calls still need containment.
POSIX transport and current live model latency remain unverified.
