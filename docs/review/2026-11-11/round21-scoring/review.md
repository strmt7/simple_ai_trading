# Five-minute forecasts and public-feed continuity

The original coordinator held its intake lock across numerical scoring.
A stalled scorer blocked a core-gap callback. After the callback finally
ran, a repeated evaluation at the same slot could still return the
observed forecast. The public feed calls the gap callback synchronously
on the event loop that also services durable control.

The coordinator now captures a causal row under the intake lock and
scores it under a separate decision lock. A required source epoch change
revokes an in-flight or cached forecast. Core gaps still block the current
market, and an optional Binance gap leaves the core-only model usable.
The selected optional model clears its temporal history before scoring
with the new epoch. The decision provider checks source admission before
fetching a quote and before returning any model-driven proposal.

The [evidence](evidence.json) records the original reproduction and
85 passing focused tests. The regression cases verify intake during
stalled scoring, cached forecast revocation, optional recovery and the
quote interval. Hard termination of five-minute inference, native
request containment and source invalidation at the final dispatch
boundary remain open. Verification used fixture data and synthetic
callbacks.
