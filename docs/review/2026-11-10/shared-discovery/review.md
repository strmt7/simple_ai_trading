# Share the supervised market with the five-minute predictor

On the previous source revision, the Round 21 data service could still wait in
its own public discovery thread after Stop. A synthetic discovery callback
confirmed that the service stayed pending after Stop while that thread was
blocked. Releasing the callback let the service terminate. The supervisor
already had an owned public discovery process for the same market.

The live CLI now gives both components the same discovery owner. It publishes
the exact currently active result after reply validation and child teardown.
The Round 21 feed waits for that publication and never issues its own discovery
request in the live composition. An expired market is unavailable through the
shared source. Publication wakes the waiting feed without waiting for its next
periodic check. Direct standalone callers retain their existing discovery
path.

The [verification](evidence.json) binds the original reproduction, source
navigation and the focused passing suite. The new feed test receives startup
and rollover results from real hidden, contained child processes using fixture
payloads and verifies no second client call. Existing persisted-Stop cases
still verify child exit before synthetic owned recovery. The live CLI wiring
and Round 21 runtime and decision cases also pass.

Forecast scoring, order-book reads in the decision provider and authenticated
native operations still need process or transport containment. No live account,
order, market request or protected data was used for this engineering check.
