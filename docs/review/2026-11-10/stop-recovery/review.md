# Stop closes ledger-owned exposure without public discovery

Two synthetic failures reproduced before the repair. Public discovery failed
after Stop, and the supervisor did not reach owned closing before the test
deadline. Separately, CLI cleanup released the runtime lease when the ledger
reported an unresolved owned order but no position. Neither test used venue
credentials or sent an order.

Stop now wakes the model loop and ends it before the supervisor takes over
owned recovery. The supervisor waits for any admitted model-loop operation to
return before starting that recovery, so it cannot start a second close while
the previous one is running. Recovery uses the ledger and remains available
without a successful public discovery. Authenticated services remain available
during recovery and receive the stop signal even if the drain raises. The CLI
checks unresolved orders as well as positions before it releases the lease;
unreadable ownership state still requests durable Stop.

The [evidence](evidence.json) binds two baseline failures and 107 distinct
passing affected cases to their source. This includes the decision expiry
regressions and a controlled critical-service failure while a forced close
is in flight. Only the CLI module was rerun after fixing a test fixture; the
other two modules had passed against the same production bytes.

This does not change the close-until-flat retry policy or guarantee that an
already admitted network operation will return. The position and order checks
are separate ledger reads, not an atomic venue snapshot. Process containment,
hard termination and recovery of unknown venue outcomes require separate
qualification. No release, new market observation or accepted edge follows.
