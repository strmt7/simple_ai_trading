# Durable Stop reaches the initial discovery child

The live supervisor started durable Stop polling after its first public
discovery. In the Windows reproduction, the initial child had started and
was stalled, and the local control store recorded a Stop request. Owned
recovery still did not start within the test deadline. A direct cleanup signal
then reaped the test child.

The initial discovery now runs as part of the supervised model task. After
authenticated preflight and the wallet backing check, the supervisor starts
the safety services and duration timer before launching discovery. Durable Stop
can then signal the child during the first request as well as at rollover.
The runtime reaps discovery before the owned close loop and keeps the safety
services available during that loop. When discovery fails without a concurrent
Stop, the startup error still reaches the caller after recovery.

The startup and rollover tests persist Stop through the real local control
store and verify process exit before the synthetic close callback. Startup
does not evaluate the model after that Stop. All 111 focused cases passed.
This covers the public discovery
interval. Authenticated preflight, forecast execution and authenticated native
operations still have separate containment work. [Verification evidence](evidence.json)
