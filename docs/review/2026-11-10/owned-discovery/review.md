# Stop can end a stalled public discovery call

At rollover the live supervisor waited on a thread that performed public market
discovery. In a synthetic reproduction, Stop reached owned recovery only
after the test released the blocked thread. That left known exposure waiting
for an unrelated public request.

The live CLI now uses an owned child for five-minute and fifteen-minute BTC
discovery. The child receives the horizon and observation time, starts with a
restricted environment, and performs one bounded public Gamma markets read.
The parent reparses the reply and checks it against the two requested slugs.
Windows starts the child hidden and suspended, then enrolls it in a job before
resuming it. The discovery deadline is 12 seconds, with a separate bounded
wait for process exit. Stop terminates and reaps the child before handing off
to owned recovery. A cleanup failure prevents a second launch from that
discovery instance.

The Windows integration test stalls a real child at rollover and verifies its
exit before the synthetic close callback. It also verifies a separate deadline,
both reply horizons and the absence of an inherited test environment value.
The normal CLI assembly installs the worker. The initial test wrongly assumed
a nonzero exit code after Windows job termination. The replacement verifies
startup, process exit and absence of normal completion. All 114 focused cases
passed across the two verified phases. [Verification evidence](evidence.json)

This process owns discovery only. Forecast evaluation, public quote reads in
the providers, and admitted authenticated operations still need independent
containment. A durable Stop issued during initial discovery is subject to the
startup deadline because its polling service starts afterward.
