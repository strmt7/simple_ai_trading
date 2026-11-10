# Shared durable runtime control

The local implementation extracts the existing Polymarket lease, pause, Stop
and dispatch ordering into one venue-neutral control core. Polymarket retains
its original table, v1 migration, v2 record digest, public types and exceptions.
The new Binance adapter has a distinct table and schema. Neither adapter imports
the other venue's execution or account subsystem.

The core uses the existing local mutex and OS file lock, checks the durable lease
before entering a submission guard, orders pause/Stop/heartbeat expiry against
that guard and preserves Stop across controller restart. The separate close
interlock remains usable when new exposure is forbidden. These are cooperative
caller guarantees: an unguarded native transport does not acquire them merely
because this module exists.

Storage identifiers are immutable validated adapter metadata; SQL values remain
bound parameters. A namespace with no legacy schema rejects missing pause fields
rather than guessing a migration. If adapters deliberately share one database,
their rows remain independent while file interlocks conservatively serialize
both. Normal integration must choose and bind the intended storage explicitly.

The former authority-import test now admits exactly the neutral core from the
Polymarket control adapter. It independently rejects venue/application imports
inside that core; unrelated authority modules retain their previous restriction.
Tests inject storage faults into the core that owns the connections instead of
requiring production adapters to export otherwise unused standard libraries.
An older fresh-interpreter test now uses the bounded owned-process helper,
including hidden Windows startup and process-tree teardown.

All **222** final affected checks pass. A separate 14-step replay against the
preserved original class matches snapshots, native control rows and digests,
returns and exception messages. The extracted adapter also reads the original
database without changing its control row. These supplied scenarios do not prove
all possible behavior equivalent. The [result](result.json) binds both test
phases, their pre-launch source snapshots and the retained compatibility replay. The
actual process test holds the submission guard in an owned child, observes that
Stop waits, releases the child, then verifies Stop rejects subsequent dispatch.
This is real cross-process locking and persistence, not an OS reboot, power-loss
or independent-supervisor test. Old test logs and source snapshots are retained.

Binance's existing state-file controller and roundtrip command are unchanged.
The new adapter is not wired into either, and no native account or order was
used. Exact fee-asset grants, balance/capability/freshness admission, partial-fill
recovery and rearm, independent storage identity, universal caller fencing and
session supervision remain required. Complete database/namespace loss is still
not protected by an independent enrollment record. No automatic service,
recurring task or startup hook was installed.

This report records the local source phase prepared for a coherent publication
batch. Financial counts remain **204
observations / 65 hypotheses / 37 scoped mechanisms / 0 qualified stable edges**.
No market request, training fit, benchmark or profitability claim follows.
