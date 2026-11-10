# Durable Spot command integration

The Spot diagnostic now routes both native writes through the existing paired
gateway. Its CLI wrapper contains no `place_order` calls. It accepts an explicit
quote principal for buy-sell or base allocation for sell-buy; `auto` resolves
those units without choosing wallet inventory. The TUI preserves decimal strings
and confirms the selected units. The native command declaration was regenerated;
no native binary was rebuilt or visually qualified.

The command borrows an existing enrolled runtime lease from its session owner.
It does not acquire/release that lease, clear Stop, repair lost enrollment,
start a worker or install persistence. Each created HTTP session closes on
normal return, rejected admission, native failure and cancellation. Local status
needs no credentials or client and creates no missing files.

## Execution and restart behavior

New execution, exact-order recovery, continuation and local completion are
separate actions. Both exact request units and client IDs pass through the
write-ahead journal before native transmission. A lost reply preserves UNKNOWN;
a new invocation cannot retransmit that leg. Recovery observes only the stored
order and trades. Continuation uses net first-leg receipts, with downward native
lot/quote rounding and no foreign-wallet subsidy. A second owned SELL can close
during opening Stop; a sell-buy acquisition still requires the active lease.

Complete native fills and commissions are necessary before completion. UNKNOWN
reports leave total asset movements missing and distinguish retained partial
observations. Completed reports retain exact asset-unit changes and never label
the operation flat or profitable. Completion changes no runtime Stop or lease.
Partial/canceled or contradictory evidence remains an obligation; unsupported
resolution and native not-found never create replacement-order authority.

Instrument checks require matching assets, active MARKET/quote-BUY capability,
quote precision and usable quantity rules. Other native filters can still reject
execution. Free balances are availability observations, not ownership or atomic
reservations. Eligible external fee routes remain blocked pending native FX/cost
admission. Independent execution-namespace loss/rollback detection, partial
resolution/rearm, universal native-writer fencing, hard deadlines, credential
isolation and an independent session supervisor remain incomplete. No native
account/order acceptance or real-capital qualification is claimed.

## Initialization resource repair

Two retained old-source probes leave a constructed HTTP session unclosed: an
invalid call-rate scalar and interruption during session header setup. The
repaired constructor finishes fallible scalar setup before allocating a session
and closes the allocated session if its configuration raises, including
cancellation. The same probes now observe zero allocated sessions for the first
case and one allocated/one closed for the second. These injected object-lifetime
observations do not establish an actual historical socket/RAM leak or explain
earlier host popups. Agent workflows now explicitly cover partial construction.

## Verification and evidence limits

The final production source is covered by **224 distinct focused cases across
two bound lanes**. The combined lane passed 223 of 224; one new fixture tried
to create invalid configuration through a constructor that correctly normalizes
it. The fixture now explicitly mutates the already constructed object. All 32
command cases then pass. Production hashes, other source hashes and every other
test definition remained unchanged between those lanes. Their union covers the
224 final-source cases; this is not a single clean whole-suite run.

The checks include real enrolled SQLite/lock state, both directions, foreign
inventory preservation, lost responses on either leg, restart without duplicate
submission, cancellation, between-leg Stop, missing enrollment, unqualified
instruments/fees, local status, safe output, TUI input units, the real native
client methods with HTTP stubbed, and four command/native-declaration checks.
Ten repeated command invocations close all ten owned sessions and preserve the
external lease. Legacy unscoped command fixtures were replaced by these enrolled
end-to-end cases; the pure receipt controls remain.

The existing 24-cycle runtime cancellation/recovery observation reports handles
**238 / 238 / 238**, retained mutexes **0 / 0 / 0**, and private committed memory
**279,113,728 / 279,113,728 / 279,113,728 bytes** at baseline/maximum/last after
warmup. No forced collection, restart or unrelated process change was used.
This is a bounded lifetime observation, not an application-wide leak audit or
performance benchmark.

The earlier 120-case lane remains separately bound to its pre-constructor-repair
source. Local parser/selector/formatter preflight failures are retained. The
range formatter panicked with both mixed and normalized line endings; native
full-fragment formatting preserved the whole test file's AST. No upstream issue
was posted and no host package/provider was modified.

The [evidence](evidence.json) binds actual raw source snapshots, test logs,
fixture-equivalence proof and constructor probes; publication records Git LF
normalization separately. No market/source capture, account call, native order,
fit, GPU benchmark or persistent registration was made. Local retry routing
established no new access trigger; the financial census and canonical ledgers
remain unchanged. Follow the [operator contract](../../../SPOT_ROUNDTRIP.md).
