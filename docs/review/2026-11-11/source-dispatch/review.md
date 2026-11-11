# Forecast admission at order dispatch

The baseline is `86390139b67881d38a364f32cc2c7b24a32dd50b`. Its
[earlier source-continuity repair](../round21-scoring/review.md) checked the
forecast before and after proposal construction. Execution then performs fresh
funding, signing and journal operations. A source gap during that interval
could revoke the forecast while its completed proposal remained dispatchable.

The real Round21 provider now attaches a source-admission assertion to each
model-driven open, complement lock or targeted reduction. The assertion retains
the exact immutable forecast and market. The supervisor and promoted execution
helpers carry it to the shared coordinator. It is transient process state and
contains no exchange or account authority.

The coordinator checks admission before its preflight and again immediately
after durable submission journaling, before calling the venue adapter. A failed
final check transitions the reserved record from submitting to rejected with
`source_admission_revoked_before_dispatch`. The exception reaches the runtime's
blocked-action accounting. The local journal therefore records a terminal
unsent rejection instead of retaining an unresolved submitting entry.

## Verification

Five focused modules passed **197 cases in 46.75 seconds**. New cases cover
revocation after proposal return, signing, reservation and the final journal
transition. They assert zero venue submissions and the exact terminal state.
Separate cases exercise valid dispatch and callback propagation through all
three model-transition types and their execution helpers. A revoked model
source leaves deterministic owned-exposure closing available.

Subsequent exception review added a pre-dispatch guard to the venue-rejection
handler. A callback that raises the venue-rejection exception class must retain
its local unsent rejection. It cannot be interpreted as a remote response or
changed to unknown. The affected live-coordinator module then passed **130
cases in 10.64 seconds**, including four added exception cases. Other source
and test definitions were unchanged from the 197-case lane.

The [initial sealed evidence](evidence.json) preserves its original source
phase. The [exception-review evidence](exception-review.json) binds the current
source map and separate verification lanes. Both retain stdout/stderr hashes.
Ruff checks passed for the changed modules. Verification used local fixture
venues and callbacks.

## Remaining boundaries

The final check is the local admission point. A later source event can arrive
after dispatch begins, and already transmitted orders retain their normal
reconciliation and recovery obligations. Exchange execution, profitable trading
and successful live recovery remain unqualified by these fixtures.

Providers constructed without a callback retain their existing behavior. The
real Round21 provider supplies it for every model-driven transition. Hard
Round21 inference termination, bounded native venue calls and independent
process supervision remain open. The earlier published evidence keeps its
original source hashes and verification scope.
