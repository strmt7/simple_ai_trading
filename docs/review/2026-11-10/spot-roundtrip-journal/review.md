# Shared durable state for the Spot roundtrip

The local [journal implementation](../../../../src/simple_ai_trading/binance_spot_roundtrip_intents.py)
reserves both legs in the existing Binance execution namespace. It records the
exact first request and UNKNOWN before dispatch can be considered. The second
request is committed only after the first has complete bound native cash/fill
and commission evidence. No order is dispatched by this module.

The second principal cannot exceed the first leg's net received asset, even if
the initial wallet/allocation contains more funds. BUY uses quote units; SELL
uses exactly transmittable base units. Fees reduce asset movements once. Missing
fees, positive unallocated-asset fees, inconsistent cash, negative allocation
balances and regressing observations reject advancement. Zero third-asset
commission does not invent a debit. Actual fee-policy qualification still belongs
before any caller dispatches an order; receipt validation is not prevention of
an unexpected native charge.

## Namespace and recovery behavior

The first reservation atomically upgrades a verified enrolled namespace to
version 4, adds the operation and control records, and commits UNKNOWN. Existing
version 1/2 migration restrictions remain. Ordinary namespaces stay at their
existing version until this explicit operation is prepared. No actual user
namespace was migrated during development.

The control record binds the active operation, checksum, revision and completed
count. Missing tables, control/active rows, changed bodies and inconsistent
counts reject; record bodies are bounded in the SQL projection before JSON
decoding. A pending operation blocks normal new openings. Separately verified
owned closing remains available with its own fresh client ID. Both planned leg
IDs remain consumed across roundtrip, opening and closing writers.

Each mutation binds scope and the preflight-start operation revision. Partial or
canceled execution remains an obligation. Only two fully reconciled FILLED
orders can release this barrier. Completion records observed asset movements;
it does not mean restored initial wealth, flat assets or profitable execution.
History remains readable after completion, and consumed operation IDs cannot
be replayed.

Recovery mutation uses a noncreating SQLite transaction (`mode=rw`). A missing
database cannot become fresh state, including deletion immediately before
connect. The first private prototype created an empty namespace before rejecting
that recovery; its exact source and comparison remain retained. This correction
does not solve independent namespace enrollment, coordinated rollback or whole
database loss followed by a separate fresh initialization request.

## Verification and current limit

All **407** cases in the final affected lane pass. It covers both orderings,
shared opening/closing and paired-storage compatibility, explicit allocations,
native cash/fees, client-ID reuse, scope/revision drift, missing/corrupt storage,
one-winner concurrent reservation, and actual hidden child exit at all four
unfinished stages. The connect-deletion race also passed separately before the
final cross-writer identity guards were added; it is included in the final lane.
The owned children
ended before their checks returned; no service or recurring worker was installed.

An earlier 131-case prototype lane passed. A later 176-case lane passed 175;
one injected storage-failure fixture expected OSError instead of the existing
public ValueError normalization. Correcting that expectation verified rollback
of scope, tables and version. The exact 131-case prototype source and all raw
logs are retained. Intermediate lanes without a complete saved source phase
are diagnostic history, not independent reproducible qualification. The
[evidence](result.json) binds source and lane identity; this is not a whole-suite,
native venue, crash-at-every-instruction or power-loss claim.

The [quote-principal API work](../spot-quote-principal/review.md) remains a distinct
preceding source phase. The actual CLI roundtrip is unchanged. Complete current
symbol/amount permission, fee and balance qualification, Stop/dispatch authority,
native receipt reconstruction, allocation admission and command recovery before
wiring either order. Third-asset fee support, partial-order resolution/rearm,
independent storage identity and the session supervisor remain open. The journal
does not grant account ownership or implement a universal native-write gateway.

This report records the local source phase prepared for a coherent development
batch. No new public
source, market, account, credential or order was accessed, and no model fit or
benchmark ran. Historical results, consumed financial populations and protected
captures are unchanged: **204 observations / 65 hypotheses / 37 scoped mechanisms
/ 0 qualified stable edges**. The full R&D/application goal remains active.
