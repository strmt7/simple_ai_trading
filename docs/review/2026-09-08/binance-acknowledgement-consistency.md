# First-acknowledgement consistency before accounting

September 8, 2026. [Source-bound checkpoint](binance-acknowledgement-consistency.json).

## Reproduced capital-risk path

Before integrating native recovery application, review found that the active
first-response fill interpreter preferred fill rows over a contradictory
cumulative executed quantity. At baseline `353b5580`, an isolated offline close
of a 1.0 lot returned executedQty 0.5 and fill rows totaling 1.0. The gateway
recorded a full 1.0 close, removed the lot and cleared its entry block. No
consistent exchange evidence justified that state. This is a reproduced
software defect, not evidence that an actual venue returned this response.

The direct CLI had a separate permissive interpreter with the same precedence
problem and a market-price fallback. Fixing only the autonomous caller would
leave that paired boundary inconsistent. Both active response/query projections
now use `binance_acknowledgements.acknowledged_fill`; explicit CLI dry-run
simulation retains its former fallback behavior.

## Accepted evidence and boundaries

Cumulative executed quantity is required before fill rows can support a fill.
Present original quantities must be positive and consistent with execution and
FILLED status. Fill rows must be bounded, well-formed, positive and sum exactly
to cumulative execution. Their price-times-quantity cash must agree with any
reported cumulative cash. Present trade IDs must be valid and unique; present
native commission fields must be valid. Invalid rows are not silently skipped.
The shared identifier parser rejects boolean, malformed and lossy order IDs.

Spot uses cummulativeQuoteQty and linear futures cumQuote. Cross-product cash
fields and cumBase are not interchangeable units. Positive cash/fill evidence
sets the gross average used by the legacy float projection; a convenience
avgPrice cannot override it. When only a valid average and executed quantity
exist, the result is explicitly average_only with no source-reported cash.
The order's limit price and caller's mark-price fallback cannot prove execution.
The CLI may still display a fallback price with zero executed quantity; it must
not turn that into a positive fill. Contradictory evidence raises before any
adaptive order query; an unfilled acknowledgement can still use the existing
exact-order query path.

Up to 1,000 products of two at-most-60-significant-digit inputs require at most
123 significant digits. The new cash arithmetic uses precision 128; a test
checks a product that would lose information at precision 100. Quantity and
cash reconciliation are exact under those bounds. Division to form an average
can be rounded, and existing float position/PnL projections remain approximate.
No instrument-specific rounding tolerance is invented; a reported-cash mismatch
requires separate precision adjudication rather than silent repair.

This validates arithmetic consistency, not every order identity, ownership,
native fee, endpoint, account or timing attribute. Autonomous request/scope and
durable-intent gates remain separate. Direct CLI commands still need complete
durable-gateway and submitted-order binding coverage. The legacy CLI projection
helper remains for explicitly simulated values and already-checked normalized
inputs; it is not an execution-evidence validator.

## Verification and remaining integration

437 execution-domain and 58 targeted CLI cases pass: 495 distinct checks,
including 50 new cases. The reproduced opening/closing contradictions preserve
UNKNOWN and leave the paired ledger unchanged. Coverage includes original vs
executed quantity, cumulative vs fill cash, malformed/duplicate/oversized rows,
native fee syntax, average precedence, maximum arithmetic bounds, CLI response
and query parity, explicit dry-run fallback and existing recovery boundaries.
CLI fixtures now provide execution cash instead of a guessed market price and
use futures cash fields for futures; negative balance scenarios still reach
their intended boundary. Initial fixture failures were corrected without
loosening production admission. Existing numerical research results are intact.

Ruff, changed-line formatting and formatter AST equivalence passed. No parser,
command metadata or native launcher interface changed. No full repository suite,
training, benchmark, account request, credential access, actual user-ledger
mutation or protected capture occurred. MRNA remained before its 13:35 UTC
research gate when this capital-risk defect was selected; no financial retry
condition or registry count was changed.

Native fees are syntax-checked here, not yet applied to net sellable inventory
or valued PnL. Next join native opening and incremental terminal closing
movements to authoritative accounting, complete direct CLI durable authority,
and integrate explicit current account/policy/process-fenced rearm. Independent
supervision, comprehensive final review and profitable-edge qualification remain
open. Do not equate this checkpoint with native recovery or enterprise readiness.
