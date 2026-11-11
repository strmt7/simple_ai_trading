# Comparable finite capital before admission

The autonomous Binance loss guard now checks mark identity and quote units for
legacy and Futures records as well as receipt-native Spot inventory. Previously,
those checks ran only when a Spot cash receipt was present. A legacy ETH position
could therefore be valued using the current BTC scalar mark; a missing mark
could turn its unrealized loss into zero. Closed legacy cash could also be added
across USDC and USDT without a conversion basis.

The guard now requires an explicit supported quote and instrument context when
the snapshot contains inventory or realized trades. Every open position must
match the scalar mark's exact symbol and product. Realized records must share
the declared quote. A snapshot without open inventory needs no invented price.
This single-instrument path has no qualified multi-instrument mark map or FX
valuation; it rejects that unsupported aggregation rather than assigning value.

Reference capital must be positive, finite and nonboolean. The loss fraction
uses the actual reference, including values below one unit. The earlier
one-unit denominator floor understated those daily and session loss fractions;
the entry gate's compensating drawdown rescaling is now unnecessary. Aggregate
cash and resulting fractions must be finite: individually finite positions can
overflow during multiplication or summation, and a tiny positive reference can
overflow a loss ratio. Such values cannot manufacture zero loss or usable profit.

Unqualified valuation returns a blocking reason with `force_close=False`.
The ordinary loop halts through its existing guard handling. No valuation
failure authorizes liquidation at an invented price. This change does not
implement portfolio valuation, a fresh-price guarantee, high-water drawdown,
reboot-resistant loss policy or independent gateway supervision. Existing
owned-closing commands retain their separate authorization and reconciliation.

The [evidence](evidence.json) preserves exact source phases and offline logs.
Thirteen of the original fourteen regression cases failed against the retained
pre-change implementation; the empty-inventory control passed. Final coverage
adds an unparsable reference and loss-ratio overflow. Tests cover Spot and
Futures instrument mismatch, missing/nonfinite marks, mixed realized quotes,
invalid reference capital, overflow and exact fractional-reference arithmetic.

The wider domain run exposed stale fixtures as well as the expected changed
admission behavior. Direct guard tests now supply their actual instrument.
Two cash-test functions now explicitly enroll their synthetic empty runtime
using the existing fixture. Their interruption wrapper also forwards the
noncreating `create=False` contract. Production storage/lease gates were not
weakened, and the tests still prove UNKNOWN retention and no repeated SELL.
Failed phases remain recorded separately from the final passing scope.

There were no market captures, credentials, account calls or orders. Protected
research and historical result bytes are unchanged. This is a capital-admission
repair, with no new financial observation or qualified stable edge.
