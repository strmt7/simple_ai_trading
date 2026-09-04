# Funding-rate sums versus funding cash flows

This is an offline exploratory review, not a new market test. The original
Backpack/Binance contract, orientation, source bytes, role boundaries, results
and retry trigger remain unchanged. Machine-readable results and a durable
journal are in `funding-notional-sensitivity.json` and its journal sidecar.

The retained runner reduces Backpack's hourly rates and Binance's eight-hour
rates to an unweighted sum of spreads. The retained official Backpack terms
explicitly define payment as funding rate × quantity × mark price. Their
original source hash, `64ef1f561e6d095d6a732a770b07dac8460d543ea592e148a517e1a7239c45b2`,
was rechecked without refetching. Source:
[Backpack Futures Specs](https://support.backpack.exchange/technical-docs/trading/futures-specs).

For a fixed-quantity linear pair, each actual settlement therefore contributes
its signed rate times its own mark notional, converted to a common numeraire.
Different settlement instants and quote units cannot silently receive weight
one. Constant-notional rebalancing is a different strategy and needs its own
feasible quantity changes and costs. Rate-only diagnostics do not reconstruct
either strategy's actual cash flow.

Synthetic counterexample: +0.0011 and -0.001 sum to a positive rate. With
one unit and settlement marks 100 then 120, funding cash is 0.11 - 0.12 =
-0.01. This proves an inference limitation, not that a historical strategy
lost or gained that amount.

The review reconstructed all nine retained asset/role gross sums exactly.
It then isolated hypothetical independent settlement-notional weights within
±10% and ±20% of the reference, leaving the original hurdle fixed. If signed
coefficients sum to S and their absolute values sum to A, the outer interval
is S ± epsilon×A, less that fixed hurdle. These are mathematical sensitivity
bounds, not observed price ranges or jointly feasible delta-neutral paths.

Even the optimistic ±20% **test-role** endpoints remain negative: BTC about
-96.02, ETH -92.77 and SOL -89.34 bips. Thus this particular missing-weight
issue does not rescue these tests under that illustrative scenario. Their
complete real cash flows, realized costs, basis and capital paths remain
unreconstructed. Do not refetch or reverse an orientation from this review.

Future eligible carry studies should retain each settlement's quantity,
contract multiplier, mark/notional, quote conversion, financing and basis
cash flow. Labels should target integrated after-cost value on total committed
capital rather than rate spread alone. Eight focused arithmetic/rejection
tests passed; no claim of profitability or new validation follows from them.
