# Carry settlement: isolate the hedgeable timing error

The new conditional linear-contract evaluator separates three different exit
risks: index-versus-spot tracking, exit-schedule timing, and actual execution
slippage. Matching settlement weights removes the second term, not the other
two. This is a financial modeling improvement, not a profitable market result
or an executable strategy. [Reproducible examples and source bindings](settlement-tracking-review.json)
preserve every earlier carry result and retry gate.

## Identity and useful bound

Let `q` be matched linear base quantity, `F0` the short entry price, `S0` the
spot acquisition price, `w_i` normalized settlement weights, `v_i` actual spot
exit quantity fractions, `I_i` index observations, `S_i` spot reference prices,
`X_i` actual execution prices where sales occur, and `C` all other net costs.
Both weight vectors sum to one. All cash amounts use the same quote unit.

The direct cash ledger is:

`P = q * (sum(v_i*X_i) - S0 + F0 - sum(w_i*I_i)) - C`.

It decomposes into:

`P = q*(F0-S0) + q*sum(w_i*(S_i-I_i))`
`    + q*sum((v_i-w_i)*S_i) + q*sum(v_i*(X_i-S_i)) - C`.

These are entry basis, index tracking, timing tracking, execution tracking and
other costs, respectively. Execution price impact is already in `X_i`; charging
that same slippage again in `C` would double count it. An unexecuted slot needs
a reference observation but has no fictitious execution price.

For a separately justified spot-reference envelope `[L,U]`, define
`D = 0.5*sum(abs(v_i-w_i))`. Then the timing term has absolute bound
`q*D*(U-L)`. This is tight: positive and negative weight differences each sum
to `D`; assigning opposite price extremes attains the endpoints. If `v=w`,
this term is identically zero on every common-price path, rising or falling.
A range inferred from realized sample extrema is only an ex-post diagnostic,
not an established future bound or probability statement.

## Fixed illustrative examples

These five examples use synthetic quantity 1, entry spot 100, entry future 101
and other costs 0.2. They are not current prices or training observations.

| Supplied case | Net quote cash flow | Decisive component |
| --- | ---: | --- |
| Matched weights; prices rise 90 to 110 | 0.8 | Timing error zero |
| Matched weights; prices fall 110 to 90 | 0.8 | Timing error zero |
| Same falling path; all spot exits at the endpoint | -9.2 | Timing loss 10 |
| Matched weights; index exceeds spot by 2 | -1.2 | Index tracking loss 2 |
| Matched weights; executions lose 3 versus reference | -2.2 | Execution loss 3 |

## Source and implementation scope

The retained clearing-source review identifies an arithmetic settlement average
and already warned that point-in-time spot execution differs from it. Its
underlying PDF hash was reverified. This increment reuses that documented
question; it does not reopen the PDF, resolve its inverse-PnL sign conflict,
establish its effective date or certify account/product applicability. See
[the retained source review](../2026-09-04/inverse-clearing-source-review.md).
No current Binance settlement rule or attainable cost saving is asserted here.

`settlement_tracking.py` uses exact rational arithmetic internally. Positive
settlement weight units normalize without approximating values such as 1/1800;
actual exit fractions must conserve the entire quantity. Result presentation
rounds recurring decimals to 60 significant digits. Input arithmetic size is
bounded. Supplied observations must match the declared complete ordered schedule;
missing, duplicate or reordered timestamps reject. The caller must independently
prove that this declaration matches the actual settlement contract.

33 focused checks pass, including a 1,800-slot equal-weight case, nonterminating
averages, missing/fictitious executions, quantity conservation and the direct
cash-flow identity. The source-audit skill kept venue claims conditional; the
Python/regression skills guided separate reference/execution prices and exact
weights. One test initially used a sum divisible by three when intending a
nonterminating average and had an incorrect expected value. Its fixture was
corrected; the model was not changed to fit that mistaken expectation.
Four normalized source bindings and full example reconstruction were verified.
No broad CI, GPU training, performance benchmark or hosted verification ran.

## Decision and next evidence

Future linear carry evaluations should price the proposed **quantity-feasible**
exit schedule and report these terms separately, rather than assume endpoint
spot equals delivery or apply one unexplained settlement haircut. Before any
forward adoption, source-bind the exact contract/index, observation timestamps,
weights and exceptional settlement rules; establish joint index/reference and
execution data, lot-size rounding, depth, costs, collateral access and interim
margin. A coarse schedule may reduce request/fee overhead while increasing
tracking distance; only matched execution evidence can decide that trade-off.

This is not an inverse/quanto model, a fill simulator or a live unwind schedule.
Unwinding collateral can itself threaten margin; terminal cash identity does
not prove pathwise liquidity, enforceable redemption, custody or solvency.
No venue/account/order request or historical repricing occurred. Rank-14 account
qualification and every consumed-study/protected-capture boundary remain intact.
There is no new accepted hypothesis, terminal market observation or qualified edge.
