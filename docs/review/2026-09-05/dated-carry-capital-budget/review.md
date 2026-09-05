# Retained dated-carry capital budget

The old snapshot does not yet establish an attractive return on committed
capital. All 12 original quantity rows across four contracts are retained in
`result.json`, with 108 conditional scenarios and zero new requests. Seventeen
scenario rows have positive headroom, not seventeen strategies or confirmations.
Every original freshness flag passed then; none establishes freshness today.

At the smallest original quantity in each contract, the following is the
**remaining total budget for noncapital costs**, in basis points of initial
spot acquisition cost, after a hypothetical 3.25% annual capital charge:

| Contract | Quantity | Gross basis | Capital = 1x spot cost | Capital = 1.5x spot cost | Break-even capital multiple with separate 35-bp reserve |
| --- | ---: | ---: | ---: | ---: | ---: |
| BTC Sep 25 | 0.001 | 41.4309 | 13.7428 | -0.1012 | 0.2323 |
| ETH Sep 25 | 0.01 | 21.7913 | -5.8968 | -19.7408 | -0.4771 |
| BTC Dec 25 | 0.001 | 145.1571 | 36.4971 | -17.8329 | 1.0138 |
| ETH Dec 25 | 0.01 | 86.4854 | -22.1746 | -76.5046 | 0.4738 |

The table is an illustrative smallest-size projection; the canonical result
retains every size. At the separate 35-bp reserve, BTC December's break-even
capital multiple ranges from 1.0008 to 1.0138 across all three sizes. Positive
headroom requires strictly less capital than that boundary. Even 1x capital is
not proved sufficient for a fully funded spot leg plus futures margin. Negative
boundaries mean no admissible fully funded capital budget under that scenario;
they are not a recommendation for negative capital or leverage.

At a hypothetical 1% annual charge, all six September quantity rows fail after
the separate 35-bp reserve; December BTC passes all nine size/capital scenarios,
and December ETH passes five of nine. At 3.25%, only December BTC at 1x passes
(three rows). At 5%, none passes. These sensitivities are not probabilities and
their rates are not observed borrowing rates or another venue's quoted yield.

## Accounting and limits

With gross basis `g` in bips, committed-capital multiple `m`, simple annual
cost `r` and original capture-to-delivery years `T`, the remaining budget is
`g - 10000*r*m*T`. Years use the original ACT/365.25 convention. The break-even
multiple after a separately allocated noncapital reserve `c` is
`(g-c)/(10000*r*T)`.

The original 35-bp **all-in** hurdle is an explicit sensitivity, not measured
fees. This review does not assume that it excluded capital, subtract capital
twice from its original result, or overwrite it. The reserve comparison is a
distinct hypothetical partition allocating 35 bips to all noncapital costs.
Opportunity cost must also not duplicate borrowing/financing charges included
elsewhere. Constant committed capital is a simplifying sensitivity, not a bound
on path-dependent margin requirements.

Before qualification, require actual eligible-account fees, source-bound
margin/collateral treatment, sufficient interim liquidity, entry execution and
rounding, delivery/exit spot-versus-settlement-index basis, every financing and
settlement cost, and current depth. Locked futures basis alone does not eliminate
liquidation, exchange, custody, stablecoin or settlement mismatch risk. No new
price access, rank-14 account trigger, model training or promotion is authorized
by these old rows. GPU computation cannot resolve these missing economic facts.

`plan.json` binds the exact old snapshot and frozen new calculation/test source.
The independent artifact test reconstructs the result and checks its accounting
identity; no historical result or canonical acceptance count changed. Next work
should resolve collateral efficiency and exact costs under an eligible contract,
not repeat this grid or treat its selected positive cells as validation.
