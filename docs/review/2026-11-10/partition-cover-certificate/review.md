# Exact certificate for every mixed long-only partition cover

The retained NYC November 11 vector now has an exact mathematical rejection
certificate covering arbitrary nonnegative mixed and unequal YES/NO quantities.
This strengthens the existing eighteen-row screen without repricing it, repairing
missing sides or requesting another response. The [contract](contract.json)
records that the rejection was known before this post-hoc audit; the
[result](result.json) is not a new economic observation or prospective validation.

## Proof and constructive separation

Assume exactly one supplied state wins. Let `a_i` and `b_i` be nonnegative linear
YES/NO acquisition costs in the common payout unit. Unavailable instruments
cannot be bought. If artificial nonnegative weights satisfy

```text
sum(w_i) = 1
w_i <= a_i                 for every available YES
1 - w_i <= b_i             for every available NO
```

then every nonnegative portfolio `y_i, n_i` satisfies

```text
cost >= sum_i [w_i*y_i + (1-w_i)*n_i]
     = sum_s w_s * payout_s >= min_s payout_s,
where payout_s = y_s + sum_i n_i - n_s.
```

Thus no portfolio has a **strictly positive nominal floor surplus** at these
costs. The weights are not forecasts, calibrated probabilities or expected-value
estimates. Zero state weights are permitted; this does not exclude every weaker
arbitrage definition or a profitable risky trade.

The constraints reduce to `max(0,1-b_i) <= w_i <= min(1,a_i)`, with zero/one
bounds for missing sides. Feasibility requires each lower bound to be at most
its upper bound and the lower/upper sums to bracket one. Starting at the lower
bounds and allocating remaining mass within the intervals constructs an exact
certificate with a linear number of rational operations. Integer bit lengths
still determine actual computation cost; this is not a measured speed claim.

Each infeasibility supplies a constructive cover: an interval conflict gives a
same-market straddle below one; an upper sum below one gives all YES below one;
a lower sum above one gives the NO subset with `b_i < 1` below its `k-1` floor.
These families therefore suffice for strict floor-profit existence, including
mixed and unequal portfolios, under the supplied linear law. No exponential
subset search, numerical LP or fitting is required. This does not optimize
native lot sizes, capital or executable size.

## Retained-data audit and controls

The one-use zero-network audit verifies the exact 90,246-byte response before
decoding. All eleven identities pass the unchanged source eligibility parser.
Native numbers are parsed as Decimal and converted to Fraction. Eleven YES and
nine NO costs match all twenty distinct legs in the original eighteen rows;
the two missing NO sides stay unavailable. All twenty certificate inequalities
pass exactly. Original contract, result and raw bytes remain unchanged. No fee,
tick, threshold or old row is recalculated, and no observation or hypothesis is
added. Existing consumed-event and retry rules remain controlling.

The shared [supplied-law module](../../../../src/simple_ai_trading/partition_cover.py)
passes **21 focused controls**, covering the three
witness families, unavailable sides, forged certificates, equality, sub-float
strictness and all 729 small mixed/unequal portfolios in one constructed
three-state example. These supplement the proof; they are not market backtests
or speed benchmarks. No account, credential, order, protected capture, training,
GPU or new market access occurred.

## Financial limits and next decision

Additional nonnegative costs cannot create nominal floor profit at the supplied
gross vector. However, Gamma snapshots and the one-minus-bestBid NO proxy do
not establish current executable costs or exclude cheaper actual NO asks.
Disputes, cancellation, exceptional resolution and other payoff states remain
unqualified. This is **not proof of no actual market arbitrage**, family-wide
negative expected value, stable profitability or deployable capacity.

Use the certificate to avoid redundant mixed-portfolio searches in future
eligible partition studies, after their own complete native payoff and cost
qualification. The old vector authorizes no refresh, favorable sibling, book or
fee endpoint. The family receives metadata only. Financial census stays
205 observations / 65 hypotheses / 37 scoped mechanisms / **0 qualified stable
edges**; broader financial, model, execution and product work remains open.
