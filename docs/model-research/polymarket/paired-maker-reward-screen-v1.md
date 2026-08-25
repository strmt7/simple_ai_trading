# Polymarket Paired-Maker Reward Screen v1

> **Rejected stale snapshot. No accepted edge, profitability, or trading
> authority.** A hypothetical paired quote had positive both-fill economics,
> but the books failed freshness and public data proves no reward payout or
> orphan-fill control.

**Scope correction:** this Moonshot condition is not BTC, ETH, or SOL and is
outside the repository's frozen Polymarket research boundary. The snapshot is
retained as a negative methodology audit, but it must not be rerun, captured
prospectively, or used for promotion. The canonical correction is
[`paired-maker-reward-scope-adjudication-v1.json`](paired-maker-reward-scope-adjudication-v1.json).

This study is independent of market direction. It considers two resting BUY
orders of equal size, one for YES and one for NO in the same binary condition.
If both fill, the shares form a complete set worth 1 pUSD at settlement or
merge. The strategy does not require predicting which outcome wins. Its primary
risk is asymmetric execution: one order can fill while the complement does not.

Polymarket's [liquidity-reward methodology][reward-method] applies a quadratic
score to qualifying resting orders, boosts balanced two-sided liquidity, and
normalizes each maker against all makers. The [fee documentation][fees] states
that makers are not charged trading fees. This candidate's canonical Gamma fee
schedule independently recorded `takerOnly=true`.

## Corrected Conservative Bound

For every maker, `Qmin <= Q1 + Q2`. Therefore:

```text
sum(old Qmin) <= sum(old Q1) + sum(old Q2)
```

The conservative instantaneous share denominator is consequently the sum of
the two old aggregate scores plus the hypothetical maker's own `Qmin`. Using
`min(sum(old Q1), sum(old Q2))` would overstate the lower bound and is forbidden.
The implementation has direct regression coverage for this correction.

Public order books mirror binary orders across YES and NO representations. A
physical BUY YES at price `p` is also a visible ASK NO at `1-p`, and conversely.
The frozen snapshot's conditional calculation omitted those hypothetical
complementary own asks when constructing its post-quote midpoints. Its
conditional share, daily-equivalent, and payback values are therefore invalid
and must not be used. The both-fill complete-set arithmetic and one-sided
maximum settlement-loss arithmetic remain valid. The active implementation
now includes the mirrored own asks while scoring each physical order only once.
Public books still do not expose maker grouping, queue position, persistence
across random samples, or the second epoch normalization.

## Source-Bound Result

The frozen candidate was the Moonshot Chinese-AI ranking condition with a
20-share minimum, 4.5-cent maximum reward spread, and 6 pUSD/day reward
allocation. The event is augmented negative-risk, so the study makes no
event-wide complete-set assumption; it uses only this condition's binary
YES+NO identity.

One-tick-improved hypothetical bids were:

| Quantity | YES bid | NO bid | Combined | Both-fill gross | Maximum orphan settlement loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.469 | 0.471 | 0.940 | 1.200 pUSD | 9.420 pUSD |

The public methodology does not define how its size-cutoff-adjusted midpoint is
constructed. The frozen artifact labeled its post-quote top-midpoint score as
conditional, but that calculation also omitted the hypothetical orders'
mirrored own asks. Its reported 4.9667% instantaneous share equivalent, 0.2980
pUSD/day, and 31.61 idealized payback are invalidated diagnostics, not payout
forecasts or lower bounds on actual earnings. The publicly proven reward payout
lower bound is exactly zero.

The book timestamps were synchronized with each other but the oldest was 8,074
ms behind receipt, failing the frozen 5,000 ms age gate. The snapshot is
therefore rejected before queue, fill, cancellation, adverse-selection, merge,
or persistence work.

The authoritative evidence is
[`paired-maker-reward-snapshot-v1-2026-08-25.json`](paired-maker-reward-snapshot-v1-2026-08-25.json),
result SHA-256
`3ed963fe2ff3473dba6c9b5146d842130d4f67ed3a0e8673451330133c68c0b0`.
Its test reconstructs the preserved artifact hash, implementation hashes, and
historical conditional quote diagnostic from the embedded books. That
reconstruction proves provenance, not correctness; the scope adjudication
records the methodology invalidation.

## Reopening Gate

Do not turn this into an order-placement loop or repeatedly poll snapshots.
Do not reopen this candidate. A distinct BTC/ETH/SOL market may advance only
under a frozen prospective public capture contract that records reward
configuration drift, book changes, quote persistence opportunities, trade
arrivals, and conservative orphan-risk proxies. Authenticated order, queue,
fill, reward, and cancellation evidence would still be required before any
paper or live promotion. No credential was used and no order was placed.

The first separately frozen BTC/ETH/SOL source screen stopped before books when
Gamma and the exact CLOB reward endpoint disagreed on BTC reward configuration.
Its contract permits no resampling. The terminal
[`attempt receipt`](crypto-paired-maker-reward-screen-attempt1-failure-v1.json)
preserves the failure and the missing-payload limitation.

[fees]: https://docs.polymarket.com/trading/fees
[reward-method]: https://docs.polymarket.com/programs/liquidity-rewards
