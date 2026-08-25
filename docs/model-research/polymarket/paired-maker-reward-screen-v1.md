# Polymarket Paired-Maker Reward Screen v1

> **Rejected stale snapshot. No accepted edge, profitability, or trading
> authority.** A hypothetical paired quote had positive both-fill economics,
> but the books failed freshness and public data proves no reward payout or
> orphan-fill control.

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

Public order books mirror binary orders across YES and NO representations. The
diagnostic counts the hypothetical physical orders once and allows duplicated
visible old liquidity in the denominator, which can only make this conditional
share smaller. It still is not a proven payout: public books do not expose
maker grouping, queue position, persistence across random samples, or the
second epoch normalization.

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
constructed. The artifact therefore labels its post-quote top-midpoint score as
conditional. Under that condition, double-counting old mirrored liquidity for
a conservative denominator produced a 4.9667% instantaneous share equivalent,
0.2980 pUSD/day, and 31.61 idealized days to cover the maximum orphan loss.
Those values are diagnostics, not payout forecasts or lower bounds on actual
earnings. The publicly proven reward payout lower bound is exactly zero.

The book timestamps were synchronized with each other but the oldest was 8,074
ms behind receipt, failing the frozen 5,000 ms age gate. The snapshot is
therefore rejected before queue, fill, cancellation, adverse-selection, merge,
or persistence work.

The authoritative evidence is
[`paired-maker-reward-snapshot-v1-2026-08-25.json`](paired-maker-reward-snapshot-v1-2026-08-25.json),
result SHA-256
`3ed963fe2ff3473dba6c9b5146d842130d4f67ed3a0e8673451330133c68c0b0`.
Its test reconstructs the artifact hash, implementation hashes, and conditional
quote diagnostic from the embedded books.

## Reopening Gate

Do not turn this into an order-placement loop or repeatedly poll snapshots.
Reopen only under a frozen prospective public capture contract that records
reward configuration drift, book changes, quote persistence opportunities,
trade arrivals, and conservative orphan-risk proxies. Authenticated order,
queue, fill, reward, and cancellation evidence would still be required before
any paper or live promotion. No credential was used and no order was placed.

[fees]: https://docs.polymarket.com/trading/fees
[reward-method]: https://docs.polymarket.com/programs/liquidity-rewards
