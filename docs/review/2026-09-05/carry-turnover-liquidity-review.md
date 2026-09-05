# Carry turnover and funding-liquidity audit

This zero-request retrospective audit uses all 17 original symbols and the
complete retained training/validation/test role union. It changes neither the
historical results nor acceptance. The [frozen plan](carry-turnover-liquidity/plan.json)
discloses that prior aggregates were already known. The
[result](carry-turnover-liquidity/result.json) and durable journal bind the
published source bytes and implementation; no ignored-cache fallback is used.

## Findings

Nine of 17 symbols have positive funding less one modeled round-trip allowance;
eight remain positive when three role-separated allowances are charged. None
is positive after one allowance plus the unchanged original role-weighted
capital stress. These are funding subtotals, not measured profit or account ROI.

| Symbol | One allowance, bp | Three allowances, bp | Funding-only prefund, bp | Funding cash drawdown, bp |
| --- | ---: | ---: | ---: | ---: |
| BTCUSDT | 87.66 | 29.49 | 30.87 | 32.36 |
| ETHUSDT | 60.84 | 5.49 | 22.50 | 25.76 |
| SOLUSDT | -33.98 | -88.28 | 64.36 | 65.56 |

Every number is normalized to that symbol's first retained funding reference
mark for one base unit, not actual capital invested. The 32-bp allowance per
round trip is a sensitivity, not measured commission. Role durations differ
(BTC approximately 99/33/33 days); references are not executable entry/exit
prices. Joining roles illustrates turnover sensitivity, not out-of-sample
validation. No symbol or exit is selected from these observed outcomes.

Prefunding is the negative minimum cumulative funding balance, including zero
initial balance and retaining every credit. Drawdown is the greatest decline
from any earlier cumulative funding peak. Neither measures total strategy loss
or sufficient liquidation protection. Spot/perpetual mark-to-market, basis,
collateral variation, borrowing, fees and conversion remain absent. Completeness
means no retained settlement was lost between joined roles, not independent
proof that the original exchange response omitted none.

## Research consequence

An integrated ledger must distinguish entry/exit turnover from statistical
evaluation boundaries and funding receipts from interim financing needs.
Keep the old capital stress explicitly hypothetical: failing it does not prove
negative expected value at every feasible financing cost. Conversely, positive
funding alone does not prove a profitable hedge. Any future strategy selection
or causal exit/abstention model needs independently precommitted evidence;
this training-inclusive sample cannot validate it.

Session routing correction: absent mainnet read-only fee credentials block that
specific account-evidence branch, not public-only or materially useful offline
R&D. Do not repeat credential checks or declare the entire investigation blocked
on that fact alone. Testnet credentials are not mainnet evidence. No credential,
account, order, new market request or protected capture was used here.

The existing broad-funding family receives this explanatory attachment only:
37 accepted scopes, 65 hypotheses, 189 terminal observations and zero current
fully qualified edges remain unchanged. No old capture retry is authorized.
