# Polymarket US fee supersession

Two native Crawl4AI captures returned HTTP 200, substantive text and the exact
intended official documents. The new changelog source independently announced
replacement of the flat maker rebate from October 19. That material change
authorized one separately frozen refresh of the September 1 fee URL. Both
request journals and complete native results remain private. The original
September source, deterministic grid and decisions retain their original scope.
[Official changelog](https://docs.polymarket.us/changelog)

The published base maker tiers now give the following standard-trade rebates.
Tiers use the preceding 30 completed ET calendar days, excluding today, with
daily recalculation. Overrides and accelerated placement require separate
account evidence. Maker, taker-fee and taker-rebate tiers are independent.
[Official fee schedule](https://docs.polymarket.us/fees)

| Maker tier | Qualifying maker volume | Rebate coefficient magnitude | Rebate on 100 contracts at $0.50 |
| --- | --- | --- | --- |
| 1 | Below $10M | 0 | $0.00 |
| 2 | $10M to below $50M | 0.0035 | $0.09 |
| 3 | $50M to below $200M | 0.0070 | $0.18 |
| 4 | At least $200M | 0.0104 | $0.26 |

The retained flat coefficient would give $0.31 on that constructed fill.
Current standard taker coefficients range from 0.095 to 0.0695 by taker tier.
The old 50-percent rounded-fee-share grid cannot qualify this new schedule.
Each eligible maker fill retains an absolute rounding-error bound of $0.005.

Fill grouping matters. In a constructed taker order with two 100-contract fills
at $0.10 and coefficient 0.0695, independently rounded fees total $1.26.
The documented cumulative ceiling is $1.25, requiring at least a $0.01
adjustment. Maker rebates are independent per fill. At coefficient 0.0104
they total $0.18, while incorrectly coalescing those fills produces $0.19.
Exact collected fees still require native fill receipts.
[Official fee rules](https://docs.polymarket.us/fees#standard-fee-rules)

The source schedules combo taker changes for November 11 at 18:00 UTC and a
combo maker fee for November 14 at 05:00 UTC. Both were future announcements
at capture. At 100 contracts and $0.50, the old combo taker formula gives
$2.00. The announced formula gives $2.56 at taker Tier 1 and $1.92 at Tier 4.
The announced maker formula gives $0.75. A smaller fixed term therefore has
different cash consequences across tiers. The onboarding terms assign the taker
coefficient at insertion. The combo schedule states applicability by fill time.
Bind both clocks and the actual fill role before valuing an execution.
[Official combo schedule](https://docs.polymarket.us/fees#combo-trading)

The [exact derivation](derive.py) and [sealed evidence](evidence.json) bind these
source facts and constructed arithmetic. This supersedes active use of the old
flat US rebate inputs. Account eligibility, actual assigned coefficients,
activation, owned receipts and a profitable underlying strategy remain
unproved. The public forward profit floor stays zero. No book, account, quote,
order or training request followed. Financial counts remain 206 observations,
65 hypotheses, 37 historical scopes and zero qualified stable edges.
