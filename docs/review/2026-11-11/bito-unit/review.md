# BITO split and hedge continuity

ProShares announced a 1-for-2 reverse split before the November 19 open.
The ticker remains BITO while the CUSIP changes from `74347G440` to
`74350X405`. The issuer says fractional shares created by the split
are redeemed for cash through brokers. Those are ETF shareholder terms.
[ProShares disclosure][issuer]

Binance plans a separate perpetual transition. Strategy bots close at
00:45 UTC without automatic reinstatement. The ordinary transition begins
at 01:00 UTC. The exchange plans to cancel open orders, including
reduce-only protection, suspend trading and funding during the halt, settle
unrealized P&L to cross margin, and reinstate ordinary positions after
adjustment. For previously isolated positions, the notice ties the new
isolated transfer to initial margin. The table schedules a post-only
phase from 03:25 to 03:30 UTC and last-price protection through 14:30 UTC.
Binance describes the timing as subject to the actual transition and
liquidity. The session called cancel-only permits certain post-only order
operations. [Binance notice][exchange]

At equivalent split references, ten old shares at 20 and five new shares
at 40 are both worth 200. A price-only series can display an apparent
100% return from the conversion. If the short hedge is absent while the
five shares fall from 40 to 36, the long leg loses 20 before costs. An
adjusted candle series cannot establish continuous exposure, protective
orders or isolated liquidity. Replay the separately timestamped close,
cash settlement, cancellation, position reinstatement and available
margin. Qualify every timestamp and quantity in the actual position units.

The [evidence](evidence.json) binds two primary records and the
[exact replay](derive.py). No exact Polymarket BITO counterpart is
established. Actual native quantities, re-entry cash, funding entitlement,
live session timing and executable returns remain unverified. No quotes,
funding, market orders or economic observation were added.

[exchange]: https://www.binance.com/en/support/announcement/detail/dfd8cf5e698146ce83b560026529a752
[issuer]: https://www.proshares.com/press-releases/proshares-announces-share-splits
