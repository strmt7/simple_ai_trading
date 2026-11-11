# November 12 BTC ladder candle clock

The frozen Gamma GET returned HTTP 200 and 11 market rows for event
`1320610`. The event and all 11 descriptions explicitly say the Binance
BTC/USDT one-minute candle is *closing at* noon Eastern. The older retained
wording names the noon candle without identifying the bar boundary. The
frozen exact wording test failed before the screen read any prices. The
[contract](contract.json), [terminal result](result.json) and
[metadata-only diagnostic](clock-diagnostic.json) preserve that sequence.

Noon Eastern on that date corresponds to 17:00 UTC. Mapping the new rule to
the candle opening at that instant risks a one-bar error. The two
contracts in a proposed hedge must reference the same exact settlement
sample. For example, suppose an earlier close is 73,000 and a later
close is 69,000. A higher-threshold NO at 72,000 on the earlier sample
and a lower-threshold YES at 70,000 on the later sample both pay zero.
Those values are constructed. Historical rule rollout and native kline
mapping require their own evidence.

The [frozen checker](adjudicate.py) also contains a general result for
future rule-complete ladders. For increasing thresholds, let `a[i]` be
the YES ask and `b[i]` the YES bid used to proxy NO cost as `1-b[i]`.
The smallest decreasing YES valuation above every available bid is the
suffix maximum of the bids. If each available ask is at least that
valuation, consecutive differences give nonnegative weights on the
ordered settlement states. They sum to one and value each available YES
and NO below its acquisition cost or at equality. Every nonnegative
portfolio then has at least one state whose payout is at most its cost.
If a valuation exceeds its ask, uncrossed individual markets imply a
higher threshold supplies the bid, which gives a strict two-leg witness.
Four constructed preflight cases cover those outcomes, a missing side
and a rule mismatch. An independent enumeration of possible state weights
agreed in all 2,197 synthetic three-threshold price configurations.
[Certificate verification](certificate-validation.json)

The exact November 12 event is consumed. No books or fees were requested.
The current result is a rule-admission failure with zero economic
observations. Independent condition disputes and other exceptional
settlements remain outside the ordinary-state proof.
