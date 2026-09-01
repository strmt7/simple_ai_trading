> For the complete documentation index, see [llms.txt](https://support.backpack.exchange/llms.txt). Markdown versions of documentation pages are available by appending `.md` to page URLs; this page is available as [Markdown](https://support.backpack.exchange/technical-docs/trading/futures-specs.md).

# Futures Specs

### Overview

* All subaccounts on Backpack are cross-margined. Margin is isolated per subaccount.
* There is only one wallet to access all products (spot, futures, spot margin, borrow/lending).
* Currently, markets are denominated and settled in USDC.
* Lent assets can be used as collateral to open and maintain futures positions. Interest rates are determined by the public utilization rate curve of the Borrow Lend market.
* PnL is continuously realized by default and counts toward net equity. Realized PnL earns or pays interest based on your borrow/lend exposure—surplus earns interest if Auto-Lend is enabled, while borrowed balances incur interest by default.
* Liquidations first go through the orderbook, and upon hitting the auto-close margin, accounts are liquidated against Backstop Liquidity Providers.

### Margin & Collateral

<table data-header-hidden data-full-width="true"><thead><tr><th width="192"></th><th width="245"></th><th></th></tr></thead><tbody><tr><td><strong>Item</strong></td><td><strong>Description</strong></td><td><strong>Formula</strong></td></tr><tr><td>Collateral Value</td><td>Notional value of collateral asset with haircut applied</td><td><p></p><div class="math math-display">\text{TokenSize} \times \text{MarkPrice} \times \text{CollateralWeight}</div></td></tr><tr><td>Total Account Collateral</td><td>Total value of collateral assets with haircut applied</td><td><p></p><div class="math math-display">\sum \text{(Collateral Value)} \text{ for all collateral assets}</div></td></tr><tr><td>Unrealized PnL</td><td>Position Unrealized PnL</td><td><p></p><div class="math math-display">\text{Position Size} \times (\text{Mark Price} - \text{Average Entry Price})</div></td></tr><tr><td>Net Exposure Quantity</td><td>Size in tokens of current open positions and open orders that increase risk within a symbol</td><td><p></p><div class="math math-display">\text{Position Size} + \text{Open Order Size Of Orders Increasing Risk}</div></td></tr><tr><td>Net Exposure Notional</td><td>Notional size of current open positions and open orders that increase risk within a symbol</td><td><p></p><div class="math math-display">\text{Net Exposure Quantity} \times \text{Mark Price}</div></td></tr><tr><td>Total Exposure Notional</td><td>Total notional sum of open positions and orders that increase risk across all symbols</td><td><p></p><div class="math math-display">\sum \text{(Net Exposure Notional)} \text{ across all futures and spot margin positions}</div></td></tr><tr><td>Base IMF</td><td>Position IMF without considering size</td><td><p></p><div class="math math-display">\max\left(\dfrac{1}{\text{maximum leverage on platform}}, \dfrac{1}{\text{maximum leverage set by user}}\right)</div></td></tr><tr><td>Position Initial Margin Fraction (IMF)</td><td>Initial margin requirement for a position adjusted for position size</td><td><p></p><div class="math math-display">\max \left(\text{Base IMF}, \text{IMF Factor} \times \sqrt{\text{Notional Position Size}}\right)</div></td></tr><tr><td>Position Maintenance Margin Fraction (MMF)</td><td>Maintenance margin requirement adjusted for position size</td><td><p></p><div class="math math-display">\max \left(\text{Base MMF}, \text{MMF Factor} \times \sqrt{\text{Notional Position Size}}\right)</div></td></tr><tr><td>Account IMF</td><td>Minimum margin fraction required to open new positions</td><td><p></p><div class="math math-display">\max\left(\dfrac{1}{\text{max leverage}}, \dfrac{\sum \text{(Notional Position Size} \times \text{Position IMF)}}{\text{Total Exposure Notional}}\right)</div></td></tr><tr><td>Account MMF</td><td>Minimum margin fraction required to not get liquidated.</td><td><p></p><div class="math math-display">\dfrac{\sum \text{(Notional Position Size} \times \text{Position MMF)}}{\text{Total Exposure Notional}}</div></td></tr><tr><td>Net Equity</td><td>Total net equity value of the account.</td><td><p></p><div class="math math-display">\text{Total Collateral Value} + \text{Total Unrealized PnL} + \text{Unsettled Balances} - \text{Total Borrow Liability}</div></td></tr><tr><td>Net Equity Locked</td><td>Total equity used to maintain open positions and orders that increase risk.</td><td><p></p><div class="math math-display">\text{Initial Margin Fraction} \times \text{Total Exposure Notional} \text{ summed across all token market positions}</div></td></tr><tr><td>Net Equity Available</td><td>Equity available to open new positions.</td><td><p></p><div class="math math-display">\text{Net Equity} - \text{Net Equity Locked}</div></td></tr><tr><td>Account Margin Fraction (MF)</td><td>How levered an account is given its current active positions and the mark prices of the coins.</td><td><p></p><div class="math math-display">\dfrac{\text{Net Equity}}{\text{Total Exposure Notional}}</div></td></tr><tr><td>Auto Close Margin Fraction</td><td>Margin fraction in which account is liquidated against BLPs.</td><td><p></p><div class="math math-display">\max \left(\dfrac{\text{Account MMF}}{\text{ACMF Divisor}}, \text{Account MMF} - \text{ACMF Offset}\right)</div></td></tr></tbody></table>

***

### Mark Price and Index Price

We have fallback logic to calculate the Mark Price. The order of preference is:<br>

1. Index price + 1 minute EWMA of (mid price - index price) delta. The mid price is the mean of the best bid and best offer.
2. Index price.
3. Median of the best bid, best offer and last traded price on Backpack.
4. Mid price (mean of best bid and best offer) on Backpack.
5. Last traded price on Backpack.

We will resort to using the fallback logic if we do not have the necessary data (e.g. it is stale).

Markets in post only state will use the index price as the mark price until the order book state changes to open.

For **Index Price**, we retrieve market price data from a set of exchanges.

* For each exchange, we take the median of {best bid, best ask, last price} and use this as the market price.
* We then calculate the median market price for the exchanges.
  * We apply a ceiling price of 100bps above the median and a floor price of 100bps below the median. If an exchange is outside of that, then they will be given the ceiling / floor price.
* Each exchange is given a weighing that affects their weight. We use that weighting in a weighted mean average market price calculation for the set of exchanges.

#### **Mark Price Bounds**

We apply a constraint to prevent abnormal mark prices from affecting user positions. If the mark price deviates from the index price beyond a percentage threshold, Backpack will bound the mark price by that threshold.

**Example:** If the index price is 100 and the threshold is 5%, the mark price will be constrained to the range \[95, 105].

This protection:

* Prevents abnormal mark prices caused by manipulation or unusual market conditions
* Protects user positions from unexpected liquidations

*Note: This constraint may be relaxed for certain new listings where more volatile market conditions are expected.*

***

### Funding Rate

*Note: on Wednesday August 20 at 08:00 UTC, the funding rate intervals across all perpetual futures markets were changed to hourly.*

Funding rate = Clamp\[(mean\_premium\_index + Clamp(interest\_rate – mean\_premium\_index, -0.05%, 0.05%)) / 8, Funding rate cap, Funding rate floor]

**The funding rate is calculated as follows:**<br>

* Every second we record the premium index
* premium = (mark price − index price) / index price
* Average premium index = moving-average of those per-second premiums over the funding interval.
* Interest-rate add-on
  * interest\_rate = 0.03 % × (funding\_interval\_hours / 24)
* The funding rate floors and ceilings are set per market and can be found on the market specs [page](https://backpack.exchange/market-info/market-specs/futures) or fetched via API using the [markets endpoint](https://docs.backpack.exchange/#tag/Markets/operation/get_markets).
* Funding payments are debited/credited at the end of the interval and are calculated as follows:
  * payment = funding rate × position quantity × mark price

***

### Price Bands

#### Limit Price Bands

* These are applied to limit orders.
* We calculate the median of {best bid, best offer, last price}. This is called the Active Price.
* We then have a configurable Max Multiplier and Min Multiplier parameters.
* If a limit order is submitted with a limit price that exceeds Active Price \* Max Multiplier, then it is rejected. Conversely, if it has a limit price below Active Price \* Min Multiplier, then it is rejected

#### Price Impact Bands

* These are applied to taker orders.
* We have parameters Max Impact Multiplier and Min Impact Multiplier.
* If we have a buy taker order, we will only allow it to to take up to the price level of best offer \* Max Impact Multiplier. If the order is not fully filled at that point then we will expire the order and it will be partially filled. Converse logic applied for Min Impact Multiplier.

#### Mean Mark Price Bands

* We calculate the 5 minute moving mean average for the mark price, mean mark price
* We then have 2 parameters Max Multiplier and Min multiplier
* If a taker order is submitted, then we will allow it to fill up until the price level of mean mark price \* Max Multiplier . If the order is not fully filled at that point then we will expire the order and it will be partially filled. Converse logic applied for Min Multiplier . This is similar to the above price impact price bands.

#### Mean Premium Bands

* We calculate the 5 minute moving average of the premium, mean premium
* We then have a parameter tolerance pct
* If, e.g. the tolerance percentage is 1% and the mean premium is 3%, then if a taker order is submitted, we will only allow it full up to a price level at which the current premium is 4%.  If the order is not fully filled at that point then we will expire the order and it will be partially filled.
* If, e.g. the tolerance percentage is 1% and the mean premium is -3%, then if a taker order is submitted, we will only allow it full up to a price level at which the current premium is -4%.  If the order is not fully filled at that point then we will expire the order and it will be partially filled.


---

# Agent Instructions
This documentation is published with GitBook. GitBook is the documentation platform designed so that both humans and AI agents can read, navigate, and reason over technical content effectively. Learn more at gitbook.com.

## Querying This Documentation
If you need additional information that is not directly available in this page, you can query the documentation dynamically by asking a question.

Perform an HTTP GET request on the current page URL with the `ask` query parameter, and the optional `goal` query parameter:

```
GET https://support.backpack.exchange/technical-docs/trading/futures-specs.md?ask=<question>&goal=<endgoal>
```

`ask` is the immediate question: it should be specific, self-contained, and written in natural language.
`goal` is optional and describes the broader end goal you are ultimately trying to accomplish on behalf of the user. GitBook uses it to tailor the answer towards what is most useful for that goal.

The response will contain a direct answer to the question and relevant excerpts and sources from the documentation.

Use this mechanism when the answer is not explicitly present in the current page, you need clarification or additional context, or you want to retrieve related documentation sections.
