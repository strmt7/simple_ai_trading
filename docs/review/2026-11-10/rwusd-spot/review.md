# Compare RWUSD exits in USDC on the same date

The selected Binance announcement was a distinct lead for RWUSD spot trading
and Cross Margin. The native crawl returned HTTP 202 and one character of
extracted text. Current trading terms, redemption rights for spot-purchased
units and account applicability remain unverified. The older Futures collateral
evidence covers a different product and period.
[Binance announcement](https://www.binance.com/en-AE/support/announcement/detail/1dfe10f0e0fb4e7ab0eb8874e9fdab0e) ·
[Source evidence](evidence.json)

For an independently required exit of an existing holding, compare two complete
disposals of the same quantity. Use usable USDC at a common horizon, such as
the latest qualified redemption completion. Under a quote-denominated spot
fee, multiply net USDT proceeds by the executable USDC per USDT conversion,
then apply the value of having those proceeds available earlier. Compare this
amount with redemption cash after fees and settlement. Include all other
route costs at the same horizon. Use the lower redemption cash bound and the
upper spot cash bound to certify a strictly better redemption exit. If either
route cannot dispose of the full quantity, account for its residual or leave
the comparison unresolved. If the proceeds are needed by a deadline, a route
is feasible only if its usable cash arrives by then.

The exact calculation uses 1,000 illustrative units, 999.50 USDC
in redemption proceeds, a 0.998 USDT spot bid per unit and a 0.10% spot fee.
With a 1:1 conversion and zero time value, redemption leads by 2.498 USDC.
With 1.002 USDC per USDT and a 1.001 common-horizon growth factor, spot sale
leads by 0.495000004 USDC. Holding these constructed inputs fixed, redemption
is better only below a spot bid of about 0.997506 USDT per unit. These inputs
and the threshold apply to the constructed example. [Exact calculation](derive.py)

Before a future price screen, establish current permitted use, holder
eligibility, redemption asset and fees, limits and usable cash timing through
an independently selected controlling source. Use ordinary commissions and
the actual same-account conversion route. A temporary fee reduction is
irrelevant to the long-term comparison. This source attempt produced zero new
economic observations.
