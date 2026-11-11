# Binance Spot Auto Order Routing

Binance documents venue-managed routing for eligible Spot orders. The
intermediary quote conversion carries no separate commission and still
has a spread. Spot commission follows the pair actually executed and can
be deducted in the intermediary asset. The FAQ describes selected U, USD,
USD1, USDC and BRL pairs. The initial rollout notice named U pairs first.
Specific symbols and regional access still require evidence.
[Launch notice][launch] · [Routing FAQ][faq]

Routing applies to immediate fills. The feature can use the original
pair for the remainder, and conversion capacity is discretionary with
no guaranteed minimum. The displayed synthetic book is indicative.
The export described in the FAQ contains the original fill prices and
applied conversion rates. An exact comparison would also reconcile
received amounts, fee assets, residual quantities and order status.
The public sources establish the trading interface and supported order
types. API and testnet availability remain unverified. [Routing FAQ][faq]

For an independently required sale into U, compare final U proceeds at
the same base quantity. In the constructed full-fill example, the direct
route nets 999 U. An intermediary sale of 1,001 USDT, after a 0.1% trading
fee and conversion at 0.9985 U per USDT, nets 998.4990015 U. Avoiding a
second 0.1% commission improves this result by 0.9984990015 U against a
manual route with the same conversion. It still loses 0.5009985 U against
direct execution. Break-even against direct execution requires a rate
of `1000/1001` U per USDT before adding any other costs. This example
tests the comparison method and makes no claim about the router's actual
selection or prices.

This is a direction-independent cost mechanism for existing organic flow.
The two retained records support a conditional one-commission comparison.
Executable improvement, fill capacity and persistence require matched
evidence. No spot graph, quotes, account calls or orders followed.
[Source and arithmetic evidence](evidence.json)

[launch]: https://www.binance.com/en/support/announcement/detail/2f6fa2926253415a81def145af35014e
[faq]: https://www.binance.com/en/support/faq/detail/1dce8dc66adc450a8e669676d64a5135
