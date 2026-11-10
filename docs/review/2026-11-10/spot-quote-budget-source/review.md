# Spot MARKET amount units and roundtrip budgeting

The pinned official [Binance Spot REST document](https://github.com/binance/binance-spot-api-docs/blob/2bae3f5bd622ebfdea593daed066df14dc7e7362/rest-api.md)
was selected from publisher revision metadata. Its initial GitHub HTML capture
returned HTTP 200 and native success but only 8,737 Markdown characters, without
the required API content. It failed admission and remains preserved unchanged.

Under the existing technical-document delegation, a separately frozen one-use
capture of the same publisher's immutable raw copy completed with robots checks,
bounded hidden execution and no retries. Its 207,075 extracted characters passed
the declared URL, status, success, size, title and four required-term gates.
The [adjudication](adjudication.json) binds both captures, their contracts and
native digests. No alternate page reader or challenge bypass was used.

The document distinguishes base `quantity` from quote `quoteOrderQty`. A MARKET
buy with the latter spends a requested quote amount, while a sell targets quote
receipts. Liquidity determines base execution, with lot-size constraints.
The source includes `quoteOrderQtyMarketAllowed`, `origQuoteOrderQty`, cumulative
quote totals and fill commissions. These are source semantics and examples,
not evidence of current per-symbol permissions or a complete account-cost bound.

The current sell-then-buy diagnostic sizes its return buy from total wallet
quote balance. That can spend funds beyond its first sale's proceeds. A future
repair needs an operation-owned cash allocation, explicit amount units, exact
per-leg scope/client identities, native fee and partial-fill accounting, and
durable recovery between legs. A base quantity estimated from price cannot prove
a quote spending bound. A quote amount alone does not prove commission coverage,
final acquired quantity or a riskless roundtrip. No transport, sizing rule or
order authority changed on the strength of this source.

No economic observation, account request, order, credential use or accepted-edge
claim follows. Protected data and consumed market populations remain untouched.
