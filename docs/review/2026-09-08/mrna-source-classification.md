# MRNA public endpoint qualification

Reviewed September 8, 2026 before the MRNA economic observation. These are
official documentation pages inspected with the web tool, not byte-retained
HTTP market observations. No example price is an economic input.

- [Binance Spot market API](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market#ticker-book-ticker)
  identifies GET `/api/v3/ticker/bookTicker` on `api.binance.com`, single-symbol
  weight 2, best bid/ask prices and quantities, and an unsigned curl request.
  The [official public-market FAQ](https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md?plain=1)
  independently lists this method among unauthenticated public market data.
- [Binance USD-M market API](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#symbol-order-book-ticker)
  identifies GET `/fapi/v1/ticker/bookTicker` on `fapi.binance.com`, weight 2 for
  one symbol, an unsigned request without an API-key header, and best bid/ask
  prices and quantities. RPI orders are excluded. Its `time` is transaction
  time, not a documented quote-update freshness field.
- [USD-M security contract](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/general-info#endpoint-security-type)
  distinguishes freely accessible NONE from key-requiring MARKET_DATA and
  signed account/trading methods. Neither selected ticker operation declares
  a keyed/signed requirement or account mutation. Classification is the
  documented public ticker operation, not an inference from HTTP GET alone.

The study uses only these two exact single-symbol requests, sequentially,
without keys, signatures, proxies, redirects or retries. A source error stops
the remaining leg. Maximum cumulative weight is 4 across the two API hosts;
there is no pagination or polling. Response retention is bounded at 10,000
bytes plus one overflow-detection byte per request. Existing bounded transport
fsyncs the exclusive intent, body and completed receipt. Socket timeout is ten
seconds; the thirty-second read budget is checked between chunks, not a hard
wall-clock deadline.

Identity/multiplier evidence is the immutable September 4 inventory result
`ba5ebb29ce89c8cc09bde5066bbe4a0cfc7dc11fa926f5a00de6321054caba26`.
It proves that retained inventory, not today's account eligibility or unchanged
corporate-action state. A positive price prefilter would still require current
contract/conversion qualification before any trading conclusion.
