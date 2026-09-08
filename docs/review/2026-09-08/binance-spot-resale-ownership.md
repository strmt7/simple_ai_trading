# Spot buy-then-sell: preserve pre-existing base inventory

[Source-bound review](binance-spot-resale-ownership.json).

At baseline `60d72b99`, `_roundtrip_second_quantity` used the lesser of gross
executed quantity and the entire free wallet balance. An offline example with
1 pre-existing BTC, a 1 BTC purchase and 0.001 BTC received-asset commission
planned a 1 BTC resale despite receiving only 0.999 BTC. The missing 0.001 BTC
would come from pre-existing holdings if the SELL filled.

The new shared `spot_buy_received_quantity` decoder requires exact supported
base/quote identity, terminal MARKET BUY state, reconciled fill/cash totals
and complete native commission rows. Base commissions reduce sale inventory;
quote and BNB fees do not reduce base quantity or imply a quote valuation.
Negative commissions are not admitted. Terminal canceled/expired partial fills
may supply their proven received quantity; a still-live partial order cannot.

The active round-trip SELL uses the lesser of that received amount and available
balance, floors to the adapter's eight-decimal wire precision, and rejects any
normalizer or wire rounding above the limit. The command-level regression buys
1, sells 0.999 and leaves the pre-existing 1 BTC intact. Missing commission
evidence produces a recorded partial failure with no resale order, rather than
assuming zero fees or using wallet inventory as proof of acquisition.

## Source and financial interpretation

Binance's [current Spot commission FAQ](https://developers.binance.com/en/docs/products/spot/faqs/commission_faq)
was inspected September 8; it identifies received quantity as the BUY commission
base and explains received-asset deduction versus BNB payment. Its example
rates/prices are explicitly fictional and were not used as current economics.
This was web-tool documentation inspection, not retained raw HTTP/account data.

The native quantity limit does **not** complete native PnL or cost accounting,
guarantee second-leg execution, or establish a profitable round trip. The
sell-then-buy branch still uses its existing wallet-based cash sizing. The direct
CLI still lacks complete write-ahead/account-scoped crash recovery. Autonomous
openings still store gross quantity and modeled fees; their intent validator
compares recorded FILLED quantity to original order quantity. That path needs
explicit gross/net receipt fields and integrated durable migration, not a blind
quantity subtraction that would conflict with original intent validation.

No account, credentials, order, funds, current book or protected capture was
accessed. Existing user ledgers and frozen outcomes were untouched. Only the
named production boundaries were semantically reviewed; whole-file hashes do
not certify all CLI lines reviewed.

## Verification and research routing

100 affected-domain checks passed: 46 initial new receipt/quantity cases, 50
shared acknowledgement cases and four CLI helper/command cases. Three further
terminal-partial cases and one installed-entrypoint/native-launcher parity check
passed separately: 104 distinct, including 49 new cases. Tests stub all exchange
I/O. Ruff and diff-whitespace checks passed. No unchanged training, GPU benchmark,
full CI suite or market observation was repeated.

Before this repair, four targeted official-source search queries failed to
establish a new eligible stablecoin Launchpool announcement. This is a bounded
discovery result, not proof that no campaign exists. A USDC headline surfaced
on a currently crawled category page, but the [identified announcement](https://www.binance.com/es/support/announcement/detail/0c94ed2f7a7d4e79a0a1aef232917690)
belongs to October-November 2025. It was excluded at discovery without a
collector or economic calculation. No research family was terminalized or
duplicated for this stale lead. Current counts remain 37 mechanism scopes,
65 hypotheses, 197 terminal observations and zero qualified profitable edges.
