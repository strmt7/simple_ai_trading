# Exact quote principal before roundtrip integration

The unchanged Spot roundtrip can spend foreign quote balance. A constructed
first sale receives 100 quote units; with total wallet cash 1,000 and an old
price of 100, its sizing helper returns one base unit. Buying that unit at a
later price of 200 requires 200 quote units. The first sale finances only 100.
This reproduces the defect offline; these are supplied prices, not market data.

The new `BinanceClient.place_spot_quote_buy` primitive instead transmits an exact
decimal-string `quoteOrderQty`, without a base `quantity` or price conversion.
It requires an explicit client ID and Spot scope, retains testnet/demo and
BTC/ETH/SOL restrictions, and checks scope again at the native request boundary.
It does not create an automatic recovery ID or retry writes.

`SpotQuoteOrderBinding` binds original quote amount and cumulative principal on
both acknowledgment and exact-ID query. A changed quote amount or principal
above the request rejects. Venue-derived base quantity is separate. Quote-query
recovery requires the explicit scope. The old base-quantity binding keeps its
same syntax checks through a shared helper; its response validation is unchanged.

The [already admitted immutable REST source](../spot-quote-budget-source/review.md)
provides the amount-unit semantics. No new source, market, account, credential
or native order request followed. Current symbol permission/precision, complete
fills, fees and account balances are not qualified by that document or this API.

## Verification and remaining work

The [source-bound evidence](result.json) records 236 distinct focused cases:
233 existing/new API cases, followed by three added real-request-path cases on
unchanged production bytes. HTTP is replaced only at the transport boundary in
the latter lane; exact signed URL serialization, no POST retry after HTTP 503,
and scope drift between admission and transmission are exercised. All pass.
The earlier 144-case phase and one failed pre-test Windows executable launch
remain retained. This is not one final 236-case run or a whole-suite claim.

Only `get_order` changes among existing `BinanceClient` method ASTs; the quote
BUY method is new. The CLI roundtrip is unchanged and still needs repair.
An accepted principal is not a total debit cap: commissions and other costs are
separate. No current user flow calls the new primitive.

Complete the operation as one durable workflow before connecting the command:

1. Reserve an explicit owned initial allocation under the shared execution
   namespace, alongside existing unresolved opening/closing barriers. Native
   wallet balance is availability evidence, not ownership of the whole balance.
2. Persist exact scope, operation/leg IDs, amount units, fee/cost allocations and
   UNKNOWN before either write. Do not encode a quote amount as a base position.
3. Require complete bound terminal first-leg fills/commissions before allocating
   its cash or received base. Do not use cumulative gross cash as net proceeds,
   infer absent fees, spend another asset without allocation, or re-send UNKNOWN.
4. Persist the second-leg principal and fee reserve before transmission. Recover
   each exact selector after interruption; partial fills/residuals and unknown
   fees remain obligations. A query result alone does not release cash or rearm.
5. Demonstrate the native capability/precision and fee treatment, process-exit
   recovery, competing admission and both buy/sell orderings. Preserve operator
   Stop and owned reduction; account-wide fencing remains a separate boundary.

The operation/journal integration, commission qualification and command migration
are unfinished. No principal-only primitive qualifies roundtrip capital safety,
profitability, a stable market edge or institutional readiness. This report
records the local source phase prepared for a coherent development batch.
