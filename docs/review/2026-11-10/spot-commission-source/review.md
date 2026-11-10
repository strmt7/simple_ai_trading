# Spot commission protocol and exact account reader

The [immutable official FAQ](https://raw.githubusercontent.com/binance/binance-spot-api-docs/2bae3f5bd622ebfdea593daed066df14dc7e7362/faqs/commission_faq.md)
passed its [one-use contract](contract.json): HTTP 200, native success, exact
original/final URL and 6,332 extracted characters. The strict robots precheck
returned explicit HTTP 404 absence; native robots checks also ran. No prior FAQ
or market sample was retried. Earlier contracts cited an older revision whose
body was not found in the inspected storage coverage.

The FAQ describes fees deducted from received assets: base on BUY, quote on
SELL. Eligible, sufficiently funded discount-asset payment has an FX-dependent
path; insufficient funds fall back to the received asset. Standard, tax and
special components all contribute, but only standard receives the returned
discount multiplier. Account side components are added once; order-test rates
already incorporate side. The figures are explicitly fictional. One worked
special-fee intermediate line disagrees with its displayed addends; the stated
sum formula and final result agree. No example percentage or intermediate
arithmetic was installed as a live rule.

## Implementation

The local [exact schedule](../../../../src/simple_ai_trading/binance_spot_commissions.py)
preserves all component rates as fractions, both routing flags, the payment
asset and charge multiplier. It distinguishes multiplier from percent saved;
only standard contributes the algebraic savings `standard * (1 - multiplier)`.
Tax/special are not discounted. Equivalent rates stay in received-asset units
before FX and native rounding; they are not actual fee-asset amounts or funding
certificates.

All rates and flags are required. Enabled routing requires asset/multiplier
evidence. When routing is authoritatively disabled, missing asset/multiplier
remain explicit `None`; no fictitious values are supplied or used. The distinct
order-test schema is rejected instead of adding side rates twice. Possible
payment assets include the received-asset fallback, including within a split fill.

`BinanceClient.get_spot_commission_schedule` requires the exact Spot scope and
supported canonical symbol, with scope forwarded through the native boundary.
It shares the existing account-commission GET transport. The legacy float
reader keeps its existing conservative calculation; it is not cash-routing
evidence. No actual account query or order was sent during development.

## Evidence and next integration

All **230** final focused API/schedule cases pass. Source was fingerprinted and
copied before launch. They cover exact arithmetic, correct side/role composition,
multiplier direction, missing/malformed data, routing flags, fallback assets,
foreign symbols, scope rotation and existing quote-order response behavior.
The earlier 116-case phase remains separately retained; do not add overlapping
counts or relabel the preceding 407-case journal phase as this revision.

Live source inspection after a source-bound semantic query found that persistent
lease/Stop/dispatch machinery is currently Polymarket-specific. Binance's
`AutonomousControl` is a filesystem state control, and the roundtrip still has
its two direct writes. Reuse a properly shared control core before command
integration; do not import a venue-branded controller as an authority substitute.
Independent session supervision remains separate unfinished work.

The roundtrip journal still admits only base/quote allocations. Positive
third-asset fees remain unresolved, not free. Add explicit fee-asset grants,
current capability/balance/fee evidence, receipt reconstruction and dispatch
authority before wiring the native workflow. Do not infer ownership of a whole
wallet, sufficient BNB or fresh fees from a parsed schedule. This source does
not qualify Options fees or reopen the consumed fee-hedge/triangle studies.

The [result](result.json) binds capture, source phases and checks. This report
records the local source phase prepared for a coherent batch. Financial counts stay **204 observations / 65
hypotheses / 37 scoped mechanisms / 0 qualified stable edges**. No model fit,
benchmark, credential use, account request, order or persistent task followed.
