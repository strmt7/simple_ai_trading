# Collateral stress: economic neutrality is not margin neutrality

Implemented a conditional linear-hedge stress evaluator in
`src/simple_ai_trading/collateral_stress.py`. It separates economic equity from
credited collateral and measures the additional fully credited cash needed to
meet supplied margin requirements and a buffer. It is a research component,
not integrated into execution, not Binance's margin engine and not a new edge.

## Source result

One bounded two-query search and one independently deduplicated official
[Portfolio Margin Pro guide](https://www.binance.com/kk-KZ/support/faq/detail/146ac090710a49f99c4732b00f8c09df)
were completed. The guide describes combined Margin/USD-M/COIN-M wallets,
haircut-adjusted collateral, and wallet-specific deposit limits. Its 95% BTC
rate is an illustrative example, not a current collateral-rate table. Its
Cross Margin leverage table does not establish dated-futures margin. The guide
is educational, warns it may be outdated, and defers to legal terms and
contract specifications. Displayed publication/update times lack a timezone
and are not normalized to UTC. A Kazakh locale does not prove user eligibility.

No numerical account haircut, current dated-contract eligibility or complete
initial/maintenance-margin calculation is qualified. The prior consumed account
documentation was not retried. Complete tool extractions and request bounds are
retained; these are not origin HTTP-byte captures. Search-only auto-exchange
and collateral-change snippets are leads, not admitted economic inputs.

## Conditional model and demonstration

For owned asset quantity `q`, short linear base quantity `n`, spot mark `S`,
futures entry `F0`, futures mark `F`, fully credited cash `C`, and cumulative
quote cost debits `K`:

- Economic equity: `C + q*S + n*(F0-F) - K`.
- Credited equity under the stated model: economic equity minus `q*S*(1-h)`,
  where `h` is the supplied asset credit ratio.
- Headroom: credited equity minus the supplied total margin requirement and
  quote buffer. Costs are deducted once, never hidden inside the credit ratio.

The synthetic artifact fixes `q=n=1`, `F0=100`, no cash/cost/buffer, `h=0.95`,
and a hypothetical requirement of 10% of futures notional. This is not a
Binance rate or requirement. Across equal spot/futures marks of 100, 200 and
1,000, economic equity remains 100, but credited equity is 95, 90 and 50;
headroom is 85, 70 and -50. Fifty extra quote units reach equality at the
worst supplied state, not strict safety. These prices have no assigned
probability; the extreme state is not a forecast.

Algebraically, with matched quantities, equal marks, fixed `h` and hypothetical
requirement rate `m`, headroom is `C + q*F0 - K - buffer - q*S*(1-h+m)`.
Thus a flat economic value does not make margin headroom flat. Even without a
haircut, a requirement rising with notional can consume headroom. Basis
divergence, unmatched quantities, changing haircuts and cost debits are retained
explicitly rather than netted away.

## What changes next

Carry capital requirements should be derived from source-qualified joint states
and actual recognition rules before computing capital-adjusted returns. Do not
equate a flat terminal payoff, a portfolio-netting feature or low initial margin
with liquidation resistance. The evaluator rejects missing/invalid numeric
inputs, duplicate scenario labels and empty populations, preserves deficits,
and permanently leaves venue-margin and edge qualification false.

The current component assumes unchanged positions, fully credited nonborrowed
cash and full recognition of short PnL; forced exchange, interim transfers,
liquidation, changing positions and interest must be modeled separately. A
finite supplied set does not prove a full stress envelope or calibrated risk.
Historical capture outcomes, fee gates, protected data and registry counts are
unchanged. No GPU, training, account or order access was needed.
