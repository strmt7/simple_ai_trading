# Native Spot entry-cost-aware risk marking

An unchanged market price is not break-even after entry fees. A 100 USDC BUY
with 0.001 BTC commission leaves 0.999 BTC, worth 99.9 USDC at the original
100 mark. The previous automatic exit and loss-budget consumers missed that
0.10 USDC loss. Both financial reproductions failed before this repair.

The [source-bound record](spot-cash-mark.json) specifies rational remaining-cost
allocation. Native cash return now feeds automatic/lifecycle exit thresholds,
daily/session loss limits and entry drawdown. Each loss-budget calculation uses
one coherent paired snapshot. When a native open lot exists, its mark must bind
the exact instrument/product and all ledger cash must share its supported quote.
Missing context, other open instruments and unvalued nonzero third-asset fees
reject admission rather than inventing a portfolio value. Unqualified guard
metrics are placeholders, not evidence of zero loss.

Explicit owned reduction does not require guessed fee FX. Existing durable
close/UNKNOWN behavior is preserved; this is not autonomous recovery or an
independent supervisor. An unqualified valuation stops the current admission
loop, not proof that a separate process continues monitoring inventory.

575 affected-domain checks pass, including 24 new cases. They cover actual
base/quote fees, partial cost conservation, restart, invalid marks, instrument/
quote refusal, take-profit and drawdown consumers, legacy behavior and owned
closing safeguards. Ruff passes. Two local test setup mistakes were corrected;
no product bypass was introduced to make those tests pass. No full-suite,
hosted-CI, benchmark or live-account validation is claimed.

The mark excludes unobserved exit fees, slippage and taxes: it is not executable
liquidation profit. Default UI/statistics still report price-only unrealized
PnL; native entry-cost marking is explicit opt-in. General portfolio mark/FX
valuation, no-open mixed-quote accounting, independent supervision and account
qualification remain open. Legacy/modelled data are not retroactively native
cash evidence. No market/account/order access, historical result rewrite,
protected-data change, campaign logic or profitable-edge claim was made.
