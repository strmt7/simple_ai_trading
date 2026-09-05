# Inverse carry: sign corroborated, economics still conditional

One distinct [official Binance guide](https://www.binance.com/en-NG/support/faq/detail/360039304272)
corroborates reciprocal entry-minus-mark PnL with positive long and negative
short direction. Its displayed update is April 13, 2026. It expressly defers
to legal rules and contract specifications. It is not a correction to the
conflicting clearing PDF.

The existing conditional short model therefore needs no sign change:
`PnL_coin = N*(1/M - 1/F0)`. This agrees with clearing section 80, not 86.2.
The new evidence narrows the research question; do not repeat generic sign
searches or count the educational guide as a binding legal amendment.

[Structured review and exact source bindings](inverse-sign-review.json) retain
the distinction. One search and one deduplicated article open were used; no
market, account, order or protected-capture access occurred. The source is
retained as tool-extracted text, not original HTTP bytes. No historical artifact
was rewritten, and no hypothesis or terminal-market count changes.

## Financial consequence and next decision

Under that conditional equation, matched coin collateral `q=N/F0` produces
terminal coin equity `N/Fd` before intervening cash flows. Its executable quote
value is `N*Sexit/Fd`, not automatically `N`. If entry acquisition costs
`N*S0/F0` and all additional quote-valued costs total `K`, then terminal net
cash is `N*(Sexit/Fd-S0/F0)-K`. Thus the exact conditional break-even ratio is
`Sexit/Fd > S0/F0 + K/N`. This is algebra, not a measured bound on that ratio.

That isolates the financially useful next work: bind the acquisition/short
fills, delivery average versus executable spot exit, coin fee debits and
capital costs. Margin resilience must hold along the path, not just at
settlement. A large apparent carry spread cannot remove custody, liquidation,
delay, quote-conversion or contract-applicability risks. No new quote sampling
is justified by this source-only review; the existing account-fee trigger and
consumed funding/carry contracts remain unchanged.

Use retained scenarios for conditional inverse-versus-linear analysis rather
than another PnL tutorial or new training job. Current effective contract
applicability and precise mechanics remain the venue-qualification question.
The full platform and stable profitable-edge objective remain incomplete.
