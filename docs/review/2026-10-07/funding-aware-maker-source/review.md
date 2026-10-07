# Funding-aware maker methodology and cash-label finding

[Source-bound structured review](review.json) records captures, source hashes,
the cash counterexample, scoped repairs and verification limits.

No qualified stable edge or model uplift. A distinct primary preprint was
retained with two frozen public document GETs: Nam Anh Le, *Funding-Aware Optimal
Market Making for Perpetual DEXs*, arXiv 2605.06405v1, May 7, 2026.
[Primary paper](https://arxiv.org/html/2605.06405v1).
The successful abstract exposed the exact HTML link; no guessed alias or PDF
retry. Contracts, raw local bodies, request journals and source receipts remain.
Scholarly bodies stay local; the abstract declares CC BY 4.0, but redistribution
of the complete arXiv HTML wrapper is not qualified by that article declaration.
The public checkpoint contains attribution, hashes and receipts, not those
bodies. This is not a self-contained paper reproduction.

Sections 1, 4–8 and Appendix A were read completely, including mathematical
units, control assumptions, calibration, tables and limitations. Sections 2–3,
all cited literature, proofs, author code/data and independent empirical
replication were not completed. No author experiment/data repository link was
found in the retained HTML; arXiv renderer/feedback links are not one.

## What is useful and what is not established

The source couples dealer inventory to cash-scaled funding and solves a
finite-difference HJB. A funding state can improve inventory control without a
price-direction forecast; that does not make inventory market-neutral.
Its final table disables optional hedging, samples independent bid/ask proxy
fills and marks residual inventory. Missing queue, latency, cancellation and
fill-conditioned adverse selection remain execution limitations. No documented
complete after-all-cost native cash reconciliation supports importing returns.

The 100 seeds share the November 26–December 31, 2025 market path; their
confidence intervals concern simulation randomness, not independent market
paths. ETH/BTC mean gains and lower inventory RMS survive two fill proxies, but
the selected stress diagnostics contain four losing asset/window comparisons.
SOL's gain comes with much higher inventory and is dominated by the risk-scaled
AS diagnostic. That comparator changes quantity/capacity while retaining the
effective fill curve. These are reported, not independently replicated results.

OU-plus-jump diagnostics favor heavier-tail funding, but jumps do not enter the
final controller. The source cash state uses price scaling; a fractional-rate
OU is not automatically an independent cash OU when price moves. The latter is
a mathematical inference, not a claimed error in independently inspected author
code. Chronological availability of funding signals and identical event-keyed
simulation innovations remain to be verified before adaptation.

The [neutral-control design](../../../model-research/FUNDING_AWARE_NEUTRAL_CONTROL.md)
records the cash/clock, hedge, risk-matched comparator and efficient evaluation
requirements. No paper constants, HJB, fit or execution policy were imported.

## Concrete repository finding and scoped repairs

Direct execution of both current funding-window helpers confirms that they
return 50 bips for a 0.005 settled rate, omitting a 102 settlement mark relative
to 100 entry price. A fixed-base position pays 51 bips. At a 50.5-bip price gain,
the pre-execution-fee control flips from +0.5 to -0.5; the short control flips
the other way. This is an algebraic counterexample, not a market observation or
after-all-cost profitable trade. Both label builders normalize their price
return at entry while their funding sum lacks that cash scaling. Warehouse
funding rows and `FundingState` omit the necessary mark.

The cash calculation remains unrepaired; do not expand affected training before
cash-qualified new targets. Historical results are preserved as rate-only
proxies, not silently changed. Repair must include mark provenance, both label
consumers and intrabar settlement entitlement rather than substitute candles.

The AI veto prompt's claim of exact taker charges and historical funding cash
was corrected to fixed execution allowances and a funding-rate proxy, with an
instruction to veto apparent profitability dependent on unqualified cash.
This is prompt truth/risk guidance, not an enforced new cash gate or measured AI
uplift. One regression failed before; both existing AI-veto tests pass after.

A type-only temporal-dataset import was moved behind `TYPE_CHECKING`, removing
unnecessary Torch coupling from pure label math. The funding helper source is
unchanged; its original complete module remains in prior Git history. Three
barrier tests plus two AI-veto tests pass in the locked lightweight environment.
No training, AI request, GPU benchmark, dependency change or account access.
The first reproduction import failed because locked default Python has no
Torch; the existing `.venv311` reproduced both helpers without a shim or install.
After decoupling, the same diagnostic is available in the lightweight runtime.
No speedup or GPU qualification is claimed.

Counts remain 201 observations, 65 hypotheses, 37 mechanism scopes and zero
qualified stable profitable edges. Source boundaries and acceptance/retry
contracts are unchanged; the next actionable work is the cash-label repair.
