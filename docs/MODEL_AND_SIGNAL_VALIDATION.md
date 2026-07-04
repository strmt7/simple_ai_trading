# Model and Signal Validation

This page records the validation standard for model and signal work. Historical
local performance numbers were removed because they did not carry the current
data-provenance fields required for repo-facing financial evidence.

## Required Local Validation

A validation report may be documented only when it includes:

- command used,
- source API or signed account source,
- symbol list,
- market type,
- interval,
- exact UTC start and end,
- candle/fill row count,
- coverage ratio and gap count,
- execution assumptions,
- compute backend and device,
- generated artifact path.

## Financial Sanity Standard

Model artifacts must pass the built-in financial sanity gate before they can be
treated as live-ready, AI-reviewable, or suitable for optimization evidence.
The gate enforces:

- finite model parameters and matching feature dimensions,
- bounded learning-rate, regularization, class-weight, threshold, and
  probability-temperature settings,
- positive accepted row counts and finite objective scores,
- complete data-coverage metadata with no failed integrity status,
- accepted stress, temporal robustness, and portfolio-risk reports,
- bounded drawdown, CVaR, deployed weight, correlation, and cluster exposure
  metrics.

These checks are intentionally conservative. They follow the practical model
risk themes in the April 17, 2026 interagency revised model-risk guidance:
sound model development and use, validation and monitoring, conceptual
soundness, outcomes analysis, governance, and controls. They also align with
Basel-style market-risk backtesting discipline: compare predicted risk with
subsequent trading outcomes and treat exceptions as model-quality evidence, not
as cosmetic reporting.

References:

- Federal Reserve SR 26-2 revised model-risk guidance:
  <https://www.federalreserve.gov/supervisionreg/srletters/SR2602.pdf>
- OCC Bulletin 2026-13 model-risk guidance summary:
  <https://www.occ.gov/news-issuances/bulletins/2026/bulletin-2026-13.html>
- Basel Committee market-risk backtesting framework:
  <https://www.bis.org/publ/bcbs22.pdf>
- SEC/FINRA market-access risk-control context:
  <https://www.finra.org/rules-guidance/key-topics/market-access>
  and
  <https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions/divisionsmarketregfaq-0>

## Market Edge Requirement

Profit alone is not accepted as evidence of a useful algorithm. A candidate must
show net market edge: realized P&L must beat the same-notional buy-and-hold
baseline after fees, spread, slippage, latency/liquidity stress, and the current
risk profile's position sizing. Objective acceptance now rejects candidates
whose `edge_vs_buy_hold / starting_cash` is below the profile threshold:

- conservative: `0.20%`,
- regular: `0.30%`,
- aggressive: `0.50%`.

This is intentionally a first-line filter, not the whole proof. Model-lab still
requires temporal robustness, stress validation, portfolio-risk acceptance,
statistical-edge evidence, financial sanity, and full data provenance. The edge
threshold is designed to reject tiny noisy improvements before the more
expensive validation layers run.

The standard follows three research cautions:

- White's Reality Check: selected strategies can look good by chance after data
  reuse, so benchmark outperformance needs explicit testing.
- Bailey and Lopez de Prado's Deflated Sharpe Ratio work: backtest selection,
  non-normal returns, and multiple trials inflate apparent performance.
- Harvey, Liu, and Zhu's multiple-testing research: a newly discovered edge
  needs a higher hurdle than ordinary single-test significance.

The code now emits a `market_edge` evidence report for training, stress, and
temporal validation payloads. Accepted model-lab artifacts are blocked by the
financial-sanity gate if any nested stress scenario or temporal window contains
failed market-edge evidence. The report captures:

- benchmark P&L and strategy P&L,
- net edge as a percentage of starting capital,
- closed-trade and sample counts,
- profit-factor and expectancy evidence when available,
- sign-test p-value over trade/window samples,
- bootstrap lower mean return,
- failed checks as stable machine-readable strings.

This makes "edge over the average market" explicit: a model must show audited
net outperformance over the same-symbol passive market benchmark, then preserve
that edge through adverse execution assumptions and chronological replay before
it can be promoted.

References:

- White, "A Reality Check for Data Snooping":
  <https://www.ssc.wisc.edu/~bhansen/718/White2000.pdf>
- Bailey and Lopez de Prado, "The Deflated Sharpe Ratio":
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551>
- Harvey, Liu, and Zhu, "... and the Cross-Section of Expected Returns":
  <https://www.nber.org/system/files/working_papers/w20592/w20592.pdf>

## Current Status

No repo-facing ROI, P&L, win-rate, or drawdown claim is made here. Regenerate
validation from real source data and attach full provenance before publishing
model performance.
