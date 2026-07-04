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

## Current Status

No repo-facing ROI, P&L, win-rate, or drawdown claim is made here. Regenerate
validation from real source data and attach full provenance before publishing
model performance.
