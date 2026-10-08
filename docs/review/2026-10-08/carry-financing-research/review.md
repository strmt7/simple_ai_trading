# Carry financing: a useful rejection bound before more training

This bounded round is complete. The user requested a commit and pause. No
profitable strategy, new independent economic observation or training admission
is claimed. Existing financial counts remain 202 observations, 65 hypotheses,
37 scoped mechanisms and zero qualified stable edges.

## Primary-source result

The mandatory isolated Crawl4AI provider extracted the
[BIS primary HTML summary of Crypto carry](https://www.bis.org/publications/working-paper-1087-crypto-carry),
Working Paper 1087, published April 4, 2023. Native HTTP status was 200,
success was true and the substantive document matched its title/authors/date.
The full 514,835-byte native result remains outside Git; [intent](crawl-intent.json)
and [digest-bound result metadata](crawl-result.json) are retained here. This
reads the HTML summary/abstract, not the 54-page PDF.

The authors link carry to leveraged investor demand and scarce arbitrage
capital, and identify margin spikes and forced liquidation as risks. This
supports investigating a structural mechanism, not importing their historical
returns into Binance, SOL or Polymarket. No contemporary commissions, financing,
collateral recognition or account eligibility were established. The source audit
keeps that distinction explicit; the research does not reopen a market study.

## New financial inference from existing evidence

The [derivation](derive_capital_constraint.py) reads only the unchanged,
hash-bound [October 7 retained frontier](../../2026-10-07/native-funding-hedge-frontier/result.json).
It does not rerun that consumed runner, read its raw markets, reprice any leg,
select an orientation or claim another independent validation.

For the original hypothetical HELD cash, 32-day duration, 32-bp noncapital
reserve, zero basis deterioration and 325-bp annual capital cost, invert:

`net_bps = funding_bps - 32 - 325 * capital_multiple * 32 / 365`.

This is a post-hoc algebraic design constraint using the original declared
sensitivity. The rate and reserve are not native venue costs. All three assets
and the original full-period panels remain included.

| Asset | Strict-positive net requires capital/reference below | Net at one reference unit and zero perpetual margin, bps |
| --- | ---: | ---: |
| BTC | 0.387071 | -17.464283 |
| ETH | 0.406543 | -16.909455 |
| SOL | 0.081309 | -26.176392 |

A spot purchase fully funded at the same reference scale already consumes one
unit in this comparison. Even reducing perpetual margin to zero cannot overcome
the capital deficit under this scenario. Increasing leverage is therefore not
a sufficient remedy. The illustrative one-twentieth margin case is also retained
in the [exact rational artifact](capital-constraint-result.json); it is not
exchange margin recognition, a leverage recommendation or a solvency claim.

This is not proof that every implementation has negative expected value.
Different financing, a properly valued collateral-yield offset, cheaper all-in
execution or favorable basis outcomes could change the conclusion, but require
separate evidence. Existing inventory does not automatically make capital free;
its counterfactual, opportunity cost and risk must be explicit. Dated futures
basis and perpetual funding also have different convergence/settlement laws.

## Decision and verification

Prioritize the financing/collateral and full paired cash comparison before more
funding-only model fits or GPU expansion. Compare equal initial wealth against
the feasible alternative over the same horizon. Record borrow costs, reward
compatibility, haircuts, fee assets, partial-fill residuals, basis exits and the
continuous margin/liquidation path. Do not add collateral yield without proving
eligibility and avoiding double counting its opportunity comparator.

The derivation checks original byte identity, complete three-asset scope,
duration, capital multiple and the original cost law, then reconciles each
zero-net boundary exactly as rational arithmetic. Output creation refuses an
existing result. Ruff passed; source/result reproducibility, tamper rejection,
native crawl digests and no-overwrite behavior were checked separately. No
unchanged trading suite, full CI suite, performance benchmark or fit was rerun.

Two local mistakes were corrected transparently: the initial search-intent
timestamp was replaced with its observed pre-search file creation time, and an
off-by-one review-script root failed before any result or financial access.
Neither changed a consumed capture, access bound, historical source or outcome.
This round changed review artifacts and handoff only, not the trading runtime.

Resume from this constraint and the qualified September funding subset. Do not
repeat the same funding sum or capture solely because time passed. No accounts,
credentials, orders, funds, automation or other repositories' providers changed.

Publication audit again found the already-disclosed historical AI placeholder
emails and Codex identities. They remain policy violations; this checkpoint does
not fix or force-rewrite shared history. Fresh remote PR-head refs were counted
separately (21 refs), not modified. The new commit must use `AI agent <>` for both
author and committer. The research capture-boundary byte hash remains unchanged.
