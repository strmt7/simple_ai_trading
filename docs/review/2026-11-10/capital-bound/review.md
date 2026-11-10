# Capital feasibility before another funding model

The retained 2025 BTC/ETH/SOL funding study does not become a candidate merely
by replacing its fully funded capital assumption with leverage. Under its
unchanged reference valuation, 80-bp role reserve and 10% simple annual capital
charge, every asset still fails at least one chronological role even if total
strategy capital were only the optimistic reference amount for one leg at the
existing 20x policy ceiling. No market request, mark-path processing or fit was
needed for this necessary rejection.

This is an exploratory, post-hoc feasibility diagnostic of known rejected
results. It is not independent validation, native margin qualification, actual
financing economics or a family-wide impossibility proof. The original study's
2N outcomes, costs, roles, training-only orientation and all historical bytes
remain unchanged. Financial counts remain **205 / 65 / 37 / zero qualified
stable edges**; no new economic observation is registered.

## Exact necessary bound

Let N be the original role's reference mark, G its original par-valued funding
cash, B = G - 0.008N, and k its original 10% annual charge multiplied by elapsed
time on a 365-day basis. Strictly positive net requires total capital C < B/k.
The comparison uses C at least N/20 for just one reference-valued leg; it leaves
all other collateral, maintenance, buffers and costs out of the optimistic bound.
It does not assert that this amount is sufficient or even available at a venue.
The reference mark is not an observed executable entry price, and USDC/USDT par
valuation remains an explicitly unqualified scenario.

| Asset | Role | Strict capital ceiling, % of reference N | One-leg reference floor | Necessary condition |
| --- | --- | ---: | ---: | --- |
| BTC | Training | 32.158329% | 5% | Passes only this necessary check |
| BTC | Validation | 1.063001% | 5% | Rejects |
| BTC | Test | 1.157324% | 5% | Rejects |
| ETH | Training | 13.422785% | 5% | Passes only this necessary check |
| ETH | Validation | 41.236367% | 5% | Passes only this necessary check |
| ETH | Test | No positive budget | 5% | Rejects before capital charge |
| SOL | Training | 19.139313% | 5% | Passes only this necessary check |
| SOL | Validation | 30.595519% | 5% | Passes only this necessary check |
| SOL | Test | No positive budget | 5% | Rejects before capital charge |

Displayed percentages round to six decimal places. All decisions use exact
rationals, including strict equality rejection. The [result](result.json)
retains quote amounts and rational ceilings. Four identities per row were
[independently reconciled](independent-reconciliation.json), without importing
the calculation helper. The [contract](contract.json) fixed the law and terminal
consequence before execution, while explicitly disclosing the post-hoc question
selection. After all candidates failed, zero raw mark paths were opened.

## Reusable model boundary

`capital_budget.evaluate_capital_budget` supplies the exact strict capital
interval under caller-supplied comparable cash, costs, minimum capital, elapsed
time and annualization convention. It refuses floats, ambiguous labels, invalid
durations and negative capital/cost inputs. A positive interval is necessary
only; native-capital and qualified-edge fields remain false. Eighteen focused
algebraic controls pass, including equality finer than floating-point precision.
This helper is not an integrated trainer, an exchange margin engine or an order
admission authority. No historical result or existing model was rewritten.

The decision is to stop capital/leverage parameter search on this fixed panel.
Do not lower its hurdles, discard the failed forward roles or treat a different
capital convention as fresh evidence. A future informative population still
needs the literal registry trigger and frozen source/cost/role law. Continuous
inventory objectives must distinguish initial entry costs, later transition
costs and actual available collateral; a role-reset diagnostic cannot prove
the return of a continuously held strategy. Qualified native fees, basis,
wallet cash, transfer clocks and collateral remain prerequisites for fits.

## Source availability and operational evidence

One separately frozen official account-document attempt passed robots checks
but returned native HTTP 302, then redirected HTTP 202 and one extracted
character. All required scope terms were absent. The [source receipt](account-source.json)
retains URLs, native success, admission failure, provider identity, hashes and
provenance coverage. No wallet/withdrawal semantics were admitted, no alias was
retried and no signed endpoint was queried. The redirect is documentation
discovery, not a funding, fee or execution-architecture retry trigger.

The calculation is independent of that failed page. Its source/implementation
bindings and bounded execution are retained in [evidence](evidence.json).
[Ledger reconciliation](ledger-binding.json) adds routing metadata only;
accepted classifications, financial frontier, decisions and all 205 terminal
entries remain intact. No credentials, account, order, protected partial data,
training, GPU benchmark, scanner retry, release or persistent worker was used.
