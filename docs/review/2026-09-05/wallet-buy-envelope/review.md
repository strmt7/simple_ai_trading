# Full buy-population exposure, not selected-lock profit

The retained page has 1,964 rows. Exactly 1,222 belong to the frozen BTC/ETH/SOL
5m/15m/4h scope across 358 conditions; 742 are explicitly out of scope. Every
scoped buy was included, irrespective of profitability or membership in the
earlier 66 locks. No lock was reselected, resized or rematched.

The scoped purchases cost **21,656.6576921811824739 quote units before fees**.
Their algebraic binary decomposition contains 8,748.967455 paired shares and
47,891.745927 residual outcome-dependent shares. 357 of 358 conditions have
an imbalance. Under the explicitly conditional complementary one-unit payout
model, the aggregate gross outer PnL envelope is
**-12,907.6902371811824739 to +34,984.0556898188175261** quote units. These
endpoints need not be jointly attainable when conditions depend on each other.
Fourteen conditions have a strictly positive gross lower endpoint; 31 have a
negative gross upper endpoint. None of those counts is a promoted strategy.

Residual claims must collectively pay more than 12,907.6902371811824739 quote
units to make this buy-only replay gross-positive. Dividing by residual share
quantity gives about 26.9518%. That is a quantity-weighted payout requirement,
**not a probability, forecast, ordinary win rate or calibrated training label**.
Fees and all other external costs would raise the break-even requirement.

## What this establishes and what it does not

This audit quantifies why the earlier positive selected-lock totals do not
prove that copying the full purchase activity is direction-independent. It
does not disprove the existing-inventory completion overlay, alter its earlier
positive historical totals or repair its failed robustness gate. It does not
justify selecting the 14 positive-floor conditions after observation.

The retained Data API taker-trade page does not establish actual starting or
ending inventory, maker fills, transfers, splits, merges, redemptions, terminal
outcomes or exact fees. Consequently this is **not actual wallet PnL** or an
owned fill ledger. The one-unit complementary payoff is an explicit model
premise, not new verification of each market's resolution exceptions. No
unstated zero-cost, zero-inventory or independent-outcome claim is admitted.

The useful next evidence is a complete causal entry/completion/residual cash
ledger plus the feasible liquidation comparator already identified in the
[inventory review](../../2026-09-04/inventory-completion-economics.md), not more
stress grids or training on selected successful locks. This audit supplies no
new account authority or market retry trigger. Stop analysis of this exact
sample unless a distinct material accounting defect or missing admissible
source can change the decision. No new venue, fee, resolution or account
request follows from this result.

## Verification and efficiency

The [contract](contract.json) binds the original raw page, original validation
contract/result, and new offline implementation. The [result](result.json)
retains every condition and all exclusions/counts. The original raw response
and historical results remain byte-identical; no protected captures were read.
The registry remains unchanged: 37 mechanism scopes, 65 hypotheses, 192
terminal observations and zero qualified stable profitable edges.

Independent rational-arithmetic reconstruction checks each condition and every
aggregate against the raw buys; synthetic tests check both outcomes, balanced
and residual inventory, identity ambiguity, invalid input and one-use output.
All 28 affected checks pass, and Ruff passes on the three touched Python files.
Pre-publication test defects were corrected, not hidden: a fixture argument
collided with an outcome field, and the initial reconstruction assertion
incorrectly expected every raw row to be scoped. Production already retained
the 742 exclusions and its consumed result was not changed. Future fixtures
should apply field mutations after construction and reconciliation should
check inclusion plus exclusion counts, not assume a raw page is homogeneous.

A previous family test also pinned the mutable global accepted count; it now
checks registry/audit agreement instead, preserving the existing instruction
against unrelated test churn. This change touches no economic result or gate.
No GPU workload, broad CI, new literature search or new capture was needed.
