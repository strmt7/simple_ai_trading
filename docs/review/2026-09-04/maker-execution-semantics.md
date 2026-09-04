# Maker execution: inventory semantics before another model

Review base `34033ab77e14626385954ed15f9fe7a04f847ee8`. This is an offline
implementation audit with synthetic counterexamples, not a new market sample.
The [reproducible result](maker-execution-review.json) binds nine source files;
self-hash `7315e011b3f7b9b184252d38a378961b5bf69e88352597532af3f85cc3fee5da`.
Historical implementations, evidence, rejection decisions and retry gates are
unchanged. No protected data, historical market outcomes, accounts or orders
were accessed. The numerical review made zero network requests.

## R16: both filled does not necessarily mean flat

`probe_round58_two_sided_maker_support._probe_symbol` gives the bid and ask
the same `order_notional_quote_per_side`. `build_passive_fill_result` converts
that to base quantity by dividing by each side's price. Therefore, for positive
bid `b`, ask `a > b` and quote notional `N`, a pair of full fills leaves
`N/b - N/a > 0` base units. Its inventory is not flat merely because both
Boolean labels are true. Round 58's full-fill support counts are not being
recomputed or declared numerically wrong; their economic interpretation needs
this additional sizing check.

An intentionally exaggerated **synthetic**, fee-free example makes the units
clear: buying 1,000 quote at 100 obtains 10 base; selling 1,000 at 125 disposes
of 8. Net transaction cash is zero with 2 base left. Selling the same 8 base
that were bought costs 800 and receives 1,000: 200 cash and zero residual.
These prices are not historical Binance observations or spread estimates.

For spot, `cash + residual_base * executable_exit_price` describes liquidation
value before further costs. For a linear derivative the analogous expression
is marked PnL, **not the exchange's margin cash-transfer ledger**. Residual
exposure, contract multipliers, funding, commissions, precision, partial sizes,
margin and liquidation remain separate requirements. In particular a residual
long can lose value after both full fills; the spread is not wholly locked.

## R17: incomplete full-fill evidence is not zero inventory

The frozen queue kernel returns completion metadata only after queue plus
the entire hypothetical order is printed through. With 5 units ahead and a
10-unit order, 4 matching printed units and 14 matching printed units produce
the same `filled=False`, zero count and zero completed-print summary. Under
that kernel's own simplified FIFO assumption, the second case has 9 units of
partial exposure. At 15 printed units it reports full completion.

This is correct censoring for a **full-fill support** statistic, but insufficient
for a lifecycle inventory/PnL label. It is not evidence of an actual owned
fill. The caller cannot recover partial size from the censored summary alone.
Accordingly, Round 58's `none` means neither side fully completed, not that
neither side traded. Likewise its singleton time and completion-time difference
do not measure exposure beginning at the first partial fill.

The separate `paper_execution.apply_passive_trade_print` helper does track
partial quantity, but is not wired into Round 58's support probe. Its local
inputs also lack unique trade identity and replay ordering state, and a fully
filled returned state fails its next `validated()` call because remaining
quantity is zero. No runtime use outside its definition and test was found by
the scoped call-site search. Do not repurpose it as a production fill ledger.
These are documented forward-use blockers, not fixes secretly applied to
hash-bound historical code.

## Polymarket: what the retained example can establish

The retained crypto-maker rebate example is already explicitly conditional on
both fills. It is not evidence of a current quote or actual rebate payment.
Its 50-share pair earns gross 1.00 if completed; the maximum one-sided
settlement loss is 24.50. In the deliberately simplified binary stress model,
conditional on at least one full fill, with every orphan losing that maximum,
`EV = p * 1 - (1-p) * 24.5`. It becomes positive only above `p = 49/51`, about
96.08%, before additional costs or rewards.

This is a stress break-even requirement, **not an estimated completion rate or
an expected-value rejection**. Actual conditional mean orphan loss is unknown
and need not equal its maximum. Partial fills require a quantity-aware model,
not forcing every event into these two outcomes. No accepted scope, terminal
observation count or rank-17 trigger changes from this arithmetic.

## The next useful implementation, only with eligible evidence

Reuse Round 57/58's existing research, not another literature or model round
claiming that joint fills and toxicity are a new idea. Before a new economic
replay, specify a versioned quantity-aware event ledger:

1. Freeze decision-time prices, net base sizes, multipliers, source receipt
   clocks, posting/cancel latency, TTL and a feasible terminal inventory rule.
2. Retain unique trade/update identities, gaps, partial increments, cumulative
   sizes, late cancel-race fills and reconciliation. Unknown stays unknown.
   Counterfactual queue assumptions remain explicit; tape prints are not owned
   execution, and hypothetical decisions cannot share consumed liquidity in a
   portfolio replay without a participation/accounting rule.
3. Reconcile cash and residual inventory after every event. Mark/close residual
   inventory with contemporaneous executable depth and costs at a common horizon.
   Do not retrospectively resize the second leg after seeing its outcome.
4. Score the joint quantity, completion and conditional orphan-cost distribution.
   Preserve no-fill, partial, failed, unknown and losing cases. Compare maker,
   feasible taker completion, liquidation and abstention on the same population.
5. Only then test a small calibrated model against a fixed economic rule on
   untouched blocks. Do not learn independent side-fill probabilities and
   multiply them without evidence; do not expand data or launch GPU training
   to repair an economically ambiguous label.

This is a specification, not a collector authorization. Polymarket rank 17
still needs its exact source-change or separately authorized owned-execution
trigger. Other frozen families retain their own boundaries. No exchange or
historical database capture was launched by this review.

## Literature receipt and efficiency correction

Three arXiv abstract pages were browsed September 4, within the independently
checked 22:46:21-22:52:48 UTC review interval. Exact request timestamps and origin
HTTP bytes were not exposed by the browser tool. The following is a paraphrased
discovery receipt, **not byte-retained methodology evidence**:

- Lalor and Swishchuk, [Market Simulation under Adverse Selection,
  v3, June 2, 2026](https://arxiv.org/abs/2409.12721v3): their abstract describes
  CME futures simulations where fill and adverse-selection treatment affects
  apparent performance. This does not validate Binance or Polymarket parameters.
- Huang, Lehalle and Rosenbaum, [Queue-reactive model,
  v2, September 3, 2014](https://arxiv.org/abs/1312.0563v2): the abstract describes
  queue-state-dependent order-flow modeling. No model was fitted here.
- Lehalle and Mounjid, [Limit Order Strategic Placement,
  v4, March 15, 2018](https://arxiv.org/abs/1610.00261v4): the abstract connects
  imbalance, adverse selection and latency. No latency advantage was measured.

The required retained-reference lookup occurred **after**, not before, these
opens. It found the queue-reactive reference in Round 57 and adverse-selection
references in Round 74 designs. That ordering was inefficient and did not meet
the repo's source-deduplication workflow. Do not call these new literature
triggers or refetch them to repair the receipt. No source numerical result was
used in the audit. Future routing must begin at the existing mechanism's code
and retained citations before spending a literature request. The standing rule
already exists; adding another duplicate rule or another collector is unnecessary.
