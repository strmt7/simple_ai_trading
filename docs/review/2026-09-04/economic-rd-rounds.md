# Economics-first R&D rounds

This is the current review-session development plan, not a new authorization
or a replacement for frozen experiment contracts. Expand a round only when its
result can change a financial or engineering decision. No guarantee of profit
or comprehensive review completion is implied.

## Round 1: mechanisms and comparable cash flows (in progress)

**Binance:** integrate entry/exit basis, correctly weighted funding, collateral
cash flows, borrowing, fees and executable quote conversion into one cash
ledger. Compare a fixed-quantity linear hedge, matched inverse collateral,
and separately qualified portfolio offsets on equal capital-at-risk and
liquidity constraints. The new inverse evaluator is conditional algebra;
official mechanics are unresolved. Retained funding sensitivity did not rescue
the prior failed samples. Do not relabel those samples independent evidence.

**Polymarket:** prove settlement identities before pricing. For existing
inventory, compare completion with immediate executable liquidation; for new
inventory, include acquiring the first leg and failed completion. Separate
completed-package payout, queue/fill probability, adverse selection, orphan
loss and incremental reward. Historical positive pair P&L alone does not
establish superiority to liquidation or justify opening inventory.

Deliverable: one unit-consistent cash-flow and exposure specification per
mechanism, with source facts, conditional identities, unknowns and decisive
counterexamples separated. Do not merge the two venues into a common payoff
model merely because both have multiple legs. Reuse existing math/transport
only where contracts match; leave historical implementations untouched.

## Round 2: evidence gaps and collection value

Inventory retained data by exact contract, event/settlement interval, timestamp,
source hash, costs, depth and role **without consuming protected outcomes**.
Choose each request by the uncertainty it resolves and whether either answer
changes the next action. Source semantics precede market values. A failed
source gate stops that source route, not unrelated offline engineering.

| Lane | Evidence that matters next | Do not substitute |
| --- | --- | --- |
| Inverse collateral | Independent current payoff/margin/fee/precision contract, then a bounded cost/basis feasibility specification | Search snippets, generic leverage settings, synthetic constant-dollar plots |
| Funding carry | Settlement notionals and quote conversion, matched costs and both-leg capital, causal funding persistence | Unweighted rates, headline APR or a directional candle predictor |
| Polymarket completion | Original-token executable liquidation comparator, completion/orphan outcomes, exact settlement semantics | Public trade prints as our fills, winner-only sampling, gross completed P&L |

Existing scheduled routes remain exact: CXMT no earlier than
`2026-09-06T08:10:00Z` under its frozen 6/3/3 roles; spot block-trade no earlier
than `2026-09-06T03:47:16.3134381Z`; MRNA September 8 and USD1/RLUSD September 11
retain every additional canonical condition. Read the **complete current
registry row** before any request. Time alone does not satisfy compound gates.
Holding-yield/protected partial captures are not reusable training data.

## Round 3: prospective economic baseline

Freeze population, units, causal features, chronological train/validation/test,
overlap purge/embargo appropriate to the label horizon, orientation, cost
scenarios, capital denominator, abstention and rejection rules **before**
outcome access. Journal every attempted variant including failures. Start with
the simplest mechanically defensible baseline. Check profit per unit capital,
capacity and time, stressed loss, settlement/custody concentration and turnover.
Report unsupported slices as unsupported; one positive average is not stability.

Use uncertainty intervals appropriate to dependence (for example event/block
resampling with its block construction frozen), not independent-row standard
errors for overlapping samples. This plan does not claim such estimation is
already implemented or that an arbitrary number of rows guarantees power.

## Round 4: models only where they improve a measurable decision

Model completion probability, conditional adverse selection, net carry
persistence or abstention value only after labels represent actual cash-flow
decisions. A forward candle-direction label is not a matched-carry objective.
Compare a modest model against the deterministic/no-model baseline using paired
untouched economic outcomes; improve after-cost risk/reward, not accuracy alone.
Keep optimization, threshold selection and calibration inside training/validation.
Expose calibration, missing-data behavior and selection history. No AI control
may bypass deterministic execution, risk, ownership or close controls.

Increase datasets when they add independent periods/events, cost observations,
rare operational failures and representative regimes. More overlapping bars,
duplicated history, synthetic rows or more symbols without matching economics
do not establish additional evidence. Log provenance, deduplication, gaps,
coverage and memory estimates before large ingestion. No wholesale download
or large training job is justified until the target and coverage gaps are fixed.

Use the verified isolated OpenCL launcher for compatible substantial fits;
small exact accounting is cheaper on CPU. Preserve frozen backend/precision.
Begin with one bounded training job, reuse materialized features across paired
candidates and bound trial counts prospectively. Current concurrency speed
evidence is unqualified; do not chase GPU utilization with redundant work.
Ordinary R&D continues alongside other tasks. Only comparative timing needs
passive before/during CPU/GPU/RAM/disk contention evidence; never modify other
processes. GPU capability is established, economic model uplift is not.

## Round 5: enterprise-grade integration after evidence

Promote only affirmative, source-bound, after-cost evidence for every required
scope. Keep research, paper execution and live-account eligibility separate.
Add versioned economic contracts and replayable state transitions to the
existing architecture rather than parallel CLI/native behavior. Review failure
recovery, stale inputs, unknown order state, sizing, precision, close capacity
and artifact tampering at their boundaries. Use focused affected-domain tests
once per coherent behavior change, not repeated broad CI as research progress.

Continue semantic review in `REVIEW_2026_09_04.md` order; the inventory is not
a completed review. Record files/functions actually read and findings fixed,
rejected or unresolved. Keep old result bytes and exact source bindings. Push
identity-audited direct-main checkpoints. Recheck dependency alerts/PRs when
new state warrants it, not on every economic calculation.

## Stop, expand or move on

Stop an exact experiment on its frozen failure. Expand research when a material
unresolved mechanism or data gap can affect net economics. Move to another
eligible lane when its information value is higher. Do not rewrite failed
gates to create winners, confuse lack of a worst-case guarantee with negative
expected value, or continue a broad research round merely to fill its outline.
