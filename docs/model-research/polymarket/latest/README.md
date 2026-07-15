# Polymarket research status

![Noncausal executable repricing ceiling](charts/repricing-ceiling.svg)

Round 8 remains the latest completed numeric evidence. Its gap-free
`2026-07-15T00:46:38.779Z` to `2026-07-15T00:55:51.787Z` capture covers 12
BTC/ETH/SOL five-minute markets and 612,522 reconstructed books. It found
171,400 complete two-taker oracle paths; 360 of 480 market/outcome/grid rows had
a positive best future-timed path after displayed depth and both fee legs.

This is a **noncausal mechanism ceiling**, not ROI or a trading strategy. The
primary gate has only three complete markets per asset versus 30 required.

Round 9 is implemented but unfitted. Its frozen
[action](../round-009-causal-action-value-contract.json),
[ridge](../round-009-ridge-implementation-contract.json), and
[MLP](../round-009-causal-mlp-challenger-contract.json) hashes are
`c8988fd548cff295800b977d6e6c92c39e9f2867b6c6e4b5f7e3d0b2b96f9800`,
`4b192e7f30af3e3d6e7dfb1b2b3342518e23de6d750b6b1cfd2334d87f2f5a12`,
and `a5d87f65036e4a6c71835ce549668d81767b2ba16bd227ea2319c24b0880f7a2`.
A post-contract capture must pass integrity, continuity, BTC/ETH/SOL
synchronized-group breadth, and immutable official-resolution checks before
the ridge may be fitted once. The MLP may run only if the ridge passes its
preregistered development gate. Its report-v2 implementation also requires
strictly positive validation stress-utility uplift over ridge and keeps the
untouched neural test closed unless it contains at least 30 synchronized time
groups. No Round 9 model score, AI edge, profitability, drawdown claim, paper
authority, or trading authority exists.

The former v6 local-AI selection is revoked because its prompt leaked expected
actions through case IDs. Fresh label-free v7 inference rejected all four
priority 8B/9B models; `ai-risk-models-rejected.json` is negative governance
evidence only and grants no AI or trading authority.

Capture attempt `eae374e2662c440fb93970d5710937b1` is development-only. Its
queue reached `100000/100000` and the interrupted run has no terminal report,
so it cannot confirm a model. A real-message writer replay selected a bounded
0.5-second coalescing window and 200,000-message queue for the next attempt; the
signed [throughput benchmark](../round-009-recorder-throughput-benchmark.json)
contains the exact source window and measurements.

The `polymarket-ridge` command claims its pipeline in DuckDB before test
evaluation. A completed claim reloads the signed report without refitting; an
interrupted or failed claim remains fail-closed so a retry cannot silently
reopen the untouched test. `polymarket-mlp` uses the same database-backed claim
before any nonlinear test access.

Inspect the [full signed report](../round-008-executable-repricing-ceiling-report.json),
[exact chart data](tables/repricing-cells.csv),
[primary market rows](tables/repricing-primary-markets.csv), and
[integrity manifest](publication-integrity.json).
