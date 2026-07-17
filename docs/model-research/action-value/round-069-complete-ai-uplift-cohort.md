# Round 69: Complete AI-uplift cohort

**Status:** implementation and contract validation only. No local-model inference or market experiment was run, so this round makes no AI-uplift, profitability, ROI, drawdown, testnet, or live-trading claim. Performance tables and graphs are unchanged.

## Problem

The live AI-uplift materializer already required causal, pre-entry, hash-chained reviews and measured paired after-cost daily returns. Its `causal_coverage` denominator, however, contained only closed shadow trades. A provider failure, downstream risk block, or still-open position could leave an audited proposal without a baseline outcome and silently remove that proposal from the uplift cohort. Positive results on the remaining trades would then be selection-biased.

## Change

`live-ai-shadow-uplift-v2` preserves the existing trade-level causal coverage and adds a separate proposal-outcome cohort:

- every case in the supplied, semantically validated audit chain is counted;
- audit case IDs are matched against bot-owned shadow trade outcomes;
- acceptance requires 100% matched proposal outcomes, independently of the existing 90% trade-level causal-evidence floor;
- unmatched counts and a deterministic SHA-256 of the sorted unmatched case IDs are persisted; and
- the shared CLI/Windows command reports both `eligible/candidate` trades and `proposal_outcomes/audited_proposals`.

Any unmatched audited proposal adds `ai_shadow_proposal_outcomes_incomplete`, keeps the report advisory-only, and cannot be offset by positive P&L or statistical significance elsewhere. This does not manufacture counterfactual outcomes or impute missing returns.

The gate covers the complete supplied durable audit chain. It cannot reconstruct a proposal that failed before durable audit append; reviewer fatal state, queue rejection, coordinator health, and audit integrity remain independent blockers. A report is still point-in-time evidence and cannot authorize orders or claim profitability.

## Verification

- All 53 live AI-assist, paired-uplift, live-uplift, and shared CLI/Windows rendering tests pass together.
- A two-proposal audit with only one baseline outcome reports 50% proposal coverage and is rejected.
- A complete 41-proposal synthetic unit cohort retains 100% proposal coverage and exercises the pre-existing statistical gate; this is contract testing, not market evidence.
- Stale approval fixtures were upgraded to the current proposal-level after-cost payoff contract instead of weakening runtime validation.
- Ruff and `git diff --check` pass on the changed implementation.

No graph was regenerated because no economic or AI-uplift experiment occurred.
