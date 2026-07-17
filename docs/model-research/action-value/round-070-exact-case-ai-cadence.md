# Round 70: Exact-case AI cadence admission

**Status:** implementation and contract validation only. No model inference or
market experiment was run, so this round makes no AI-uplift, profitability,
ROI, drawdown, testnet, or live-trading claim. Performance tables and graphs
are unchanged.

## Problem

Live AI reviews are bound to the exact candle timestamp, model fingerprint,
proposal, costs, regime, and causal evidence. The asynchronous result can be
used only when a later coordinator iteration reconstructs that same case. A
one-second candle with a one-second poll and ten-second provider deadline can
never do so. Previously that configuration could start, spend local-model
tokens, and repeatedly suppress otherwise eligible entries as pending or
superseded evidence.

## Change

Active AI startup now computes the minimum nominal stable window needed for
both submission and a later retrieval poll:

`(ceil(provider timeout / effective poll) + 1) * effective poll`

The effective autonomous poll retains the coordinator's one-second floor. If
the selected Binance candle is shorter than that window, startup fails before
exchange-client construction and explains the exact interval, poll, timeout,
and remedies. AI-disabled operation is unchanged. Longer intervals retain the
existing exact-case, hash-bound, fail-closed review path.

This gate deliberately does not reuse a review on different market evidence,
weaken freshness, or claim that nominal cadence guarantees operating-system,
network, or provider latency. Subsecond ML and a slower foundation-model risk
supervisor require a separate prospective contract; the current per-entry LLM
must not pretend to provide that capability.

## Verification

- Unit cases cover exact and multi-poll deadlines plus non-finite inputs.
- CLI coverage proves an impossible one-second configuration is rejected before
  exchange setup.
- The existing active-AI autonomous path still reaches the guarded loop when
  its interval can revisit the exact case.

No graph was regenerated because no economic or AI-uplift experiment occurred.
