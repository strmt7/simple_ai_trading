# Agent Start

Read `AGENTS.md`, then `docs/CONTINUATION.md`. Those two files are the current
operating contract. Historical handoff text is archived under
`docs/archive/agent-history/`; use it only for provenance.

## Current Truth

| Boundary | State |
| --- | --- |
| Release | `0.1.0-beta.1`; experimental |
| Development branch | `main` only |
| Binance | BTC, ETH, and SOL; paper or testnet/Demo only |
| Polymarket | Independent BTC 5-minute/15-minute research; disabled by default |
| Accepted edge | None |
| Live-money authority | None |
| Historical cutoff | `2026-08-14T00:00:00Z` |

No model, AI component, backtest, capture, or paper result may be described as
profitable without reproducible source-bound after-cost evidence. AI may veto
or reduce risk only; it never creates positions, selects leverage, overrides a
safety gate, blocks Stop, or submits an order.

## Non-Negotiable Gates

- Aggregate performance is not an all-market edge. Promotion requires causal
  bull, bear, sideways, choppy, high-volatility, liquidity-stress, and
  latency-stress slices under
  `docs/model-research/cross-regime-edge-acceptance-contract-v1.json`.
  Unsupported regimes must abstain from new risk.
- Only provably bot-owned orders and positions may be modified. Unknown order,
  fill, fee, balance, redemption, or reconciliation state blocks exposure.
- Risk, ownership, reconciliation, Pause, and Stop remain deterministic and
  independent of model or AI availability.
- Historical labels, future books, fills, resolutions, and PnL never enter live
  inference. Secrets never enter prompts, logs, tests, artifacts, or commits.
- Leverage changes exposure, not edge. No leverage profile bypasses liquidity,
  drawdown, cooldown, ownership, execution-cost, or regime gates.

## Research State

- Round 75 ended and is rejected. Its metadata-only terminal audit found 35
  result slots, 33 admitted training epochs, 684 missed slots, incomplete slot
  67, no tuning or test epochs, and a retained WAL. The training-role raw
  eligible anchor was `28,903,469,878,300 ns` versus
  `394,740,000,000,000 ns` required. Do not train, tune, access sealed targets,
  or claim an edge from this campaign.
- The exact Round 75 sources are preserved under
  `docs/model-research/action-value/round-075-frozen-v4-source/`. Active
  supervisor v3 treats an expired campaign as non-restartable.
- Polymarket Round 29 is blocked before feature, target, or model access. Stage
  1 produced one terminal primary slot, one incomplete slot, and no third
  primary date; it cannot satisfy the frozen three-date/300-market source gate.
- The 2026-08-25 target-free structural-parity screens found no accepted edge.
  Polymarket had zero gross-positive paths across 22 fixed BTC/ETH/SOL
  negative-risk events and zero gross-positive logical-implication bundles
  across 2,572 threshold/deadline pairs. Binance's best three-leg spot cycle
  was only 0.6462 bps gross and required less than 0.2154 bps fee per leg to
  break even; exact account fees remain unaudited, so this is not an executable
  edge. Binance option vertical/convexity parity covered 365,592 exact payoff
  identities across 1,538 unit-one contracts. Two ticker-only candidates
  disappeared at displayed depth, where every exact minimum portfolio was
  already negative before fees. A distinct fixed-payoff box screen found six
  strict short-box and one nominal long-box ticker candidates across 13,344
  strike pairs; every candidate lacked executable fresh depth. Do not repeat
  these screens without a frozen prospective sampling contract or materially
  new execution evidence. Polymarket cross-condition duplicate discovery found
  one repeated exact question across 607 eligible binary markets, but its two
  canonical rule sets differed; zero exact payout-rule duplicates advanced to
  pricing.
- A shared source-continuity gate now permits only slot-local failure
  containment for future, separately activated Binance and Polymarket
  campaigns. It is design-only: no future schedule, capture, target, model, or
  authority is active.
- The independent Round 21 sidecar remains protected until
  `2026-08-29T23:40:00Z`. Do not stop, restart, stage, clean, reset, switch,
  commit, or modify its process, worktree, state, database, or WAL.

## Task Routing

| Work | Read first |
| --- | --- |
| Current plan or handoff | `docs/CONTINUATION.md` |
| Binance model/backtest | `docs/model-research/action-value/latest/README.md` |
| Polymarket model | `docs/model-research/polymarket/latest/README.md` |
| Structural parity | `structural_parity.py`, `logical_parity.py`, and the three 2026-08-25 snapshots |
| Model promotion | `docs/MODEL_AND_SIGNAL_VALIDATION.md` and cross-regime contract |
| Execution/risk | `docs/LIVE_MARKET_SIMULATION.md` and venue runbook |
| AI | `docs/ai/risk-review/latest/comparison.json` |
| Windows/CLI parity | `src/simple_ai_trading/command_contract.py` and parity tests |
| CI/release | `docs/AGENT_WORKFLOWS.md` and `docs/release.md` |

Before editing, verify `git status`, `git worktree list`, active processes,
scheduled tasks, `origin/main`, open alerts, and the exact evidence boundary.
Never infer current host state from an old PID or archived note.
