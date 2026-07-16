# Agent Start

This is the smallest safe entry point for work in this repository. It routes
agents to canonical evidence without replacing that evidence.

## Non-negotiable truth

- Scope is BTC, ETH, and SOL. Binance is testnet/Demo or paper only. Polymarket
  is paper and research only. No mainnet or live-money authority exists.
- Conservative is the default profile. Leverage is a risk ceiling, never a
  source of edge. Profitability, ROI, and drawdown claims require reproducible
  source-bound after-cost evidence.
- Risk, reconciliation, Stop, and ownership checks are deterministic. AI may
  veto or downsize only after matched uplift evidence and may never block a
  close or override a safety gate.
- Historical labels, future books, resolutions, fills, and PnL must never enter
  a live inference payload. Unknown order state blocks new exposure.
- Secrets must never enter prompts, logs, artifacts, tests, commits, or docs.

## Task routing

| Task | Read first | Canonical evidence |
|---|---|---|
| Binance model or backtest | nearest model module and test | `docs/model-research/action-value/latest/README.md`, then selected rows from `progress.csv` |
| Prior model failure | last row plus the relevant mechanism row in `docs/model-research/action-value/latest/progress.csv` | that row's named design/report only |
| Polymarket model | `src/simple_ai_trading/polymarket_model.py` and `tests/test_polymarket_model.py` | `docs/model-research/polymarket/latest/README.md` and the nearest numbered contract |
| Polymarket recorder/replay | matching recorder or replay module and test | `docs/model-research/polymarket/prospective-continuity-contract-v2.json` |
| Risk or execution | nearest risk/execution module and test | `docs/LIVE_MARKET_SIMULATION.md` or `docs/POLYMARKET_PAPER_TRADING.md` only at the relevant heading |
| AI provider/model | nearest AI module and test | `docs/ai/risk-review/latest/comparison.json` and its sibling provenance |
| CLI | command handler, parser definition, and CLI tests | parser-generated help; do not infer parity from docs |
| Windows app | `src/simple_ai_trading/command_contract.py`, `src/simple_ai_trading/windows_app.py`, and the UI/parity tests | `native/windows/generated/command_contract.hpp` and `tests/test_ai_runtime_and_parity.py`; edit the shared taxonomy, never the generated header |
| CI/release | one workflow and its test/lint config | `docs/AGENT_WORKFLOWS.md` |
| Broad architecture | `docs/SIMILAR_TRADING_REPOS_REVIEW.md` | source and tests for each affected boundary |

## Model research state

- The compact cross-round ledger is
  `docs/model-research/action-value/latest/progress.csv`. Read its header and
  only the last few rows unless a task names an older mechanism.
- Round 61 rejected elevated-funding spot/perpetual carry on capacity, median
  after-cost return, and lower-confidence-bound gates. Do not tune or retrain
  that family.
- Polymarket currently has a market-anchored baseline, purged BTC/ETH/SOL
  splits, exact depth/fee/latency replay, causal resolution-time cash locking,
  hash-bound label-free inference, a frozen causal action-value contract, an
  implemented hash-persistent ridge baseline, and a preregistered warning-free
  nonlinear challenger. No Round 9 model has been fitted or scored.
  Post-contract continuity-qualified outcomes and prospective results are still
  pending, so no profitability or execution authority exists.
- Captures `eae374e2662c440fb93970d5710937b1`,
  `3a67757c7f174df4b62f2722ea9211cb`, and
  `b8a270da20fe4116a01a4626607e42da` are permanently development-only. The
  first two queues saturated. The third was terminalized `failed` after
  9,887,714 persisted messages when its indexed v2 writer reproduced the same
  long-duration throughput collapse. Storage-v3 capture
  `79ac19539d384352b865c21cb0c43627` is also permanently development-only: its
  queue reached `500000/500000` after 10.1 hours, it was deliberately stopped,
  fully drained, and terminalized `failed`. Never use any of these runs for
  model, confirmation, or profitability claims.
- Recorder storage v4 writes only bounded, checksummed frames containing exact
  payload bytes and receipt metadata; normalized events are reconstructed and
  the terminal report binds the ordered chunk-manifest root. Its 2,000,000-
  message infrastructure benchmark sustained 48,189 messages/s, replayed at
  66,049 messages/s, passed the full audit, and used 198,717,440 bytes. The
  repeated payload sample was real and hash-verified, but receipt metadata was
  synthetic and the source run failed, so this is not live-capture, model, or
  profitability evidence. A subsequent five-minute real-feed soak captured
  470,422 messages with queue high-water `569/500000`, zero recorder or
  integrity errors, and exact reopen verification. One audited CLOB disconnect
  made it `degraded`; it validates writer liveness only. A 15-hour confirmation
  is still required.
- Round 9 MLP report v3 requires positive validation stress-utility uplift over
  ridge and at least 30 untouched synchronized test groups before reading its
  test partition. Do not weaken or bypass either admission gate.
- Round 9 maps `itode` to the independent 250 ms crypto taker delay and rejects
  nonzero general `sd`. V2 platform fees use `fd`; recorded base-fee fields are
  not additive and no builder code is modeled. The primary-source audit binds
  the official status record and SDK revisions.
- Round 9's one-second two-leg replay proves only causal CLOB book matches. It
  does not prove onchain confirmation or that newly bought tokens are sellable;
  official SELL prerequisites require confirmed conditional-token inventory.
  Keep all Round 9 outputs research-only until a separate settlement/inventory
  contract passes current source-bound failure and mark-to-market stress-test
  acceptance criteria.
- Ridge admission fails on any unproven post-submission entry state. Never
  censor or relabel it as no-fill; only a definite entry rejection such as an
  invalidated tick is a classifier-eligible zero-utility no-fill.
- Run Round 9 fits only through `polymarket-ridge` and `polymarket-mlp`. Both
  read only opaque row identities before writing a durable claim; clear labels
  load afterward. Completed claims load the signed report, while interrupted
  or failed claims block silent retries.
- Finance-LLM v6 is revoked for case-ID label leakage. V7 recorded Qwen3 8B at
  `9/11` and three 8B/9B models at `8/11`, but its permissive response parser
  invalidates the valid-JSON admission contract; keep those results as rejected
  historical evidence only. V8 preserves the 11 label-free cases and requires
  exact typed JSON. No AI model is selected. Kronos also failed its causal
  random-walk benchmark. Any AI treatment must pass current governance, then
  beat same-period non-AI execution after costs without worsening tail risk.
- AI permission maps are default-deny. Only a valid, timely approval for the
  exact hash-bound condition may permit that proposal; missing cases, malformed
  types, duplicate JSON keys, low confidence, and latency failures remain vetoes.
- AI review v4 also requires post-inference Ollama `/api/ps` evidence for the
  exact weight digest with positive VRAM-resident bytes. DirectML selection is
  separate and does not prove that the review model ran on GPU.
- Qwen3 14B is the next one-shot v8 candidate, frozen before installation in
  `docs/ai/risk-review/qwen3-14b-v8-preregistration.json`. Run it only after
  a fresh confirmation recorder ends `complete`; do not alter prompts or cases
  first. The `ai-benchmark` CLI rejects this model without that preregistration,
  the confirmation database, and its run ID. Its DuckDB claim consumes the test
  once before inference; interrupted and failed claims cannot reopen it. A valid
  output also requires identical Ollama digest/metadata hashes before and after
  inference plus exact post-inference GPU residency. CPU-only or changed weights
  fail the consumed claim.
- The exact terminal facts for failed confirmation capture
  `79ac19539d384352b865c21cb0c43627` are in
  `docs/model-research/polymarket/round-009-confirmation5-failure-2026-07-16.json`.
  Its terminal integrity audit is incomplete; retain it only for recorder
  diagnosis and audit any payload sample before reuse.
- Confirmation capture `e34d349771da4c35bcc8ae436c2fe9f6` currently owns
  `data/polymarket-round9-confirmation-v4-20260716-152838Z.duckdb`; never open
  that database while its recorder process is active. Its sidecar has recorded
  CLOB reconnect gaps, so the full run cannot be called continuous even if it
  completes; only independently audited continuous segments may be admitted.
- Build current AI provenance with `tools/build_ai_model_provenance.py`; it must
  match protected inference evidence to the local manifest, `/api/show`
  metadata, and every blob. Never hand-edit the result or infer identity from a
  mutable tag.

## Efficient workflow

1. Inspect `git status` and the nearest source/test pair.
2. Use exact `rg` queries. Use CocoIndex only for genuinely broad semantic
   routing; confirm its candidates in source.
3. Freeze causal inputs, costs, roles, and rejection gates before reading a new
   evaluation outcome.
4. Run the smallest focused regression during development. Run the complete
   affected-domain suite once at the behavior checkpoint; run the repository
   suite only when the change crosses domains or before significant handoff.
5. Keep numeric evidence in canonical JSON/CSV and regenerate charts from it.
   Do not duplicate full evidence tables in prose.
6. Record a rejected mechanism in the compact progress ledger so later agents
   do not repeat it.

## Freshness rule

This file is routing context, not result evidence. If it conflicts with a
hash-bound report, source code, or test, the canonical artifact wins and this
file must be corrected in the same change.
