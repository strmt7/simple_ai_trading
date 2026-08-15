# Documentation

This index is the human entry point. JSON, CSV, and generated SVG files are
evidence, not prose documentation; their bytes remain stable so results can be
reproduced and audited.

## Operate

| Need | Read |
| --- | --- |
| Install or choose compute | [Compute portability](DIRECTML_GPU.md) |
| Run Binance paper/testnet | [Non-mainnet runbook](../LIVE_TESTNET_RUNBOOK.md) |
| Understand execution realism | [Live-market simulation](LIVE_MARKET_SIMULATION.md) |
| Use Polymarket paper mode | [Polymarket paper trading](POLYMARKET_PAPER_TRADING.md) |
| Inspect the disabled live boundary | [Polymarket execution](POLYMARKET_LIVE_EXECUTION.md) |
| Build a beta release | [Release guide](release.md) |

## Understand

| Topic | Read |
| --- | --- |
| Architecture and open-source comparison | [Repository comparison](SIMILAR_TRADING_REPOS_REVIEW.md) |
| Data provenance and cutoff rules | [Data provenance policy](DATA_PROVENANCE_POLICY.md) |
| Model promotion and contamination controls | [Model validation](MODEL_AND_SIGNAL_VALIDATION.md) |
| Current model program | [Research and optimization](MODEL_RESEARCH_AND_OPTIMIZATION.md) |
| Model and AI selection | [AI model selection](AI_MODEL_SELECTION.md) |
| External integration boundaries | [Integrations](INTEGRATIONS_PLAN.md) |
| Security reporting | [Security policy](../SECURITY.md) |

## Current Evidence

- [Binance latest status](model-research/action-value/latest/README.md) links to
  source CSV/JSON and generated performance charts.
- [Polymarket latest status](model-research/polymarket/latest/README.md) records
  the current rejected, open, and blocked experiments.
- [Local AI comparison](ai/risk-review/latest/comparison.json) is the canonical
  machine-readable comparison. No AI model currently has order authority.
- [Microstructure availability](microstructure/availability.json) records the
  exact public archive boundary used by event-level research.

Historical round files are intentionally retained. They are useful for audit
and failure analysis, but they do not describe the current product unless a
current-status document links to them.

## Contribute

- [Continuation guide](CONTINUATION.md) is the authoritative current state and
  ordered handoff for a new development session.
- [Agent start](AGENT_START.md) routes code changes to the smallest canonical
  evidence surface.
- [Agent workflows](AGENT_WORKFLOWS.md) defines lint, test, documentation, and
  imported-tool contracts.
- [Main-only policy](MAIN_ONLY_BRANCH_POLICY.md) requires development on
  `main` and removal of temporary remote branches after verified integration.
- [AI commit identity](AI_COMMIT_IDENTITY.md) defines the required automated
  Git author and committer.

When prose and executable behavior disagree, source, tests, parser-generated
help, and hash-bound evidence take precedence. Update the prose in the same
commit that changes the behavior.
