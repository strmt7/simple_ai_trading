---
name: search-first
description: Research the existing implementation, tests, primary technical sources, and financial rationale before adding trading code or dependencies.
metadata:
  origin: "adapted from ZMB-UZH/omero-docker-extended at 246110b1045cfd4ca318b4e870b5a38d213399b6; ECC v2.0.0 reviewed"
---

# Search First

Use this skill before new model logic, exchange behavior, risk controls,
dependencies, data pipelines, workflows, or broad refactors.

## Tool Availability Preflight

Check only the channels relevant to the task before relying on them:

- repository search: prove `rg --files` and targeted `rg` work;
- package registry: prove the project package manager can query metadata;
- GitHub: check `gh auth status`, or use public Git and official web sources;
- MCP/docs tools: inspect the active tool list and use official docs if absent;
- skills: inspect the repository and local Codex skill catalogs.

State any unavailable or skipped channel. Never report "nothing found" when a
search path was unavailable.

## Search Order

1. Read `README.md`, the nearest `src/simple_ai_trading/` module, and its tests.
2. Use `rg` for exact questions. Use `cocoindex-code-search` first only for
   genuinely broad semantic routing.
3. Read the closest design, provenance, model, or simulation document.
4. Check official exchange/API documentation and primary library/release docs.
5. For model or market claims, inspect peer-reviewed papers or authoritative
   market-microstructure references and record assumptions and limitations.
6. Compare maintained open-source implementations only after understanding
   their data, leakage controls, execution model, and license.
7. Adopt an existing local pattern, extend it minimally, or document why a new
   implementation is necessary.

## Rules

- Do not treat TradingView scripts, forum claims, stars, backtest screenshots,
  or vendor marketing as proof of edge.
- Pin version-sensitive facts and cite exact sources in durable research docs.
- Keep tests offline unless the network boundary is explicitly stubbed.
- Never put credentials, account details, or private endpoints in queries.
- Do not use subagents or separate sessions for research in this repository.
- Treat any upstream `Agent(...)` or legacy `Task(...)` research example as
  reference-only; this repository's single-session rule takes precedence.
