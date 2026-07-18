---
name: python-patterns
description: Python patterns for this repo — pure-stdlib trading core, typed dataclasses, no hidden dependencies, deterministic I/O.
metadata:
  origin: "adapted from ZMB-UZH/omero-docker-extended at 246110b1045cfd4ca318b4e870b5a38d213399b6; ECC v2.0.0 reviewed"
---

# Python Patterns

Use this skill whenever you add or refactor Python in `src/simple_ai_trading/`, `tests/`, or `tools/`.

## Ground rules

- Keep exchange, risk, accounting, and artifact contracts lightweight and deterministic. Optional model acceleration may use the declared NumPy/PyTorch/DirectML stack, but every runtime dependency must be explicit and every accelerated inference path needs a CPU parity test.
- Every module exposes **frozen dataclasses** for data-in-motion (`Candle`, `ModelRow`, `BacktestResult`, `ClassificationReport`, …) and regular dataclasses for config. Keep that split.
- New public functions must carry type hints and a one-line docstring when the "why" is non-obvious. Do not document obvious behavior.
- Avoid hidden globals. Pass configuration explicitly. If you need a session-lived object (e.g. the shell state), inject it through the constructor.
- Prefer small files. Any module crossing ~300 logical lines should be split along clean responsibility lines (`api`, `features`, `model`, `backtest`, `shell`, `chart`, `style`, …).

## Trading-specific rules

- **Credential redaction is a contract**, not a formatting choice. Any payload that could be printed, logged, or persisted runs through the existing redaction helpers (`RuntimeConfig.public_dict`, `_redact_request_url`, `_redact_sensitive_text`). New printable surfaces that skip this are blocking bugs.
- **Deterministic artifacts**: JSON written under `data/` must use `sort_keys=True` and keep a stable top-level schema (`command`, `timestamp`, `runtime`, …). Never embed raw API keys.
- **No side effects at import time** — no network calls, no file writes, no environment variable mutation. Tests rely on import being free.
- **Safety defaults survive**: `testnet=True`, `dry_run=True`, conservative risk, and the BTC/ETH/SOL major-asset scope. A change that weakens any of these requires an explicit user opt-in path with a test.

## Formatting / lint

- Ruff is the lint + format baseline (see `pyproject.toml`). Run `ruff format` and `ruff check --fix` on touched files before committing.
- Python ≥ 3.11 is required. Use `X | Y` unions, `list[T]`, `dict[K, V]`, and `from __future__ import annotations` consistently.

## Don't

- Don't silently widen exception handling to make a test pass — catch only what you know you need to recover from.
- Don't add CLI output that isn't covered by a test. The repo runs `coverage --fail-under=100`; an uncovered branch will break CI, not just coverage drift.
- Don't stash config inside module-level mutable state — use `config_paths()` / `load_runtime()` each time.
