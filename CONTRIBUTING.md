# Contributing

Keep changes coherent, evidence-backed and reviewable. Large architectural
refactors are appropriate when justified; deliver them in verified checkpoints
and preserve public contracts and historical evidence.

## Ground rules

- Python >= 3.11. Runtime dependencies are constrained in `pyproject.toml`;
  adding a new one needs a justification in the PR description.
- 100% branch coverage is enforced in CI. Every new branch needs a direct
  test assertion.
- Testnet-first safety defaults are load-bearing. A change that flips
  `testnet=True` or `dry_run=True` as a default is a contract break and must be
  flagged explicitly in the PR.
- Never commit secrets. `.gitignore` excludes `.env` and the project configs
  are stored under `~/.config/simple_ai_trading/` (mode 0600)
  at runtime.
- The local CLI reads environment variables from the process environment; it
  does not auto-load `.env`. Docker Compose may read `.env` and pass listed
  values through `docker-compose.yml`.

## Local workflow

Use the locked project environment on Windows/PowerShell and Unix shells.
For a fresh contributor environment, run `uv sync --locked --group test`.
Do not install unpinned development tools into an ad hoc environment.

Replace these example paths with the affected tests and authored Python files:

```bash
uv run --locked --group test python -m pytest tests/test_threshold_counts.py
uv run --locked --group test python -m ruff check src/simple_ai_trading/threshold_counts.py
uv run --locked --group test python -m ruff format --check src/simple_ai_trading/threshold_counts.py
```

Run focused tests during iteration and the affected domain at a behavior
checkpoint. At a shared-core or release verification boundary, use:

```bash
uv run --locked --group test python -m coverage run -m pytest
uv run --locked --group test python -m coverage report --fail-under=100
```

The coverage configuration in `pyproject.toml` defines branch and source scope.
Use the existing isolated accelerator launcher for GPU-specific work; the base
locked environment is not proof that an optional GPU backend is enabled.
Launcher and native-interface changes also require platform-specific smoke and
shared-command-contract checks.

## Codebase consistency standard

Engineering judgment leads this standard. Understand intent, financial semantics,
failure modes and the surrounding architecture before changing code. Challenge
unnecessary complexity, duplication and stale conventions. Tests and formatters
support this reasoning; they cannot establish design quality, semantic review or
financial correctness by themselves. Do not optimize for a green badge, comment
count or uniform appearance at the expense of clarity and correct behavior.

Repair root causes at the owning shared boundary. Do not mask defects with
caller-specific bypasses, silent exception suppression, weakened validation or
test-only production branches. Any supported degraded mode must be explicit,
observable and preserve the relevant safety invariants; it is not permission
to fabricate successful state. Challenge each proposed action's information gain,
failure consequences and complexity cost. Enterprise readiness is an acceptance
obligation backed by evidence, never a label inferred from passing tests alone.

Use one shared style for maintained code throughout the revamp and final
line-by-line review. This is a migration requirement, not a claim that legacy
code already complies:

- **Formatting:** use the pinned Ruff formatter and lint configuration in
  `pyproject.toml`, resolved through `uv.lock`. Avoid per-file quote, indentation
  and wrapping preferences. Respect idiomatic conventions in other languages.
- **Names and types:** use Python `snake_case` functions/variables/modules,
  `PascalCase` types and `UPPER_SNAKE_CASE` constants. Prefer precise domain names,
  explicit units and modern annotations (`list[T]`, `X | Y`). Distinguish quote
  amounts, quantities, percentages, fractions, basis points and timestamp kinds.
  Public/persisted names need compatibility and migration consideration, not a
  cosmetic rename. Keep genuinely different venue contracts distinct.
- **Structure:** keep modules cohesive, dependencies explicit and I/O at clear
  boundaries. Use frozen dataclasses for data in motion and existing config,
  persistence and ownership contracts. Share abstractions when semantics match;
  avoid duplicate utilities and speculative framework layers.
- **Comments:** write concise, grammatical English in a consistent voice.
  Explain intent, invariants, safety/concurrency constraints and non-obvious
  tradeoffs; do not narrate obvious statements. Avoid decorative banners,
  promotional language, author/model signatures and commented-out dead code in
  maintained implementations. Preserve licenses, provenance, safety rationale
  and tool directives. TODOs need a concrete completion condition and a tracked
  issue or repo-plan reference.
- **Docstrings:** prefer a concise imperative summary; add consistently labeled
  `Args:`, `Returns:` and `Raises:` sections only for nontrivial contracts where
  helpful. Explain units, side effects and failure behavior rather than repeat
  annotations. Do not add boilerplate to every trivial helper.
- **Errors and logging:** reuse domain exceptions, stable reason codes and
  redaction helpers. Preserve causal errors; failed/unknown account state is not
  success or zero. Keep operational events structured and user messages clear.
  Never expose secrets or signed requests through diagnostics.
- **Tests and interfaces:** use descriptive behavior-based names, readable
  arrange/act/assert structure and deterministic inputs. Share fixtures only
  when they clarify contracts. Align terminology, units, settings, status and
  action semantics across CLI/native interfaces and their common backend.

Migrate in reviewable units. Check new modules in full and changed ranges in
legacy modules while their remaining review is pending, following current CI
policy. Do not mass-format retained third-party data, generated artifacts or
consumed hash-bound implementations. Regenerate outputs from their authoritative
source where appropriate; review frozen historical code without rewriting it.
Document justified exceptions and their review disposition. Similar appearance
is not a reason to erase meaningful language, ownership or venue differences.

The final [whole-codebase review and bug hunt](docs/CAPITAL_PROTECTION_ARCHITECTURE.md)
must account for every file and code line, including style and architecture
consistency. Reasoned review and behavioral evidence are both required; neither
automated checks alone nor reviewer confidence alone establishes completion.

## Commit etiquette

- One logical change per commit. Tests first (or alongside) the code.
- Subject line in imperative mood (`add`, `fix`, `refactor`), <= 72 chars.
- Body explains the *why* when it is non-obvious; the code already says the
  *what*.

## Pushing

If you need to push with a Personal Access Token, use the helper:

```bash
python3 tools/push_with_pat.py origin my-branch
```

The helper serves your PAT to `git push` over a short-lived UNIX socket so it
never appears in `argv`, remotes, logs, shell history, or long-lived credential
stores.
