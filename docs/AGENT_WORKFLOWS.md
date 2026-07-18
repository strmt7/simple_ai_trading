# Agent Workflows

This repository carries the applicable agent tooling from
[`ZMB-UZH/omero-docker-extended`](https://github.com/ZMB-UZH/omero-docker-extended)
at commit `246110b1045cfd4ca318b4e870b5a38d213399b6`. The files are adapted for a
Windows-first Python trading repository; OMERO, Django, and container-specific
skills are intentionally not copied.

## Tooling

| Tool | Pinned version | Repository entry point |
| --- | --- | --- |
| CocoIndex Code | `0.2.37` | `tools/cocoindex_agent_search.py` |
| uv | `0.11.29` | `pyproject.toml` and `uv.lock` |
| Ruff | `0.15.22` | `.github/workflows/ruff.yml` |
| Vulture | `2.16` | `tools/vulture_check.py` and `.github/workflows/vulture.yml` |
| Super-Linter | `v8.7.0` | `.github/workflows/super-linter.yml` |
| Agent skills | ECC `2.0.0` | `.agents/skills/` |
| Karpathy guidelines | commit `2c606141936f1eeef17fa3043a72095b4765b9c2` | `.agents/skills/karpathy-guidelines/` |

A fresh 2026-07-18 upstream check found CocoIndex Code `0.2.37`, Vulture
`2.16`, and Super-Linter `v8.7.0` still current. The pinned Karpathy commit is
still its upstream `HEAD`; OMERO advanced to the exact commit above and its
applicable ECC `2.0.0` skill changes were reviewed and adapted. Ruff and uv were
updated only after reviewing their release notes and pinned action commits.

CI and release jobs use `uv sync --locked`; `uv.lock` is the cross-platform,
hash-bound dependency record. Dependabot may propose monthly `uv` and GitHub
Actions updates, but it cannot merge them. Accelerator and numerical-library
changes still require host compatibility and model-parity evidence.

The main CI workflow also runs `tools/audit_financial_terminology.py`. It rejects
superseded labels in authored documentation, Windows UI text, publication
generators, and tracked evidence filenames while preserving immutable raw model
responses and backward-compatible serialized identifiers.

The imported repo-local skills are `cocoindex-code-search`, `search-first`,
`source-audit`, `ai-regression-testing`, `docs-knowledge-maintainer`, and
`karpathy-guidelines`; `context-budget` is the repository's local context
overlay. Together they enforce testnet safety, reproducible financial evidence,
CLI/Windows parity, secret hygiene, and the single-session rule.

## CocoIndex Contract

The `cocoindex-code-search` skill is mandatory for broad repository
semantic routing. Exact symbols and small result sets still use `rg`; every semantic
candidate must be confirmed with `rg` and direct reads before editing.

```powershell
python tools/cocoindex_agent_search.py mcp-config
python tools/cocoindex_agent_search.py mcp-install
python tools/cocoindex_agent_search.py mcp-smoke
```

The wrapper stores its install, mirror, runtime, and database under
`AGENT_COCOINDEX_HOME`. Its default is an external cache under
`%LOCALAPPDATA%\SimpleAITrading` on Windows or the XDG data root on POSIX. A
cold index is created only by an explicit indexing command; `.cocoindex_code/`
must never be written to the live checkout.

Use `index --allow-dirty-index` only when the worktree snapshot is intentional,
or `search --refresh "<query>"` on a clean tree. MCP search itself never refreshes.
It can therefore return stale active-index text until an explicit
refresh. The mirror includes Git-visible text-decodable files and skips binary
content; semantic results are routing evidence, not correctness evidence.

`mcp-smoke` validates registration and the JSON-RPC handshake without creating
an index. The package benchmark cases and dependency hashes are recorded in
[`reference/cocoindex-code-agent-benchmark-2026-07-11.md`](reference/cocoindex-code-agent-benchmark-2026-07-11.md).

Semantic routing defaults to five results and rejects limits above ten. Refine
queries or add path/language filters before widening. Benchmark artifacts record
both characters and exact UTF-8 output bytes for broad `rg`, semantic routing,
and the focused hybrid path. Bytes are a reproducible context-volume proxy, not
a claim about model-specific token usage.

## Verification Lanes

Use the narrowest relevant checks while iterating, then the complete suite at a
promotion or release boundary:

Record each passing command, relevant tree state, and artifact. Do not rerun an
unchanged gate merely for reassurance; invalidate it only when code,
configuration, fixtures, dependencies, runtime artifacts, or platform inputs
change. Run the complete required matrix once against the final release tree.

```powershell
python -m ruff check .
python -m ruff format --check .
python tools/vulture_check.py
python tools/update_readme_badges.py --check
python -m pytest -q
```

The badges in `README.md` are generated from `.github/readme_badges.json` and
must not be hand-edited.

After a current AI governance benchmark, use
`tools/build_ai_model_provenance.py` to rescore the exact reports and verify the
Ollama manifest, config, and every referenced blob before atomically writing
`model-provenance.json`. Protected one-shot reports must also carry matching
pre/post-inference digest and metadata hashes plus positive exact-digest GPU
residency; provenance v2 rejects local files that differ from that evidence. Do
not scan or hash model files manually, and do not use this tool with historical
benchmark contracts.

## Transfer Verification

The 2026-07-11 Windows-host transfer check passed the six-skill validator,
CocoIndex contract suite and four-version JSON-RPC handshake, Ruff `0.15.22`,
Vulture `2.16`, yamllint `1.38.0`, markdownlint-cli `0.49.0`, actionlint
`1.7.12`, and Zizmor `1.26.1 --pedantic`. Zizmor reported no findings after
all remote Actions were commit-pinned. The full Super-Linter container remains
the GitHub-hosted integration check represented by its README badge.
