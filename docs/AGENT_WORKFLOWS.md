# Agent Workflows

This repository carries the applicable agent tooling from
[`ZMB-UZH/omero-docker-extended`](https://github.com/ZMB-UZH/omero-docker-extended)
at commit `b27dbe990703d64d13e540c40cf4e122954c664d`. The files are adapted for a
Windows-first Python trading repository; OMERO, Django, and container-specific
skills are intentionally not copied.

## Tooling

| Tool | Pinned version | Repository entry point |
| --- | --- | --- |
| CocoIndex Code | `0.2.37` | `tools/cocoindex_agent_search.py` |
| Ruff | `0.15.21` | `.github/workflows/ruff.yml` |
| Vulture | `2.16` | `tools/vulture_check.py` and `.github/workflows/vulture.yml` |
| Super-Linter | `v8.7.0` | `.github/workflows/super-linter.yml` |
| Karpathy guidelines | commit `2c606141936f1eeef17fa3043a72095b4765b9c2` | `.agents/skills/karpathy-guidelines/` |

The imported repo-local skills are `cocoindex-code-search`, `search-first`,
`source-audit`, `ai-regression-testing`, `docs-knowledge-maintainer`, and
`karpathy-guidelines`. Their overlays enforce testnet safety, reproducible
financial evidence, CLI/Windows parity, secret hygiene, and the repository's
single-session rule.

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

## Verification Lanes

Use the narrowest relevant checks while iterating, then the complete suite at a
promotion or release boundary:

```powershell
python -m ruff check .
python -m ruff format --check .
python tools/vulture_check.py
python tools/update_readme_badges.py --check
python -m pytest -q
```

The badges in `README.md` are generated from `.github/readme_badges.json` and
must not be hand-edited.

## Transfer Verification

The 2026-07-11 Windows-host transfer check passed the six-skill validator,
CocoIndex contract suite and four-version JSON-RPC handshake, Ruff `0.15.21`,
Vulture `2.16`, yamllint `1.38.0`, markdownlint-cli `0.49.0`, actionlint
`1.7.12`, and Zizmor `1.26.1 --pedantic`. Zizmor reported no findings after
all remote Actions were commit-pinned. The full Super-Linter container remains
the GitHub-hosted integration check represented by its README badge.
