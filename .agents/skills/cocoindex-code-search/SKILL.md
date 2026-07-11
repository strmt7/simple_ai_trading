---
name: cocoindex-code-search
description: Use the pinned external CocoIndex Code MCP workflow for broad repository routing, then prove exact candidates with rg and direct file reads.
metadata:
  origin: "adapted from ZMB-UZH/omero-docker-extended at b27dbe990703d64d13e540c40cf4e122954c664d"
---

# CocoIndex Code Search

Use this skill only when a broad question would otherwise require a large
repository scan. Exact strings, symbols, filenames, and small result sets go
directly to `rg`.

## Required Flow

1. Check for an existing MCP server/tool named `cocoindex-code`.
2. If absent or stale, run `python tools/cocoindex_agent_search.py mcp-install`.
3. Prove registration with `python tools/cocoindex_agent_search.py mcp-smoke`.
   Registration alone is not evidence; the JSON-RPC initialize and tool-list
   probes must pass.
4. Create or refresh an index only when the disk write is intentional:
   `index --allow-dirty-index` for an intentional worktree snapshot, or
   `search --refresh "<query>"` on a clean tree.
5. Route broad queries through MCP or
   `python tools/cocoindex_agent_search.py search --limit 5 "<query>"`.
6. Confirm every candidate with exact `rg` and direct reads in the live repo.
7. Run the ten-case benchmark after changing this workflow or upgrading the
   pinned package.

## Safety Contract

- Keep `cocoindex-code[full]==0.2.37` pinned.
- Never run `ccc init` in the live checkout and never commit
  `.cocoindex_code/`.
- Store the shared install and per-repository mirrors/databases under
  `AGENT_COCOINDEX_HOME` or the platform-specific external default.
- Do not mirror real `.env` files, credentials, ignored files, or private
  artifacts. Binary files are not semantic-search evidence.
- MCP search is read-only and never refreshes an index implicitly. Treat stale
  results as routing hints, not current truth.
- The wrapper may reuse an existing daemon and must stop only daemons it owns.
- Do not build a cold index during high-load model training or without enough
  disk headroom.

The portable contract, package hashes, and benchmark inputs are documented in
`docs/AGENT_WORKFLOWS.md` and `docs/reference/cocoindex-code-agent-benchmark-2026-07-11.md`.
