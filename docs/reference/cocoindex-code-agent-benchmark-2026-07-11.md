# CocoIndex Code Agent Contract 2026-07-11

This note records the reproducible package evidence and the adapted benchmark
inputs for this repository. It does **not** claim a semantic-search performance
result: a full cold index was intentionally not created during the transfer.

## Package Evidence

- Package: `cocoindex-code[full]==0.2.37`
- PyPI version checked: 2026-07-15 (`0.2.37` remained latest)
- Wheel: `cocoindex_code-0.2.37-py3-none-any.whl`
- Wheel SHA256:
  `9510e2810fcec5cfe9c9fb42e42acb9910155d5b5de0d7514bfa42daeb21b9ba`
- Source SHA256:
  `089888f455f71bfcdef6426150ce7bbbb3ff067b4448a6aef943152e38b4b214`
- Source: [PyPI 0.2.37](https://pypi.org/project/cocoindex-code/0.2.37/)

The hashes above came from the PyPI release JSON and match the upstream OMERO
workflow. The wrapper is ported to Windows and POSIX, uses a pinned external
virtual environment, mirrors only Git-visible non-secret files, and keeps all
index state outside the checkout.

## Contract Checks

```powershell
python -m pytest -q tests/test_cocoindex_agent_search.py
python tools/cocoindex_agent_search.py mcp-config
python tools/cocoindex_agent_search.py mcp-smoke
```

The ten trading-specific benchmark cases are in
`cocoindex-code-agent-benchmark-2026-07-11-cases.json`. Run the full benchmark
only when a cold index is operationally justified:

```powershell
python tools/cocoindex_agent_search.py benchmark `
  --cases docs/reference/cocoindex-code-agent-benchmark-2026-07-11-cases.json `
  --output cocoindex-benchmark.json
```

Any future benchmark publication must include the repository commit, mirror
digest, package version, case-level expected-path ranks, timings, output volume,
database size, and the raw JSON result. Missing runs remain explicitly marked
as not run; they must never be reconstructed or inferred.

Benchmark schema 2 records characters and exact UTF-8 output bytes for each
route, plus the hybrid-to-broad-output ratio. The wrapper defaults to five
semantic candidates and rejects more than ten. Output bytes are a deterministic
context-volume proxy; they are not tokenizer-specific token counts.

## Live Host Result

On 2026-07-11, the Windows wrapper registered a host-stable launcher through
the direct TOML fallback because the packaged Codex executable was blocked by
WindowsApps ACLs. The written config was re-parsed and matched the expected
command, arguments, environment, and timeouts. Raw stdio probes negotiated
`2024-11-05`, `2025-03-26`, `2025-06-18`, and `2025-11-25`; every probe listed
the `search` tool. No package install, cold index, semantic query, or benchmark
was run, so this is integration evidence only.
