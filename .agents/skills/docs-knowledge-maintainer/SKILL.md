---
name: docs-knowledge-maintainer
description: Keep concise user docs and machine-readable model evidence synchronized with real behavior without losing provenance or caveats.
metadata:
  origin: "adapted from ZMB-UZH/omero-docker-extended at 246110b1045cfd4ca318b4e870b5a38d213399b6; ECC v2.0.0 reviewed"
---

# Docs Knowledge Maintainer

Use this skill whenever behavior, operating assumptions, workflows, data,
models, risk gates, UI controls, or published evidence changes.

## Routing

- User scope, install, safety, and primary commands: `README.md`.
- Agent tools and provenance: `docs/AGENT_WORKFLOWS.md`.
- Data lineage and truth requirements: `docs/DATA_PROVENANCE_POLICY.md`.
- Execution realism: `docs/LIVE_MARKET_SIMULATION.md`.
- Model methodology and promotion: model research documents and generated
  `latest/` evidence directories.
- CLI/Windows behavior: generated command contracts plus parity tests.
- Release behavior: `docs/release.md` and the manual release workflow.

## Evidence Rules

- Keep the README short enough to scan; link to the nearest complete deep doc.
- Do not delete information unless it is obsolete and its replacement preserves
  the operating contract.
- Generate charts from committed CSV/JSON evidence. Never edit plotted values by
  hand or publish a graph without its source table and UTC span.
- Publish only the latest iteration charts plus the cumulative progress series,
  while retaining truthful machine-readable provenance needed to reproduce them.
- Label unavailable, blocked, simulated, testnet, and not-run evidence plainly.
- Update docs, tests, and generated badge metadata together when a contract
  changes.

Verify links, generated badge drift, relevant doc contracts, and the focused
tests for the changed behavior before publication.
