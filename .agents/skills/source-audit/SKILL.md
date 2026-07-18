---
name: source-audit
description: Audit financial, data, model, and dependency claims for primary-source quality, dates, provenance, and reproducibility before implementation or publication.
metadata:
  origin: "adapted from ZMB-UZH/omero-docker-extended at 246110b1045cfd4ca318b4e870b5a38d213399b6; ECC v2.0.0 reviewed"
---

# Source Audit

Use this skill for research that can affect execution, risk, model selection,
historical data, dependencies, or published performance claims.

## Evidence Record

For each material claim record:

- publisher, canonical URL, publication/access date, and exact version;
- whether the source is primary, peer reviewed, official, or secondary;
- the directly supported fact, separate from local inference;
- conflicts, missing assumptions, survivorship bias, leakage risk, or scope
  mismatch;
- the repository test, artifact, or experiment needed before adoption.

## Financial Rules

- Exchange rules and rate limits come from official exchange documentation or
  observed response headers.
- Dataset claims require checksums, UTC coverage, interval, gaps, schema, and
  source lineage.
- Strategy claims require out-of-sample results after fees, spread, latency,
  slippage, funding, liquidation, and selection effects.
- AI/ML improvement requires paired, hash-bound evidence on untouched periods;
  a higher score or a plausible narrative is not financial edge.
- Never backfill missing results from a chart, another repository, or memory.

State unresolved uncertainty instead of filling it with an assumption.
