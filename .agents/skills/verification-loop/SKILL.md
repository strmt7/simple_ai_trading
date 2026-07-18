---
name: verification-loop
description: Verify changed trading boundaries efficiently, invalidate checks by input hash, and run the complete matrix once on the final tree.
metadata:
  origin: "adapted from ZMB-UZH/omero-docker-extended at 246110b1045cfd4ca318b4e870b5a38d213399b6; ECC v2.0.0 reviewed"
---

# Verification Loop

## Principle

Prefer small, deterministic verification that directly confirms each assumption.

## Efficiency and invalidation

- Keep a ledger of the exact command, relevant tree state, result, and artifact.
- Do not repeat a passing command until its code, configuration, fixtures,
  dependencies, runtime artifact, or platform input changes.
- Run independent read-only checks in parallel when resources permit. Serialize
  builds, training, and live checks that mutate shared databases or exchanges.
- A final repository-wide matrix supersedes unchanged targeted checks.

## Process

1. Run targeted tests first (same module).
2. Add/adjust tests for any new branch.
3. Run the full suite once against the final tree before finishing broad changes.
4. Run coverage and confirm critical lanes have no uncovered new branches.
5. For CI-facing behavior, validate local workflow equivalents (same Python version and command shape used in `.github/workflows/ci.yml`).

## Rule

Do not report completion without showing the exact command/output outcome for each changed domain.
