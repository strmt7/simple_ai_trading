---
name: karpathy-guidelines
description: Apply four anti-error coding principles plus this repository's stricter financial-evidence, safety, and single-session rules.
license: MIT
metadata:
  origin: "multica-ai/andrej-karpathy-skills at 2c606141936f1eeef17fa3043a72095b4765b9c2"
---

# Karpathy Guidelines

Apply these principles before the repository-specific overlay. They reduce
wrong assumptions and unnecessary code; they do not replace risk controls or
verification.

## Four Principles

1. **Think before coding.** State uncertain assumptions, expose conflicting
   interpretations, and verify unfamiliar behavior before choosing a path.
2. **Prefer the simplest complete solution.** Add no speculative abstraction or
   configuration. Complexity must remove demonstrated risk or duplication.
3. **Make surgical changes.** Preserve unrelated work, match established local
   contracts, and remove only orphans created by the current change.
4. **Work to verifiable outcomes.** Define the behavior and evidence that prove
   success, then loop until the relevant checks pass.

## Trading Overlay

- Never infer profitability, financial edge, risk containment, data quality, or
  production readiness. Require reproducible artifacts and independent gates.
- Keep signed execution testnet-only and fail closed on uncertain ownership,
  stale market data, incomplete reconciliation, exhausted API budget, or
  invalid model evidence.
- Preserve CLI/Windows functional parity through generated contracts.
- Use focused checks while iterating and full verification at promotion or
  release boundaries.
- Never expose credentials or use another agent/session. `AGENTS.md` is the
  authoritative stricter policy.
