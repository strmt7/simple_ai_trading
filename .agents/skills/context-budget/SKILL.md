---
name: context-budget
description: Bound repository context reads and semantic-search output without weakening source, test, or evidence verification.
---

# Context Budget

## Goal

Limit reads to the minimum high-signal surface required to implement the task correctly, then widen only when evidence is missing.

## Workflow

1. Read `AGENTS.md`.
2. Identify exactly which module and command path is affected.
3. Open one source file, one test file, and one skill that governs the area.
4. Implement with tight scope.
5. If uncertainty remains, open one more implementation file and one more test file only.

## Hard limits

- Avoid bulk reading unrelated directories and generated artifacts.
- Avoid opening entire modules for tangential concerns.
- Prefer `rg` for symbol search over opening large files.
- Keep semantic routing at the five-result default; ten results is the hard
  maximum. Refine the query or use path/language filters before widening.
- Treat measured UTF-8 output bytes as a context-volume proxy only. Do not
  infer or publish token savings without a tokenizer-specific measurement.

## Exit condition

Before coding the next step, verify that remaining assumptions are either covered by
existing tests or addressed by newly added tests.
