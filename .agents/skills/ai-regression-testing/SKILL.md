---
name: ai-regression-testing
description: Choose staged contract tests that catch partial AI-generated fixes across models, data, risk, execution ownership, and CLI/Windows parity.
metadata:
  origin: "adapted from ZMB-UZH/omero-docker-extended at 246110b1045cfd4ca318b4e870b5a38d213399b6; ECC v2.0.0 reviewed"
---

# AI Regression Testing

Use this skill when an agent changes behavior that can look correct on one path
while breaking a paired path or safety boundary.

## Staged Lanes

1. Reproduce the exact defect or violated contract with the smallest test.
2. Exercise both success and failure branches, including timeout, malformed
   input, partial exchange response, interruption, and restart when relevant.
3. Run paired boundaries touched by the change:
   CLI/Windows, spot/futures, long/short, CPU/accelerator, fresh/reconnect,
   model/AI-disabled, train/reload, and simulated/live-testnet.
4. Validate artifacts and hashes separately from in-memory results.
5. Run the complete suite only at a promotion, integration, or release boundary
   unless the blast radius requires it earlier.

## Non-Negotiable Contracts

- No network access in unit tests.
- No order may bypass ownership, reconciliation, rate-limit, liquidity, loss,
  or stale-data gates.
- Generated CLI/Windows parity must fail when a command, option, choice, or
  control mapping drifts.
- Financial tests must reject non-finite values, leakage, impossible accounting,
  missing provenance, and fabricated or hand-edited evidence.
- A passing smoke test never substitutes for model validation or risk evidence.
