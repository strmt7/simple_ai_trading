# Whole-product reassessment — first substantive pass

Scope includes ideas, economics, evidence, plans, architecture, interface,
execution, integrations, dependencies, tests and operating instructions. This
document does not certify the whole repository reviewed. The file inventory
and remaining semantic coverage are in `docs/REVIEW_2026_09_04.md`.

## Central judgment

The inspected surfaces are organized mainly around running model experiments
and preserving their provenance. The user's primary objective is different:
find and measure repeatable after-cost market edges. Provenance and safety are
necessary, but more experiment commands, accepted mechanism counts, GPU
capability or green tests do not establish progress toward that objective.

A direction-neutral strategy can still depend on liquidity, financing,
collateral, execution, venue solvency and resolution rules. Reevaluate stability
as a repeatable decision rule with explicit abstention conditions and capacity,
not a promise of positive profit in every imaginable market state.

## Keep, change, retire or investigate

| Area | Decision | Evidence and rationale |
| --- | --- | --- |
| Payoff arithmetic | Keep, strengthen the rule-proof boundary | `structural_parity.py`, `logical_parity.py` and `quarterly_carry.py` provide reusable Decimal/depth logic. Labels and ordered deadlines still cannot prove identical observation functions. |
| Funding/carry ideas | Investigate integrated cash flow first | Entry basis, received funding, exit basis and total capital must be evaluated together. A one-snapshot entry screen is not a funding persistence study. |
| Maker package execution | Investigate completion-adjusted value | Matched-package profit must pay for adverse selection and failed/partial completion. Public fills are not queue-position evidence. |
| Existing-inventory locking | Retain as a separate promising overlay | The retained stressed aggregate remains positive but failed its predeclared breadth gate and is BTC-concentrated. It does not justify acquiring first-leg exposure. |
| Rebates/promotions | Demote as a research objective | Useful on otherwise justified activity; accepted scope counts and headline pools are not standalone stable profit. Keep exact eligibility and no-double-counting rules. |
| AI/GPU work | Require marginal economic value before expansion | `entrypoint.py`, `command_contract.py` and native research shortcuts expose many model/AI actions. Capability and behavioral benchmark success must not substitute for matched after-cost uplift. |
| Scientific controls | Keep independence; separate exploration and confirmation | Immutable captures and failure histories are valuable. Missing guarantees, negative stress cases and family-wide economic impossibility are different conclusions. Apply only the documented session clarification. |
| One-off research tools | Stop multiplying near-duplicates by default | The initial inventory has 487 files under tools and extensive event-specific artifacts. For future studies prefer parameterized, bounded, versioned collectors. Preserve historical implementations instead of deleting evidence. |
| Current planning | Correct contradictory scope and dates | `PLANNING.md` had August work labeled active. `INTEGRATIONS_PLAN.md` banned non-major research despite the current public structural-research scope. Both are corrected without widening execution. |
| Native overview | Correct false empty-state claims now; then connect real data | `main.cpp` unconditionally painted “No bot-owned positions” even when its independently fetched ledger state could say positions were tracked. It also always said no verified run existed. The labels now say details/series are not loaded. |
| Native research workflow | Reorient after the underlying status contract is designed | Overview “Research Run” and Research shortcuts launch conservative/regular/aggressive `model-lab` commands. No structural-edge decision view is provided by those shortcuts. This is a source-level workflow gap, not a visual-layout test result. |
| Console dashboard | Fix invalid state handling | `load_artifact_preview` could crash the menu on a valid JSON artifact with non-object runtime. Missing or string-valued environment flags could also be displayed as an actual venue environment. Corrected with direct regression cases. |
| Execution lifecycle | Keep deterministic open/close separation | `build_execution_lifecycle_plan` does not let ordinary risk/capacity blocks prevent closes, while unknown ownership/reconciliation still blocks unsafe operations. Full execution-path review remains open. |
| CLI/native contract | Keep one registration and generated metadata path | `entrypoint.py`, `command_contract.py` and `generate_windows_contract.py` prevent a native-only command interface. Native build and installed-entrypoint verification passed for the label correction. |
| Dependency boundary | Patch the affected optional library | Tornado 6.5.7 was in the optional microstructure chain. 6.5.8 passes socket-free advisory regressions; product server exploitability remains unproved. See the separate dependency report. |

“Retire” means stop initiating that pattern for future work, not delete consumed
contracts, raw evidence, code history, or unrelated user changes. No strategy
has been promoted by these judgments.

## Proposed operator decision view — not yet implemented

The first screen should answer, for each venue and candidate:

- What economic mechanism is proposed, and which cash flows are owned?
- Is the evidence a hypothesis, retrospective observation, prospective survivor
  or account-qualified opportunity? Never collapse these into “accepted.”
- What are the after-cost surplus, break-even cost, required capital, capacity,
  source timestamp and unresolved assumptions? Use unknown when unavailable.
- What would invalidate it, and what is the next independently satisfied retry
  condition? A date displayed by the UI must not itself authorize a request.
- Which actions are read-only, paper/testnet, disabled or separately authorized?
  Keep venue-specific Pause/Stop independent from research controls.

Use a shared read-only backend contract verified against current canonical
registry/audit hashes; do not scrape handoff prose or duplicate economic logic
in C++. Missing, stale or mismatched sources should produce an explicit
unavailable state. Never populate a performance chart with decorative returns,
assume an account is flat, or equate local ledger state with exchange ownership.

This proposal requires actual data wiring and CLI/native parity before it can
replace the existing shortcuts. The two corrected native labels do not pretend
to implement that dashboard. Native visual inspection was not performed:
available automation does not provide native application control, and launching
an account-configured app is unnecessary for checking two static labels.

## Review coverage and verification

Fully inspected the Windows launcher, dashboard helper, native CMake file,
installed entrypoint and integrations plan. Inspected the command taxonomy and
generator, native overview renderer, research shortcuts and compact-status
handling; the complete native source and all terminal-UI interaction paths are
not yet semantically reviewed. The July design notes remain historical evidence
of intent, not fresh confirmation of external research or current implementation.

Dashboard regression results: nine assertions failed on the unchanged helper
(four malformed-runtime crashes and five misleading environment cases), then
all fourteen tests passed after correction. Native C++ compilation/linking and
installed-entrypoint verification passed. No native window was opened; no
account request or order occurred. Generated command metadata was regenerated
by the repository build workflow, not hand-edited.
The final affected-interface pass also included launcher and generated-contract
checks: twenty tests passed. Ruff formatting and checks passed on changed Python.

Next review work: source-complete lifecycle/ownership and status freshness,
causal/selection gates, all active interface flows, then remaining active
modules/configuration/plans. Record findings and keep/change decisions as each
surface is actually read. Mechanical inventory is never semantic completion.
