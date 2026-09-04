# Dependabot and branch reconciliation

Review base: `4689aa952c4be34835b800576f4f506bf220347d`.
GitHub was checked on September 4, 2026, beginning at 20:18 UTC.
The authenticated GitHub client was used only for repository maintenance;
no exchange credentials, account access or trading actions were used.

## Current decisions

This section records successive snapshots; later bot activity does not change
the retained initial PR-head identities or reopen old validation results.

Update after `9b43812a`: the three fully superseded PRs were closed. Dependabot
itself closed #9 at 20:31:36 UTC after the group configuration refresh and
opened #13 (CodeQL 4.37.9) and #14 (huggingface-hub 1.29.0, Ruff 0.16.5,
build 1.6.0). The agent did not create either PR. Both new batches are being
integrated directly on main with the same identity policy. The SDK migration
remains unresolved despite the bot closing its former PR.

Exact-SHA checks on `9b43812a`: Ruff, Vulture and the Windows Python/native
smoke passed. The main Python job was still running at the last observation.
Super-Linter exposed a second retained raw-data root under `data/**/raw/**`
and a multiline inline formula parsed as a list in CONTINUATION. The raw-data
exclusion now covers both roots; the same formula is on one inline-code line.
No captured data was modified.

The refreshed Dependabot updater also reported incompatible *proposed*
upgrades: NumPy 2.4.6 conflicts with hftbacktest's `<2.3` requirement;
websockets 17.1 conflicts with the SDK's `<16` requirement; and uniform pandas
3.0.5 conflicts with NumPy 2.2.6 on Python 3.14. The current lock resolves.
These coupled migrations remain explicit, not hidden by broad ignore rules,
dependency removal, narrowing supported Python versions or forced installs.

The final refreshed batch passes 46 affected model/foundation/dependency tests,
13 dependency/terminology checks after the raw-root correction, Ruff, and lock
verification. `build` 1.6.0 successfully produced a local wheel under
`.tmp/dependency-review-20260904-wheel`; it was not released or uploaded.
The new CodeQL pin resolves the official 4.37.9 action release, whose stated
change is its default CodeQL bundle. Hosted completion remains separately
observable at the published SHA.

The fresh open Dependabot security-alert response was empty. That is not a
claim of a complete security audit. The prior Tornado 6.5.8 fix remains pinned.
There were four open PRs and exactly four non-main remote branches:

The integrated Python/build changes are setuptools, coverage, diff-cover,
DuckDB, huggingface-hub, Ruff, tqdm, LightGBM, Numba and llvmlite. Comparing
the old PR lock against current main also exposed stale pandas and Tornado
downgrades; those are not copied into the combined lock.

| PR | Head | Decision |
| --- | --- | --- |
| #8 | `5f26eb58006753f4ab8d732ea1eab3d04fafe7af` | Integrate all three SHA-pinned Actions updates on current main. |
| #9 | `0ca83f9ee47314efe18d96ba975793040c7e62aa` | Integrate seven routine updates; retain Polymarket SDK 0.2.0 pending an explicit adapter migration. |
| #10 | `d616fffde9591c5632fa64b3cccbfea3bebee557` | Integrate Numba 0.67.0 and its resolved llvmlite 0.49.0 dependency. |
| #11 | `0465a8cb0e3306a6eb17109d1905e3eba42e84fe` | Integrate LightGBM 4.7.0; do not relabel historical model evidence. |

Fully superseded PRs may close after the replacement commit is verified on
GitHub. PR #9 remains an explicit migration item, not a silently accepted SDK
upgrade. Branches are retained for traceability; none is deleted or rewritten.
Twelve historical PR-head refs were inventoried separately from branch heads.
Fresh contributor metadata still exposes previously reported legacy AI
identities. Shared history is not rewritten; new commits require `AI agent <>`.

## Why the SDK update is not mechanical

`polymarket_live_settlement.py` requires the exact audited version and uses
`SecureClient._create`, internal deployment checks, position-context resolution,
gasless transaction construction and relayer polling. Merely raising the
dependency pin would make the real adapter reject installation; merely raising
the adapter constant would bypass its audit assertion without proof.

The official SDK explicitly permits breaking changes in 0.x minor versions.
Its upgrade is separated from future routine dependency groups. A regression
now binds the declared dependency to the audited adapter version. Complete the
0.6.0 migration by comparing those exact internal call/return and retry paths,
then using socket-free failure, ownership, redemption and restart checks. No
account request is required or authorized for that review.

## Old check failures and narrow corrections

- September 1 Python CI stopped at the terminology audit, before tests.
- Ruff was linting retained official SDK source as authored code, plus an
  unused local variable in a test. The variable is removed; raw evidence is
  excluded without modifying its bytes.
- Super-Linter treated retained official Markdown as authored Markdown.
  Only the research raw-data subtree is newly excluded.
- Ruff 0.16 expands its default rule selection. The previous E4/E7/E9/F
  contract is now explicit, with the workflow version matching the lockfile.
  A single F401 exception preserves the exact consumed NFL adjudicator source.
- Two hash-bound economic adjudications contain historical wording rejected
  by the terminology checker. They are exact-path exclusions, not rewritten
  results. Current authored research prose remains checked.

The new terminology test proves the raw source is not opened, including a
non-UTF-8 fixture, and that adjacent authored research remains in scope.
No lint auto-fix was applied across historical code. An initial diagnostic
command produced excessive output after the Ruff default change; subsequent
lint checks captured output and printed only a bounded summary. An attempted
whitespace check with `core.autocrlf=false` also produced artificial CRLF
diagnostics; the normal repository-aware check passes. Do not override line
normalization in future checks, and always capture diagnostics before output.

## Verification and limits

The combined lock resolves 140 packages and `uv lock --check` passes.
The project builds with setuptools 84.0.0. Ruff passes for src/tools/tests.
LightGBM 4.7.0 CPU training, model serialization and reload produce identical
finite predictions on synthetic data. The affected-domain run covers LightGBM
backend selection, Numba barriers/features, DuckDB warehouse, settlement,
manifest and foundation-loading contracts. Its only failure was the two
historical terminology matches above; only that changed domain is rerun.
Separate dependency-pin/workflow contract tests pass.

These are software compatibility checks, not evidence of strategy uplift,
new predictive accuracy, profitability, all-platform support or GPU validation.
No historical market experiment was rerun. Full hosted status is separate and
must be reported from the exact published commit, not from old PR checks.

## Primary references

- [LightGBM 4.7.0](https://github.com/lightgbm-org/LightGBM/releases/tag/v4.7.0):
  includes a weighted-percentile regression fix and backend changes.
- [Numba 0.67.0](https://github.com/numba/numba/releases/tag/0.67.0).
- [setup-python 7.0.0](https://github.com/actions/setup-python/releases/tag/v7.0.0):
  removed pip-install input is not used by this repository.
- [setup-uv 9.0.0](https://github.com/astral-sh/setup-uv/releases/tag/v9.0.0) and
  [10.0.0](https://github.com/astral-sh/setup-uv/releases/tag/v10.0.0): cache
  defaults changed; this repository explicitly enables cache in existing
  push/pull-request jobs and retains the pinned uv version.
- [Official SDK 0.6.0 metadata](https://pypi.org/project/polymarket-client/0.6.0/):
  supports Python 3.11+, but minor-version API compatibility is not guaranteed.
