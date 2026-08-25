# Agent Workflows

This repository carries the applicable agent tooling from
[`ZMB-UZH/omero-docker-extended`](https://github.com/ZMB-UZH/omero-docker-extended)
at commit `246110b1045cfd4ca318b4e870b5a38d213399b6`. The files are adapted for a
Windows-first Python trading repository; OMERO, Django, and container-specific
skills are intentionally not copied.

## Tooling

| Tool | Pinned version | Repository entry point |
| --- | --- | --- |
| CocoIndex Code | `0.2.37` | `tools/cocoindex_agent_search.py` |
| CodeQL | `v4.37.8` | `.github/workflows/codeql.yml` |
| uv | `0.12.1` | `pyproject.toml` and `uv.lock` |
| Ruff | `0.15.22` | `.github/workflows/ruff.yml` |
| Vulture | `2.16` | `tools/vulture_check.py` and `.github/workflows/vulture.yml` |
| Super-Linter | `v8.7.0` | `.github/workflows/super-linter.yml` |
| Agent skills | ECC `2.0.0` | `.agents/skills/` |
| Karpathy guidelines | commit `2c606141936f1eeef17fa3043a72095b4765b9c2` | `.agents/skills/karpathy-guidelines/` |

## AI Commit Identity Gate

Run the author, committer, and trailer audit as a standalone read command and
inspect its output before invoking `git add`, `git commit`, amend, merge,
cherry-pick, squash, or rebase. Never combine the identity audit and a mutating
Git command in one shell or tool call: a prohibited identity must stop
publication before mutation. Use the command-scoped literal identity from
`docs/AI_COMMIT_IDENTITY.md` for every AI-created commit. If a violation is
already shared, surface its exact commit before publishing another checkpoint
and do not rewrite shared history without explicit approval.

A fresh 2026-07-18 upstream check found CocoIndex Code `0.2.37`, Vulture
`2.16`, and Super-Linter `v8.7.0` still current. The pinned Karpathy commit is
still its upstream `HEAD`; OMERO advanced to the exact commit above and its
applicable ECC `2.0.0` skill changes were reviewed and adapted. Ruff and uv were
updated only after reviewing their release notes and pinned action commits.
CodeQL `v4.37.8` is bound to its official `2026-08-21` tag commit.

CI and release jobs use `uv sync --locked`; `uv.lock` is the cross-platform,
hash-bound dependency record. Dependabot may propose monthly `uv` and GitHub
Actions updates, but it cannot merge them. Accelerator and numerical-library
changes still require host compatibility and model-parity evidence.

The main CI workflow also runs `tools/audit_financial_terminology.py`. It rejects
superseded labels in authored documentation, Windows UI text, publication
generators, and tracked evidence filenames while preserving immutable raw model
responses and backward-compatible serialized identifiers.

CodeQL runs `security-extended` queries against Python and a real manual build
of the native Win32 C++ app. It uploads SARIF on pushes, pull requests, manual
runs, and a weekly schedule; it does not replace dependency or secret scanning.

The imported repo-local skills are `cocoindex-code-search`, `search-first`,
`source-audit`, `ai-regression-testing`, `docs-knowledge-maintainer`, and
`karpathy-guidelines`; `context-budget` is the repository's local context
overlay. Together they enforce testnet safety, reproducible financial evidence,
CLI/Windows parity, secret hygiene, and the single-session rule.

## CocoIndex Contract

The `cocoindex-code-search` skill is mandatory for broad repository
semantic routing. Exact symbols and small result sets still use `rg`; every semantic
candidate must be confirmed with `rg` and direct reads before editing.

```powershell
python tools/cocoindex_agent_search.py mcp-config
python tools/cocoindex_agent_search.py mcp-install
python tools/cocoindex_agent_search.py mcp-smoke
```

The wrapper stores its install, mirror, runtime, and database under
`AGENT_COCOINDEX_HOME`. Its default is an external cache under
`%LOCALAPPDATA%\SimpleAITrading` on Windows or the XDG data root on POSIX. A
cold index is created only by an explicit indexing command; `.cocoindex_code/`
must never be written to the live checkout.

Use `index --allow-dirty-index` only when the worktree snapshot is intentional,
or `search --refresh "<query>"` on a clean tree. MCP search itself never refreshes.
It can therefore return stale active-index text until an explicit
refresh. The mirror includes Git-visible text-decodable files and skips binary
content; semantic results are routing evidence, not correctness evidence.

`mcp-smoke` validates registration and the JSON-RPC handshake without creating
an index. The package benchmark cases and dependency hashes are recorded in
[`reference/cocoindex-code-agent-benchmark-2026-07-11.md`](reference/cocoindex-code-agent-benchmark-2026-07-11.md).

Semantic routing defaults to five results and rejects limits above ten. Refine
queries or add path/language filters before widening. Benchmark artifacts record
both characters and exact UTF-8 output bytes for broad `rg`, semantic routing,
and the focused hybrid path. Bytes are a reproducible context-volume proxy, not
a claim about model-specific token usage.

## Structural Edge Evidence Boundaries

Before starting structural research, read
`docs/model-research/structural-edge-priority-registry-v1.json`. Advance the
highest-ranked hypothesis whose explicit retry trigger is satisfied. If no
trigger is satisfied, do not compensate with more snapshots, broader search, or
a nearby formula; record the blocker and preserve the evidence path instead.

Treat public venue metadata as a candidate filter, not account evidence. In
particular, Polymarket's public `holdingRewardsEnabled` Gamma field can identify
an exact market for a later study, but it does not prove that the account, the
balance's acquisition path, or a particular reward day is eligible. Promotion
requires an authenticated account record explicitly labeled as a holding
reward, a reconciled daily pUSD balance change, and a complete split-to-merge
cash-flow cycle. Do not substitute generic CLOB rewards endpoints documented
with maker, competitiveness, or liquidity-reward fields unless a current
official source explicitly binds them to the holding program. Official relayer
documentation can establish zero direct user gas for successfully relayed split
and merge operations; it does not erase transfer, wrapping, withdrawal,
opportunity, custody, tax, failure, or availability costs.

Do not attribute a Data API `REWARD` activity row to a particular condition or
reward program when its `conditionId` is blank. A public transaction hash and
amount without an exact condition and explicit holding-reward subtype cannot
establish market eligibility, split-origin eligibility, the applicable rate, or
realized holding-reward payout. Top-holder overlap is only a capped diagnostic;
zero overlap within returned rows does not establish absence outside the cap.

Do not reuse the Round 74 USD-M futures commission capture as Binance spot fee
evidence. Spot triangles require signed `GET /api/v3/account/commission`
responses for every exact leg and must include standard, special, tax, buyer or
seller, and discount fields. The existing futures capture uses a different
endpoint and symbol universe. Credential absence is a terminal missing-evidence
state for the current assessment, not permission to substitute a public tier or
invent a discount.

For BFUSD/RWUSD stable-value yield allocation, use the official signed
`rateHistory`, `quota`, and exact USDT/USDC flexible-product-list GETs as a
read-only prequalification stage. A public promotion or documentation example
is not a current account rate or fee. Do not assume one history row represents
one day; prove timestamp cadence before integrating APR. Compare the candidate
with the best exact eligible
same-currency alternative yield, never an assumed zero-yield cash balance, and
subtract entry, redemption, transfer, delay, custody, tax, and opportunity
costs. Do not implement a signed collector while either designated ephemeral
credential variable is absent. Subscription or redemption is a separate funded
stage and requires new explicit authority after the read-only gate passes.

For Binance liquid-staking-token conversion parity, do not sample WBETH or
BNSOL books before same-account signed evidence proves the current conversion
ratio, redemption quota, enabled state, commission, delay, and account
eligibility. An executable discount is not direction-neutral while redemption
leaves unhedged ETH or SOL exposure. Any later screen must price an equal-base
hedge, spot and hedge fees, funding, basis movement, margin and liquidation,
transfers, delay, custody, tax, and realized redemption reconciliation. Do not
implement the signed prequalification while either designated ephemeral
credential variable is absent.

For Polymarket taker tiers, rebates may reduce fees on legitimate organic taker
flow but do not create authority to manufacture volume. Never self-match, wash
trade, generate inauthentic volume, assume a rebate is applied to trade notional,
ignore fee rounding or market-specific minimum order and tick sizes, or amortize
a one-time level-up bonus as persistent edge. Current liquidity-reward pool
figures are configured caps, not payout floors; do not rush a study into the
remaining tail of an allocation window after a protected boundary. Require a
new documented allocation with enough preregistered horizon plus authenticated
queue, fill, cancellation, adverse-selection, orphan-PnL, and realized-payout
evidence.

For Binance quarterly cash-and-carry, preserve spot buyer and seller commission
components separately; the generic conservative helper intentionally collapses
them and is not exact route evidence. Capture fees before refreshing books.
Credential presence is a preflight outside the one-use attempt: if either
documented environment variable is absent, create no journal, make no request,
and do not search chat, shell history, logs, or repository content for the raw
secret. A signed capture must use GET only, a fresh venue clock before every
signed request, zero retries, disabled redirects, a bounded streamed body, and a
durable secret-free journal that records the bounded body hash before response
validation or parsing. Query and retain only the minimum fields needed for
adjudication; do not collect balances or positions during a fee gate.

Do not classify an equal-base long-current/short-next quarterly futures spread
as locked carry. Its pre-expiry payoff is the initial calendar spread minus the
exit spread; at near expiry it is the initial spread minus the still-live far
contract's basis to spot. Require a genuinely fixed terminal payoff identity or
state an explicit term-structure forecast. If that identity fails, record the
terminal mechanism rejection and do not spend requests on a price backtest.

Binance spot maker-rebate work has a separate account-evidence boundary. The
official symbol commission response explicitly excludes both the spot market-
maker rebate rate and the BNB discount effect. Require the same account's
liquidity-program overview, performance, weekly final-rebate result, exact
symbol commission, and spot rebate history; do not substitute a cached public
program table. A rebate rate alone never proves fills, queue priority, adverse
selection, inventory neutrality, capacity, or after-cost profit.

For Binance Prediction Trading, do not trust a generated Markdown
`No authorization required` label without checking the generated transport and
one bounded live preflight. The 2026-08-25 connector Markdown labeled market
list public, while its generated Java transport attached `binanceSignature` and
the live no-key request returned `-2014 API-key format invalid`. That mismatch
is terminal for public research; do not inspect stored credentials or retry with
authentication without explicit authority and a newly frozen contract.
Bounded HTTP preflights must capture the status code and size-limited response
body on the first request, including non-2xx responses. Do not use a success-only
body helper that discards the error payload and forces an identical retry.

## Verification Lanes

Use the narrowest relevant checks while iterating, then the complete suite at a
promotion or release boundary:

Record each passing command, relevant tree state, and artifact. Do not rerun an
unchanged gate merely for reassurance; invalidate it only when code,
configuration, fixtures, dependencies, runtime artifacts, or platform inputs
change. Run the complete required matrix once against the final release tree.
Resolve focused test paths with `rg --files tests` before invoking pytest; do
not infer filenames from module names. Copy the returned path into the command.
When bundling PowerShell verification gates, join dependent commands with
`&&` or run them separately; `;` continues after a failed gate and a later
successful command can mask the nonzero exit status.

```powershell
uv run --group test ruff check .
uv run --group test ruff format --check path/to/changed.py
uv run --with vulture==2.16 python tools/vulture_check.py
uv run --group test python tools/update_readme_badges.py --check
uv run --group test python -m pytest -q
```

For 100% focused branch coverage of a newly isolated module, use the installed
`coverage` executable directly; `pytest-cov` is not a project dependency and
`--cov` arguments are therefore invalid:

```powershell
uv run --group test coverage erase
uv run --group test coverage run --branch --source=simple_ai_trading.module_name -m pytest -q tests/test_module_name.py
uv run --group test coverage report --show-missing --fail-under=100
```

The badges in `README.md` are generated from `.github/readme_badges.json` and
must not be hand-edited.

## One-Use Public Source Runs

For same-expiry Binance option/future parity, matching contract timestamps do
not prove a fixed payoff. First bind the option `realStrikePrice` and quarterly
futures `deliveryPrice` using exact historical records. A historical futures
`deliveryTime` at 00:00 is only a calendar-date marker; do not silently add
eight hours. Historical equality may justify one synchronized displayed-depth
screen, but cannot accept an edge. Ticker-only option prices have no displayed
quantity, and non-synchronous option/futures snapshots are never execution
evidence.

Before the first network request, a bounded one-use research tool must create
or reserve its terminal receipt path and initialize a self-hashed persistent
request journal. Before each request, atomically persist its exact fingerprint.
After every completed response, atomically persist its request timing, status,
decoded payload, canonical payload hash, and raw-response hash before parsing,
validation, or another request. An in-memory ledger alone is insufficient. A
validation, HTTP, rate-limit, serialization, or later economic-evaluation
failure must retain every accumulated response in the rejected terminal
receipt or journal before the exception is propagated. A traceback alone is
not evidence and never permits a retry, replacement market, or adaptive time
shift under a one-attempt contract.

For a bounded historical endpoint whose retention or archive coverage is not
explicitly source-bound, preregister request windows newest-first. The first
response must prove that the required record shape is currently available
before older windows consume the request budget. A valid empty response is a
terminal source-availability failure under that frozen contract: preserve it,
make no later request, and do not adapt the window, symbol, endpoint, or order.
Record the ordering defect in the adjudication so a different future hypothesis
cannot silently repeat it.

Format and test the exact implementation before freezing and publishing it.
Then publish the frozen contract and implementation before the live attempt.
The terminal receipt must bind both source hashes and must state explicitly
whether books, economics, credentials, or orders were reached. Never repair a
failed one-use artifact by resampling; fix the workflow for a separately frozen
hypothesis only when its own nonadaptive contract permits a new attempt.

The 2026-08-25 cross-stablecoin funding full-history v3 run violated this rule:
it fetched the bounded source pages, then failed on a missing historical FX bar,
but its generic failure writer discarded the accumulated payloads. The explicit
v4 recovery was the only deliberate regeneration, removed the unnecessary
daily-conversion assumption, and checkpointed all 20 funding responses before
evaluation. It is terminal and grants no precedent for another regeneration.

Do not treat the presence or absence of returned historical klines as proof of
exchange order state, trading availability, or a delivery cutoff. An expired
Binance USD-M quarterly response on 2026-08-25 returned flat, zero-volume,
zero-trade bars after its normal delivery schedule. A cutoff-dependent study
must bind an authoritative state source or preregister explicitly sourced
trade-count and volume semantics; otherwise it fails closed without salvaging
the surrounding price rows.

After a current AI governance benchmark, use
`tools/build_ai_model_provenance.py` to rescore the exact reports and verify the
Ollama manifest, config, and every referenced blob before atomically writing
`model-provenance.json`. Protected one-shot reports must also carry matching
pre/post-inference digest and metadata hashes plus positive exact-digest GPU
residency; provenance v2 rejects local files that differ from that evidence. Do
not scan or hash model files manually, and do not use this tool with historical
benchmark contracts.

## Transfer Verification

The 2026-07-11 Windows-host transfer check passed the six-skill validator,
CocoIndex contract suite and four-version JSON-RPC handshake, Ruff `0.15.22`,
Vulture `2.16`, yamllint `1.38.0`, markdownlint-cli `0.49.0`, actionlint
`1.7.12`, and Zizmor `1.26.1 --pedantic`. Zizmor reported no findings after
all remote Actions were commit-pinned. The full Super-Linter container remains
the GitHub-hosted integration check represented by its README badge.
