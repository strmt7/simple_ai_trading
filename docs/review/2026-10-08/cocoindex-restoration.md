# CocoIndex restoration

User requirement: restore actual semantic routing, without broad-search bypasses.
This is agent tooling, not a qualified trading edge or a training result.

## Repairs

- Resolve the installed Windows interpreter under `Scripts/python.exe`.
- Give automated commands an empty input pipe. Windows NUL may still report
  itself as a terminal; it does not reliably suppress interactive initialization.
- Exclude market data, artifacts, historical review/model-research/archive trees
  and binary formats before content reads. Index source, configuration and active
  instructions only; benchmark broad search uses that same mirror.
- Retain actual CLI/MCP search output, query arguments, snapshot binding and
  candidate live/mirror SHA-256 comparisons in the external cache. Reject stale
  sources before searching and changes detected after searching.
- Refuse to overwrite foreign or unbound MCP registrations.
- Contain newly started Windows daemon trees in the existing owned no-breakaway
  job implementation. Assign suspended children before resuming; failure rejects
  startup. Cleanup closes the original job, not a daemon-reported PID.
- Capture commands, including index and MCP probes, use the same suspended-child
  job boundary on Windows. Timeout cleanup cannot leave launcher descendants
  running or use their reported identifiers as permission to terminate them.
- Capture UTF-8 explicitly in isolated child environments; this workstation's
  preferred encoding is CP1252. Non-ASCII output must round-trip without silent
  replacement, and the parent environment must not change.
- Preserve literal Windows CLI arguments using the pinned interpreter and
  Click's supported `windows_expand_args=False` entry point. The original native
  CLI expanded `src/simple_ai_trading/**` into hundreds of backslash filenames:
  the first became a non-matching filter and the rest polluted the query. Native
  client and read-only SQLite diagnostics returned source matches, ruling out the
  initial database/filter hypothesis. Do not patch vendor SQL or upgrade the pin
  to conceal this root cause.
- Explicit `index --reuse-active-cache` retains prior source bytes and updates
  only changed mirror files at the same working path. Storage identity and source
  fingerprint are separate, so CocoIndex's memoized embeddings can survive source
  edits without admitting stale results. A pending/failed update blocks searches;
  source, mirror and manifest must match. Missing tracked paths retain their
  fingerprint semantics. This cache reuse still requires an actual index run.

## Boundaries

Pinned package remains `cocoindex-code[full]==0.2.37`. Installation, mirrors,
databases, local model download and receipts remain under the trading external
cache. Embeddings use the package's local Snowflake XS default, with two-thread
OpenMP/MKL settings and usage tracking disabled. This does not qualify financial
models or imply GPU acceleration. Other repositories' runtimes, indexes and
unrelated processes are not changed. Never query or serialize secrets.

## Verification

Functional CLI and MCP semantic searching are restored, including literal native
path filtering. Retrieval recall is not fully qualified. The first actual query
returned five source files with live/mirror byte agreement. The initial MCP search
failed because its explicit environment omitted Windows home/runtime variables;
initialize/tool listing had not exercised that dependency. Environment overlays
now preserve inherited OS/runtime variables without changing the parent. The
complete search passed, and initialize/listing passed four supported protocols:
2024-11-05, 2025-03-26, 2025-06-18 and 2025-11-25.

The final explicit incremental index contains 57,337 chunks in 2,158 files.
Source fingerprint: `0a3d46f0a50745c6de9ed5806ef60ce3`. Physical storage remains
`55a118c547f096d674ef96ec0e06a4e2`; this is not a source freshness assertion.
161 affected tests and Ruff passed. No unchanged financial suite, model fit or
market capture was repeated for these agent-tool changes.

Retained external routing artifacts, relative to the trading cache's `receipts/`:

| Run | Frozen expected-path hits | Scope |
| --- | --- | --- |
| `routing-2026-10-08-53193cf4.json` | 0/10 | Original unfiltered questions; old reports dominated |
| `routing-python-2026-10-08.json` | 0/10 | Same questions/expectations, Python-only control |
| `routing-source-2026-10-08.json` | 3/10 | Same questions/expectations, source-path control after argument repair |

Both control fixtures were frozen before their respective runs and retained here.
The final source
control changed only declared scope, not questions, regexes or expected paths;
it reused the current index and retained ten individual raw search receipts.
The three hits were outage recovery, action policy and GPU resolution. Seven
misses remain. Source-only scope excludes some expected tests/docs/tools, so this
is neither an equivalent whole-corpus result nor evidence of reliable recall.
Smaller output is not sufficient proof of usefulness, speed or token savings.
The original failed path pilot remains in
`search-29fcc35ff3184d359a27f67c36f1b938.json`; it must not be overwritten.

The final real stdio MCP query used `path: src/simple_ai_trading/**` and
`lang: [python]` for incumbent inventory/resize/flatten cash accounting. Its raw
receipt is `search-fa035cbc7d91440993bef25800825ca9.json`, output SHA-256
`dd1620822e412d5a89727774a945a97d88d44922dbfe7476657f3aec23a9901d`.
The whole indexed source matched the current source fingerprint; all five
candidate SHA-256 comparisons passed and each returned span was read in live
source: `funding_cash_inventory.py:159-174`, `stateful_cash_replay.py:86-94`,
`stateful_cash_labels.py:69-77`, `completion_economics.py:27-45` and
`inventory_cash_actions.py:156-181`. This is actual retrieval evidence, not a
registration marker or a financial qualification.

The first legacy daemon predates the new owned-job boundary and remains reused.
Do not claim it has retroactively acquired containment or kill it by reported
PID. Newly launched children have the original job handle boundary; unrelated
processes and other repositories' provider registrations remain untouched.

Timing on this shared workstation is provisional unless concurrent CPU, GPU,
memory and disk headroom is monitored. Functional routing and source identity
checks are distinct from timing claims. Preserve old July results unchanged.

## Publication identity and remaining work

The fresh anonymous contributor audit surfaced historical placeholder AI emails:
`ai-agent@example.invalid` (18), `agent@localhost` (15), `agent@local.invalid`
(7), `ai.agent@noreply.local` (1), plus historical Codex/hyphenated identities.
Shared commit `c431b76fdf8bcd1f3b6c63b4ecbfa16cfb49ebcb` has `AI agent` with
`agent@local.invalid` in both identity fields. This is a policy violation, not
fixed by correctly naming the next commit. It was disclosed before publication;
do not force-rewrite shared history without explicit approval. Remote PR-head
refs were inspected separately and none was modified by this restoration.

Continue using real current-snapshot semantic searches and live confirmation.
Improve retrieval recall prospectively with bounded, independently declared
controls; do not change consumed benchmark expectations or read protected market
evidence to make an agent-routing metric pass. No stable market edge was promoted.
