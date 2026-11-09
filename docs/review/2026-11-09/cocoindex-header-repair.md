# Source-index restart repair

The first incremental restart failed on the owned legacy global settings file.
The pinned CLI had originally written its stock comment with a CP1252 em dash;
the current UTF-8 runtime rejected that byte. Its error wrapper then obscured
the UnicodeDecodeError with a TypeError. An earlier running daemon had masked
the restart defect. No package upgrade or other-repository change was needed.

The adapter now migrates only that exact non-UTF-8 stock first-line comment.
Already valid UTF-8 is unchanged. Unknown/mixed encodings, foreign roots and
symlinks reject rather than guessing. It preserves every configuration-body
byte, checks concurrent edits, uses an atomic replacement and retains a metadata
receipt without settings values. Failed replacement preserves the original and
cleans its temporary file. A file lock serializes this owned migration.

The actual completed receipt is under the external cache's `receipts/` directory
with prefix `settings-header-b338ddba65292b7524aa71e610552e274d18b285d308bb42fbb1442f6b442a9f`.
The original file SHA256 was
`b338ddba65292b7524aa71e610552e274d18b285d308bb42fbb1442f6b442a9f`;
the replacement SHA256 was
`749708ed46ef09fbe35c6b8ff79e74589a4932ded0062e2e1c11165ed3b14ffc`.
No configuration values are copied into that receipt. The package remains
`cocoindex-code[full]==0.2.37`, with the same local embedding model and cache.

An actual incremental index and five-result tools/Python semantic query then
completed, with live-source confirmation of each candidate. Query receipts are
external and retain snapshot identity. That snapshot preceded the later source
and handoff edits; it is not asserted current after those edits. Final refresh
and the required unchanged ten-case workflow benchmark are recorded separately
below. No cold rebuild or unfiltered control repeat is needed.

The complete affected adapter/Windows-source-scope/owned-job suite passed 170
checks, including nine migration cases. Two earlier runs completed assertions
but exited with a shared `pytest-current` teardown PermissionError; those are
not clean passes. A unique per-run `--basetemp` gave a clean exit without
modifying another task's temporary files. This repair does not establish better
semantic recall, financial edge, training accuracy or benchmark speed.

The helper lifecycle remains bounded and uses owned handles/jobs. No recurring
task, service, logon hook or detached long-term worker was installed. Historical
capture data and the inert launcher/task-definition backups remain preserved.

## Closing verification

The final explicit incremental refresh indexed 2,161 files / 57,398 chunks,
reusing physical storage key `55a118c547f096d674ef96ec0e06a4e2`. The precommit
content snapshot was `718abe18db6d9eb4d75a868cf8b5df34`. The unchanged ten-case
source-filtered benchmark completed with actual per-query receipts: three of
ten top-five expected-path hits, unchanged from the earlier source control.
Seven misses remain; this is working search with limited measured recall, not
a semantic-quality improvement. Unfiltered controls were not repeated.

The external result is
`C:\Users\strat\AppData\Local\SimpleAITrading\agent-cocoindex-code\receipts\routing-source-2026-11-09-header-repair.json`,
SHA256 `829c08db424b598bb959e3e22122c0ffd77af0819ea36509854093da3a37ed9b`.
It records `reused_current_index=true` and `timing_qualified=false`. Output byte
counts are not measured model tokens; recorded durations are not a clean speed
benchmark. No timing or GPU improvement is claimed. Post-refresh edits are
confined to excluded review reports; the included source content is unchanged.

Closing host scans found zero matching project scheduled tasks, zero legacy
capture workers and zero matching external `run-daemon` processes. Under the
same runtime lock, the native handshake was absent and stale PID/socket metadata
was removed; the daemon log was preserved. No PID authorized process termination.
Ten client-connected Python adapter processes were separately observed; these
are application MCP infrastructure, not installed recurring capture tasks.
Their ownership/lifetime was not assumed stale, and they were not killed or
used as evidence that all background processes on the shared PC were removed.
The local Codex automation inventory had zero `automation.toml` files; this
does not establish universal app/cloud schedule absence. Pause the current goal
at handoff and leave unrelated clients and applications untouched.

Fresh GitHub inspection reported zero open critical Dependabot alerts. Other
severities/scanners remain outside that statement. The anonymous-contributor
inventory and 2,386 local history records still expose previously disclosed
shared-history AI identity violations. No coauthor trailers were found; 24 remote
PR-head refs were counted separately. This checkpoint uses `AI agent <>` for
both identities; no shared-history rewrite or PR-head modification is authorized.
