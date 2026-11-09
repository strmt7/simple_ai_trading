# Compact operating guidance with immutable history

The two current handoff files had accumulated 17,028 published lines and
1,099,244 canonical Git bytes. Updating them required resending that content
through the official connector, despite agents normally reading only the top
80 lines. The new entry points contain 46 and 107 lines, totaling 10,149 bytes:
a **99.0767% reduction in handoff bytes**. This measures stored/source payload,
not model tokens, elapsed speed, cost savings or search-quality improvement.

The full originals are unchanged published Git blobs at
`eaad2d1605f6256ba43cd78ab6b96479294b6155`. The
[manifest](../../../archive/agent-history/handoff-2026-11-09.json) records each
original path/revision/blob/SHA-256/byte count/line count and an immutable
permalink. The original source revision remains in ancestry. No historical
result or registry record was edited, no old path/secret content was copied into
another published file, and no source history was rewritten. A separately
retained private copy preserves the original working bytes, including the old
Continuation's mixed line endings. The published-object archive consistently
binds canonical Git bytes.

The compact guidance retains the full active goal, mandatory research/capture
routes, exact retry/account constraints, current evidence and open qualification,
model/GPU conditions, deterministic capital/close controls, whole-application
scope, connector attribution, host lifecycle and final semantic review/bug hunt.
Detailed decisions link their existing source-bound reviews. Non-superseded
study constraints in the original handoffs remain applicable; omission from the
summary is never permission. The revamp plan now also distinguishes the repaired
local native initialization from still-unqualified historical clock/economics.

The bounded `tools.read_handoff_archive` reader validates the manifest and native
revision/path/blob/size/digest/line count before returning text. Exact searches
return at most 20 locations with an explicit total/truncation flag. Text windows
are limited to 120 lines and 16,384 output bytes. ASCII JSON preserves Unicode
without assuming a host code page. Git plumbing has a ten-second bound, hidden
Windows child-job containment, `--no-lazy-fetch` and `--no-replace-objects`. Missing
objects or an unsupported Git fail; the reader does not fetch or substitute
current text. No service, daemon, task, credentials or index registration is used.
This developer tool does not add an application runtime dependency. It reuses
the existing owned-process runner; UTF-8 transport must reconstruct the exact
recorded blob digest before text is admitted. Child environment inheritance is
restricted to the declared system/locale keys.

Review of that shared runner found an uncovered enrollment-failure path: a
suspended child not yet enrolled in its job could survive closing the empty job.
Cleanup now checks and terminates only the original Popen handle before waiting,
even if job closure raises. ExitStack attempts every pipe closure even when
another close fails. Focused failure controls exercise enrollment, timeout, job
close and pipe-close errors. This does not establish the cause of a past host
incident and never grants authority over a PID reported by child output.

Native local reads verify both original blobs and positive exact lookup; an
earlier exact `exerciseHistory` query correctly returned no matches and is not
an absence claim about implementations or other sources. **186 focused checks
pass**, including unchanged agent-workflow contracts, the adapter's source/lifecycle
checks, archive validation, bounded output, missing-object behavior, Unicode and
no-lazy-fetch invocation. Authored
Python passes Ruff. Relative current guidance links are checked separately.

The existing source index was refreshed incrementally, without a cold rebuild or
provider-registration change. Four protocol probes and the stdio search-tool
probe pass. The canonical ten-case semantic benchmark still retrieves none of
its named expected files in the first five results. This remains weak routing,
not proof of absent code; use the documented language/path scope and confirm
live source. The failure is retained. Recorded benchmark times are explicitly
unqualified because concurrent host load was not measured; no speed or recall
improvement is claimed from this compaction.

Source bindings and measurements are retained in [verification](verification.json).
This is preservation/navigation verification, not semantic review of all 17,028
historical lines or the whole repository. Qualified stable edges remain zero;
no network research, market dataset access, fit, benchmark or order occurred.
Abrupt parent death before job enrollment is not newly qualified by these checks;
the failure cleanup fix is not a complete process-supervision audit. The full
goal remains active. Keep the new entry files current and bounded;
put future detail in review artifacts rather than appending transcripts.
