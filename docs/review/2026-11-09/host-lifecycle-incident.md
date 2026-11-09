# Persistent host task cleanup

The user's report was substantiated by Windows Task Scheduler, not dismissed on
the basis of the empty Codex automation directory inspected in an older session.
The previous wrap-up did not provide a complete host persistence audit.

## Confirmed evidence and action

- 28 `SimpleAITrading-*` scheduled tasks were registered. 27 were disabled; one
  capture supervisor was enabled. Its exact action was `wscript.exe` with
  `C:\trader\automation\run_capture_supervisor_hidden.vbs`.
- That wrapper invoked `capture_supervisor.ps1`, bound to this trading project
  and its older capture worktrees. The task had a five-minute repetition trigger
  with a 40-day duration and a separate logon trigger. Its last recorded run was
  November 9, 2026 at 06:52:50 Europe/Berlin. The logon trigger had no dated end.
- The supervisor could restart old capture runners. Hidden outer wrappers are
  not evidence that no descendant can display a terminal or outlive development.
  These facts establish unwanted persistence; individual historical popup events
  have not been traced to specific process-creation records.
- The enabled task was disabled and stopped first. Every task's action was
  attributed to the project before removal. All 28 definitions were exported to
  recoverable inert CLIXML backups, then stopped and unregistered. Nothing was
  recreated. No unrelated scheduled task was removed.
- Four external PS1/VBS launchers were moved from `C:\trader\automation` into
  `C:\trader\retired-task-evidence\2026-11-09-0735\disabled-launchers` with
  `.disabled.txt` suffixes. Hashes were retained. Historical logs, captured data
  and state were not deleted or repaired.

Task-definition backups are under
`C:\trader\retired-task-evidence\2026-11-09-0735`. They are evidence, not active
tasks or executable startup files. Do not restore or re-register them without a
new explicit user request. The user requested no ongoing background installation.

## Verified scope

The post-removal scan found zero remaining direct or encoded project task
actions, zero matching Windows services, zero matching checked HKCU/HKLM
Run/RunOnce entries (including checked WOW6432Node locations), zero matching
startup shortcut targets or WMI command consumers, and zero running legacy
Round 21/25/75 capture/supervisor processes. The local Codex automation directory
contained only an internal marker file, not automation directories; that alone
does not establish global app/cloud schedule state. This is a scoped snapshot,
not a claim about every unrelated
startup mechanism on the PC or a recreation of historical terminal behavior.

Eight Python processes were also identified as four pairs of this repository's
stdio MCP adapter, not capture workers. Missing launcher parents alone do not
prove that their client-owned pipes or sessions are stale; they were not killed
by PID or used as proof of capture activity. No detached CocoIndex daemon was
found in the process snapshot. Treat remaining adapter lifetime separately from
the now-removed Windows persistence, and do not silently interfere with clients.

Repository instructions now forbid persistent installation from a generic goal
or tool requirement, require silent bounded helpers with owned lifecycle cleanup,
and require authoritative teardown verification. Routine tool narration is also
disabled by default, as requested separately. Required safety/evidence disclosures
remain. No financial study, credential, account, order or model was accessed.

No index refresh, embedding run or new daemon was started for this exact-scope
cleanup. The edited instructions invalidate the prior whole-source search
fingerprint: explicitly refresh the reusable cache before the next broad
semantic navigation; do not bypass it or treat the old index as current.

The fresh publication audit still reports historical AI placeholder-email and
Codex identities. These previously disclosed shared-history violations remain;
do not force-rewrite history without approval. PR-head refs were checked
separately (24 refs); none was modified by this cleanup.
