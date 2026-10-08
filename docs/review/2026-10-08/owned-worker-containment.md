# Foundation worker: owned process boundary, not PID trust

Baseline: `6afa80911e17119f5c6416edbd60c3a2193ad4ba`.
This repairs a demonstrated operational risk on the shared workstation. It is
not market evidence, a fitted model, a timing benchmark or enterprise readiness.

## Reproduced cause and repair

`FoundationWorkerSupervisor.start` previously accepted any positive reported
runtime PID. `stop` could call `os.kill` on that PID even after the original
launcher exited. No kernel ownership or creation identity was checked. A safe
mock of an exited launcher PID 101/report PID 202 records a SIGTERM to 202;
no real process was signaled in that reproduction. PID reuse or false worker
metadata could target a different task. A venv launcher/runtime distinction is
not sufficient authority to control a process.

`windows_owned_job.py` uses Windows SDK ABI declarations and explicit API
signatures. It creates an unnamed non-inheritable Job with kill-on-close and no
breakaway flags. The original CPython Popen process is created with
`CREATE_SUSPENDED`. Its existing owned process handle anchors identity; a
read-only thread snapshot finds its unique initial thread, and the opened thread
handle's owner and original process liveness are checked before use. Assignment
to the Job precedes `ResumeThread`. Missing/unsupported handles, assignment or
thread invariants reject startup—there is no uncontained execution fallback.

The foundation client checks a ready PID against OS Job membership, read-only.
Booleans, string IDs and DWORD-overflow IDs cannot qualify. Shutdown closes the
owned Job handle; it never signals a reported PID. Descendants inherit the Job,
so a Windows venv launcher exiting does not orphan its runtime. Startup exceptions
now clean up before propagating, including failed context-manager entry.
Non-Windows ready identity must equal the original launched child PID; no
arbitrary runtime PID is signaled there either. Non-Windows tree containment
has not been implemented or qualified by these Windows checks.

## Native evidence and limits

57 affected checks pass; 29 are new. Five real Windows scenarios establish:

1. Child and actual interpreter grandchild belong to the Job and exit after
   Job close, while a separate test-owned sibling remains alive.
2. A worker claiming that sibling's PID is rejected and cleaned up; sibling
   remains alive.
3. A startup timeout cleans up the contained worker.
4. A valid own runtime starts successfully and exits on Stop.
5. An actual controller `os._exit(17)` closes its last Job handle and terminates
   the worker, verified against an already-opened original process handle.

Only explicitly test-created processes were controlled. No user task was
selected, killed, reprioritized or otherwise modified. The small stdlib control
fixtures use a direct base interpreter when they require one process handle;
production model workers retain their original venv command. No model weights,
training, prediction batch, GPU load, benchmark timing, account or order ran.

An initial native assertion incorrectly required a nonzero Job-termination exit
code. Windows returned zero even though the process exited. The corrected proof
uses original handle/liveness and descendant waits, not an assumed exit code.
This did not change production behavior to accommodate a test. The earlier
control fixture used a venv wrapper for its uncontained sibling; its cleanup
could leave that test-owned interpreter until its 60-second self-exit. Future
control fixtures now launch their stdlib-only sibling directly. No orphan
control PID was selected for an unverified termination.

ABI/runtime evidence: Windows, CPython 3.12.10, eight-byte pointers; Basic Limit,
Extended Limit and Thread Entry sizes 64/144/28 bytes. Installed Microsoft SDK
10.0.26100.0 `winnt.h`, `TlHelp32.h`, `processthreadsapi.h`, `winbase.h` bind
the layout/constants/prototypes. Existing September 5 Microsoft Job Objects
tool extractions bind the kill-on-last-handle/no-breakaway mechanics; they are
retained tool text, not origin HTTP bytes. No alias/source refresh was made.
These native tests qualify this host/runtime, not every supported OS release.
The controller-crash proof begins after successful enrollment. Death between
process creation and assignment can leave an unassigned suspended child; this
two-phase launch does not prove atomic creation-in-job or startup-orphan cleanup.
The suspended model has not been permitted to execute, but that resource/recovery
gap still belongs in the independent supervisor's final qualification.

This boundary is not a security sandbox: inherited environment/credentials,
worker output/resource limits, blocking request-pipe writes, independent trading
gateway containment, useful-progress monitoring, persisted recovery/rearm and
model quarantine remain separate unfinished requirements. Terminating a worker
does not cancel venue orders. The execution/emergency gateway must remain outside
strategy jobs; no trading process has been enrolled by this repair.

## Research/process discipline

Source reuse confirmed the inverse-collateral accounting question is already
covered and its legal sign/applicability gate remains unresolved. No duplicate
inverse collector, box-spread workaround or known-negative sports screen was
built. No new financial observation or acceptance is recorded; counts stay
202 observations/65 hypotheses/37 scoped mechanisms/zero qualified stable edges.

Initial registry routing printed oversized repeated rows. `AGENTS.md` now
requires keys/counts/field lengths, bounded triage and paged remaining full-row
review; triage still cannot authorize capture. Process control must use owned
OS handles, never worker-reported PID authority. Source-audit, Python-patterns,
regression and documentation skills shaped those ownership/evidence boundaries.
Old source extractions, financial results, captures and receipts remain intact.
The goal stays active, with no automation created or edited.
