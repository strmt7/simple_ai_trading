# Dependency compatibility checkpoint

The latest compatible resolution updates polymarket-client 0.17.0 to 0.18.0,
Tornado 6.6 to 6.6.1 and filelock 4.2.9 to 4.2.10. Both declared filelock
minimums are raised, preserving the protection for installations outside uv.
Fresh public package metadata, publisher tags and container manifests are
retained privately with digests in the [result](result.json).

The admitted primary [Tornado release notes](https://www.tornadoweb.org/en/latest/releases/v6.6.1.html)
describe fixes to Windows alternate-stream serving, redirect resource handling,
URL parsing and open redirects. Its native crawler result contains substantive
6.6.1 content, HTTP 200 and the exact source digest. This is source evidence,
not an exploitability claim about this application's deployment.

The SDK wheel was verified against PyPI's published hash. Comparison of the
existing secure-client and cancellation functions found no non-docstring
behavior changes; new wallet-wide listing/cancellation APIs remain unused.
Session-key authorization changes accept the additional gasless key category.
Wallet-wide access must not replace the app's exact owned-order boundary.
Existing single-submission, lost-response and native serialization checks were
included in the candidate lane. Filelock's public imports, native/strict/lease
implementation changes and normalized source delta were retained for review;
this is not an exhaustive security audit of either dependency.

The universal resolver retains compatibility constraints: HftBacktest requires
NumPy below 2.3, the SDK requires websockets below 16, and the Ethereum and Torch
chains constrain their transitives. Thus latest compatible does not mean
forcing every available major version. Publisher action tags, the supported
Python 3.13 patch, developer-tool versions and all four container manifest
digests match the existing pins. CodeQL's bundle release was explicitly
distinguished from the action tag; Tornado's absent GitHub latest-release
record was not treated as absence of an upstream update.

## Verification and coverage

A fresh project-owned Python 3.13.18 environment installed the locked development
groups and GPU, foundation-AI, Polymarket, reporting, microstructure and terminal
test extras. The prior ready environment and shared research provider were not
modified. Package compatibility, universal lock validation, native-library
imports and CLI startup passed. The source-snapshotted focused lane passed with
zero failures, errors or skips; its exact case count and source hashes are in
the result. It covers the SDK boundary, modified Spot journal, owned-intent
dependencies and research-ledger guard. This was not a full-suite run.

This Windows Torch build reports neither CUDA nor HIP, and CUDA availability
is false. No GPU acceleration or performance improvement follows. The mutually
exclusive DirectML profile was not installed or executed in this round. Keep
backend qualification and useful workload timing separate from package currency.

## Alert inventory is not scan coverage

The authenticated paginated API responses list zero currently open Dependabot
alerts, zero open code-scanning alerts and zero open pull requests. These are
inventory counts at capture, not zero vulnerability findings or proof of a
fresh successful scan. This capture does not establish scanner freshness,
quota availability or complete analysis coverage. The user's reported quota
concern remains unresolved; no scan was dispatched to test it.

The agent rules now require explicit separation of quota exhaustion, rate
limits, disabled/stale analysis and successful scans. An explicit exhausted
quota stops dependent work and retries until the provider reset or independently
verified availability change. Unknown quota remains unknown. Independent public
dependency research and local validation may continue without spending scanner
quota. Historical reports retain their original scope and dates.

No release, draft, tag, workflow dispatch, real account/order request, training
run, benchmark, service or persistent worker was created. Stable-release
readiness and qualified stable-edge count remain unestablished and zero,
respectively. The unfinished storage-enrollment prototype is preserved privately
and excluded from this checkpoint; the reproduced database-loss gap remains open.
