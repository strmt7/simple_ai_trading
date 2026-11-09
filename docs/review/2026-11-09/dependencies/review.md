# Dependency and publication refresh

The November 9 registry/API audit found 17 open Dependabot alerts: two high,
11 medium and four low. The refreshed universal lock resolves their affected
packages above the reported patched versions: Tornado 6.6, filelock 4.2.9,
msgpack 1.2.4 and setuptools 84.2.1. This is a local remediation result;
hosted closure requires publication and GitHub's subsequent evaluation.

Subsequent API readback at 16:57 UTC for published commit
`969b06d912dbe15da742171225a7cff12b2dd09c` confirmed all 17 alerts fixed and zero
open Dependabot alerts. The five superseded dependency PRs (#24, #23, #22, #18,
#17) were closed. The leased historical rewrite and connector-created development
commit were published with exact tree/attribution verification. This closes the
audited dependency alerts, not every hosted code-quality finding or release gate.

The lock upgrades the scientific, SDK, test and build stack together. Current
direct pins include DuckDB 2.0.0, Numba 0.68.1, Torch 2.15.0,
polymarket-client 0.17.0, py-clob-client-v2 1.2.1, huggingface_hub 2.4.0,
Ruff 0.17.4 and pytest 9.2.0. Project-local tools are uv 0.13.7, CMake 4.4.4
and Ninja 1.13.2. Actions are bound to publisher tag commits; container bases,
the Dockerfile frontend and the uv image use verified registry digests.
Crawl4AI's existing isolated provider already uses upstream 0.9.5 with its
separately reviewed portability repairs; no other repository/provider was changed.

“Latest” means latest compatible, not unconditional major-version replacement.
HftBacktest 2.4.4 requires NumPy below 2.3, retaining 2.2.6. The unified SDK
requires websockets below 16, retaining 15.0.1. Python 3.11 uses SciPy 1.17.1;
Python 3.12+ resolves 1.18.1. Torch fixes its compatible CUDA component versions;
the Ethereum SDK chain also constrains several transitive packages. Do not
override these constraints just to match every registry's newest version.
Python 3.13 is the installer's wheel-compatible managed-interpreter target;
this is not proof that every declared interpreter/platform combination was run.

## Runtime compatibility repairs

- DuckDB 2 requires pending execution to complete before registering or removing
  columnar views. The insertion helpers complete results on the original
  connection, preserving the caller's transaction; they do not switch cursors
  and silently split transaction ownership.
- The new unified SDK can resubmit a deposit-wallet transaction after a nonce
  rejection. The adapter now builds one signed payload and performs one lower-
  level submission, preserving single-use preflight and ledger-owned recovery.
  Offline checks cover lost responses and nonce rejection without a second write.
- Python's argparse wrapping differed between 3.12 and 3.13. Generated fallback
  usage is whitespace-normalized before fingerprinting; explicit descriptions
  and command/option semantics remain intact. Both interpreters produce the
  same contract digest, and a real option change still changes it.
- The locked Linux image installs its OpenMP runtime inside the image and puts
  Numba's cache in writable container temporary storage. Native-library imports
  and CLI help pass with no network and a read-only root filesystem plus tmpfs.

## Owned installation and validation

`tools/install_environment.py` requires a bootstrap CPython 3.11+ on Windows.
It installs an exact hash-checked uv wheel without package-index resolution,
downloads a managed interpreter privately, and syncs locked dependencies and
tools into an owned environment per lock/profile/platform. It rejects unknown
or conflicting extras, foreign directories and reparse paths, excludes ambient
package-manager configuration, and never rewrites a ready or incomplete runtime.
Ready environments receive a read-only entrypoint check. All installer children
are bounded, hidden and contained in an owned Windows job. No global packages,
PATH changes, administrator installation, service, startup hook or recurring
task is introduced. Compiler/GPU/system prerequisites remain explicit; a clean
Windows VM and every hardware/OS combination have not been qualified.

The default development environment was left untouched because other processes
were using it. The updated Python 3.13 environment installed 118 packages with
all tested extras except the mutually exclusive DirectML profile. `uv pip check`
and lock validation pass. A fresh automatic install and ready-environment reuse
also succeeded. The native Windows build and CTest presentation check pass.
DirectML's unchanged pin is resolved in the universal lock; its alternate
installation and accelerator execution were not tested this round.

The initial broad integration run produced 9,436 passes, 97 failures and
28 skips. After repairs, rerunning only those failures produced 11 passes and
86 remaining failures. Separate focused checks include 238 SDK/tooling checks,
126 database/cash checks, 81 integration checks and 36 policy/dependency checks;
these lanes overlap and must not be summed as independent coverage. The later
nonce-rejection control adds a second single-submission case. There is no claim
of a green full suite or complete security clearance.

Remaining failures include mutable registry totals/status expectations, broken
historical source bindings, frozen Round 27–29 source-closure checks, existing
Binance receipt/commission fixtures, and research-readiness fixtures. The
Binance identity/recovery/readiness failures also reproduce under the older
environment. The tape/depth synthetic example is rejected for the same minimum-
action reason under both stacks. Original results and financial/source guards
were preserved; they were not re-signed or loosened to turn tests green.
Tests now isolate Decimal context so imports of legacy high-precision research
tools cannot silently alter another test's arithmetic.

Private raw package/GitHub responses, native test reports, install/build logs,
search receipts and recovery material remain under the ignored dependency-audit
directory. The oversized full pydantic-core metadata response remains a failed
capture; a bounded exact-version response separately verifies 2.50.4.

CocoIndex Code 0.2.43 passes its four-version MCP handshake, builds a current
source index and returns retained, live-confirmed search hits. The original
unfiltered ten-case routing benchmark found none of its named expected files
in the first five results; a separate diagnostic using the documented Python/
source filters found three of ten. Both are retained, including failures. This
is limited routing quality, not comprehensive code discovery or a speed claim.
Windows path separators are now normalized before file-identity comparisons.
Keep queries bounded and confirm candidates in live source; improve missed-case
routing prospectively rather than rewriting either benchmark's results.

## Publication and development status

The user replaced the former synthetic-identity rule with official-connector
publication. Empty connector commit
`0236a3e3db2ab6d8bf2033e75e73c403f412488f` supplies the exact attribution
reference for the explicitly authorized historical cleanup. Detailed identity
fields and recovery material remain private. The prepared rewrite checks all
2,400 exported commits: trees, messages, timestamps, topology and non-synthetic
attribution are preserved. It changes 2,384 author fields and 2,384 committer
fields. Eight old signature headers cannot authenticate changed commit objects;
their originals are retained in the verified recovery bundle. Historical result
files keep their original source identifiers and bytes.

The premature `v0.1.0-beta.1` prerelease, its six assets and its public release
tag were deleted at the user's explicit request. API/ref readback found no
published releases or tags. Do not recreate a release or release tag without a
new explicit request. Development and the broader R&D goal remain active;
qualified stable-edge count stays zero. This round used no market/account API,
credentials, order, financial fit or new economic observation.
