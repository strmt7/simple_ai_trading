# Working local OpenCL runtime; performance qualification remains provisional

LightGBM 4.7.0 now executes real OpenCL training on the AMD Radeon RX 9070 XT.
The application backend resolver also selected `opencl:0:0:AMD Radeon RX 9070 XT`
with FP64 accumulation. The local wheel and loaded DLL are bound in
[gpu-runtime-manifest.json](gpu-runtime-manifest.json). This enables LightGBM
trees, not PyTorch/neural training. No historical model or market outcome changed.

## Use without losing the GPU build on dependency sync

From the repository root:

```powershell
& tools/run_lightgbm_opencl.ps1 -PythonArgs @('-m', 'simple_ai_trading', '--help')
```

Pass the intended Python arguments in the `PythonArgs` array. The launcher
checks the durable local wheel hash, uses a cached `uv --with` overlay, checks
the loaded library hash, and executes a real OpenCL update before the supplied
command. It does not alter the locked base environment or override a command's
explicit CPU/frozen backend choices. The wheel is stored outside temporary
build directories at `C:/trader/runtimes/lightgbm-4.7.0-opencl/`. It is a local
artifact, not a binary distributed through Git. Missing/changed wheels,
library mismatches, absent devices and CPU fallback stop this launcher.

An ordinary base `uv run` still uses the CPU wheel. Use the verified launcher
for justified new accelerated work; do not assume every task automatically
inherits the GPU overlay. New builds need their own hash/capability review.

## What was measured, and what was not

All datasets below are generated synthetic data, not financial observations.
The first and large benchmarks used three paired CPU/GPU repetitions, FP64
GPU accumulation, and exact within-fit save/reload checks. Their data,
parameters, tolerances and trial order were specified before execution.

| Workload | Observed CPU/GPU total time | Scope |
| --- | --- | --- |
| 4,096 rows, 32 features, 50 rounds, 8 threads | 0.021 / 0.184 s median | GPU slower; maximum prediction tolerance also failed |
| 65,536 rows, same shape/settings | 0.133 / 0.276 s median | GPU slower; paired tolerances passed |
| 262,144 rows, same shape/settings | 0.427 / 0.509 s median | GPU slower; paired tolerances passed |
| 1,048,576 rows, 128 features, 200 rounds, 8 threads | 8.928 / 4.085 s median | Observed ratio 2.186; paired tolerances passed |
| 4,194,304 rows, 128 features, 200 rounds | CPU16: 36.929/38.650 s; GPU16: 13.069/12.887 s | Two repetitions; stage/own-process telemetry profile |

**These are provisional observations, not isolated-machine speed claims.**
After these measurements, the user disclosed other concurrent tasks and
required uncontended benchmarking. Background CPU/GPU/disk load was not
recorded. The large observed gain motivates careful confirmation; it does not
prove a clean causal speedup, a hardware optimum, or improved model accuracy.
The successful training, finite predictions, and reload checks remain valid.

The 4.19-million-row profile measured a 2 GiB feature matrix. Single-fit GPU
engine busy-time means were about 23–25 percent; 8 CPU threads used about
7.5 CPU-core equivalents, versus about 15 for 16 threads. Sampled dedicated
GPU allocations were around 699 MB maximum in those runs. These Windows
process counters are neither shader occupancy nor total-device memory peaks.
They indicate substantial CPU-side work, not a reason to manufacture features
or extra data solely to raise utilization.

The bounded concurrency sweep planned four complete jobs per configuration
at 1×16, 2×8 and 4×4 worker/thread settings. The serial configuration finished;
the next configuration was interrupted on the user's instruction. Only the
verified task-owned worker was terminated; the controller recorded failure.
No concurrent configuration is qualified. Preserve every partial journal and
do not silently resume/overwrite it. See `gpu-concurrency-result/result.json`.

## Shared-workstation policy

Never stop, pause, reprioritize or change affinity of the user's other tasks.
Before performance-timing benchmarks, collect a passive workload-budgeted headroom window with
`tools/measure_benchmark_background_load.ps1` and a fresh output path. It
defaults to reserving 50 CPU percentage points, 40 GPU percentage points and
12 GiB RAM for our expected work, with combined CPU/GPU ceilings of 90%,
4 GiB remaining RAM and an average disk queue ceiling of 4. Three consecutive
pressure samples defer the run; brief spikes and normal desktop activity do
not. Adjust the explicit reservations for the actual intended workload,
especially concurrent workers. Missing/invalid counters still fail closed.

These are practical screening budgets, not a guarantee: aggregate utilization
below 100% can still hide shared-core, cache, memory-bandwidth or SSD contention.
The user explicitly rejected requiring an almost-idle PC. The earlier strict
15-second receipt and its exact `gpu-load-gate-v1.ps1` source are preserved,
but those old thresholds are not the continuing policy.

Deferral is only for benchmark timing validity. Ordinary model training and
R&D may continue with bounded resources alongside the user's other tasks;
do not require an idle window for ordinary work. Never modify those tasks.

That tool is **preflight-only**: its receipt is not a lease on future idle
capacity. During-run background-load monitoring, with our own process-tree
usage separated, is still required before another performance-qualified run.
If competing heavy work starts, stop only our verified benchmark processes,
retain the interruption and invalidate the affected timing comparison. No
new large benchmark is authorized merely because a prior preflight was quiet.

## Rebuild provenance and the app-picker incident

[gpu-build-evidence/manifest.json](gpu-build-evidence/manifest.json) binds the
retained failed/successful logs and exact successful machine-local commands.
The integrated build first failed on CMake's removed old-policy compatibility,
then Boost auto-detected `msvc-6.0`, used a nonexistent compiler setup path,
and produced conflicting 32/64-bit outputs. Four app-picker dialogs were
reported during that attempt; the exact dialog-launching command is unproved.

The successful route used the existing downloaded sources without editing
upstream source files:

1. Boost 1.74 commit `a7090e8ce184501cfc9e80afa6cafb5bfd3b371c`, explicit
   `msvc-14.3`, compiler toolset `14.42.34433`, x64 only, eight build jobs and
   an exact setup script; automatic site/user/project configuration bypassed.
2. Khronos OpenCL headers and loader commits recorded in the runtime manifest;
   a shared-loader import library built with the same compiler. The installed
   system OpenCL runtime supplies the actual AMD device.
3. The downloaded LightGBM 4.7.0 source distribution, `USE_GPU=ON`, explicit
   Boost include/library paths, OpenCL include/import paths, and CMake's
   FindBoost compatibility policy. The exact `uv build --wheel` invocation
   is retained in `gpu-build-evidence/build-wheel-explicit.cmd`.

These are actual machine-local recipes with prerequisite paths, not a tested
clean-machine installer. Invoke future compiler helpers hidden/noninteractive
and retain new logs; do not restore the failed autodetection prototype.

## Larger financial datasets

More computation is useful only when additional data changes the financial
learning problem constructively. Add independent events and relevant assets,
regimes, executions and costs where authorized; never duplicate observations
or include future books/labels to fill GPU memory. Keep decision-time features,
purged event/time splits, immutable holdouts and search accounting. Compare
sample-efficiency learning curves and after-cost economic decisions, not just
classification accuracy. No actual market-training dataset was enlarged in
these synthetic runtime experiments. Accuracy and profitability remain unproved.
