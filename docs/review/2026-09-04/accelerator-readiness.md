# Accelerator readiness: measured capability, not a hardware assumption

Observed September 4, 2026, from the clean `bbc1e943` research runtime. This is
a local capability probe using one synthetic tree update, not market training
or evidence of financial uplift.

| Surface | Observed state |
| --- | --- |
| Windows device | AMD Radeon RX 9070 XT; driver 32.0.31041.1004 |
| OpenCL device | Platform 0, device 0, gfx1201, board AMD Radeon RX 9070 XT |
| OpenCL memory | 17,095,983,104 bytes reported by the driver |
| OpenCL driver | 3679.0 (PAL,LC) |
| Current LightGBM | 4.7.0; installed binary lacks the GPU tree learner |
| Real auto-backend probe | GPU update failed with USE_GPU build requirement; resolved CPU |
| Current environment PyTorch | Not installed |
| Current environment ONNX DirectML | Not installed |

The loaded library was
`.venv/Lib/site-packages/lightgbm/bin/lib_lightgbm.dll`, SHA-256
`7e366d2e49cd061aac3ab21676b2f99b0c7a758dc3e888d4b23812af1b7d301c`.
The existing `lightgbm_backend_parameters("auto", 20260904,
pin_opencl_device=True)` made the real synthetic probe. Package/driver presence
was not substituted for successful execution. No GPU speedup was measured or
claimed. The separate `C:/trader/.venv-lightgbm` reports version 4.6.0; it was
not substituted for the project's 4.7.0 lock or used for new market fits.

## Bounded next step before large training

Visual Studio 18 Community C++ tooling is available, and the installed AMD
ROCm directory is version 7.1. The inspected ROCm tree did not contain the
OpenCL headers/import library searched for; Boost was not located in the
inspected build-dependency locations. This is not an exhaustive machine-wide
absence claim. No driver, SDK or system-wide package was installed.

The [official LightGBM installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html)
specifies OpenCL/Boost/CMake/MSVC for Windows GPU builds and directs Windows
users to OpenCL rather than its Linux-only ROCm implementation. It also notes
that OpenCL accelerates histogram construction, not every training operation.

For large new LightGBM fits, prepare a separate reproducible 4.7.0 OpenCL build
with explicit source and dependency identities. Probe its real device, then
compare end-to-end CPU and GPU time on the *same* predeclared workload, seed,
hyperparameters, data type, precision and thread limits. Include transfer,
dataset construction, warm-up and peak-memory costs; retain all repetitions.
Check finite outputs, predictive metrics and train/save/reload behavior before
using that backend. Use FP64 accumulation where the existing model contract
requires it. Do not silently replace a frozen CPU model with a GPU fit.

For neural training, use a separately verified compatible AMD runtime and
check actual tensor allocation, training update and serialization. An ONNX
inference provider does not establish PyTorch training capability. Do not
downgrade the project or force conflicting optional extras just to obtain a
GPU label. Any CPU fallback must be recorded in the training artifact.

The 66-row Decimal/source-schema economic review does not warrant a GPU or a
new training run. Acceleration is for justified computational workloads, not
for increasing the number of weak financial hypotheses tested.
