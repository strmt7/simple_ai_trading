# Compute Portability Contract

The application defaults to `auto`, not to an operating system or GPU vendor.
Runtime discovery supports modern `torch.accelerator`, CUDA, ROCm, Intel XPU,
Apple MPS, legacy DirectML, and an always-available CPU reference.

## Guarantees

- Models serialize backend-independent numeric parameters and preprocessing.
- CPU float64 is the reference implementation for artifact parity.
- `auto` may use CPU when no accelerator passes discovery.
- A pinned backend may not silently fall back. Resolution or operation failure
  aborts before a training, calibration, scoring, or backtest artifact is used.
- Multi-device CUDA/ROCm selection uses measured free memory when available,
  then the framework's current device. Other runtimes use their reported current
  device. `SIMPLE_AI_TRADING_DEVICE_INDEX` overrides either path only after a
  range check.
- Backend, device, selection method, package versions, and fallback reason are
  evidence, not model features. They must not alter a prediction.
- Tensor and tree runtimes are discovered independently. A working PyTorch or
  DirectML device does not imply that the installed LightGBM library has CUDA
  or OpenCL support. LightGBM acceleration is accepted only after one real tree
  update succeeds for the selected target and OpenCL device override.

## Installation Profiles

```powershell
# Current PyTorch runtime for native backends
python -m pip install -e .[gpu]

# Version-pinned compatibility path for DirectML
python -m pip install -e .[directml]

# Portable CPU reference
python -m pip install -e .
```

ROCm and other driver-specific wheels come from the official PyTorch selector.
The project deliberately does not encode a universal wheel URL, driver version,
device index, or GPU architecture. Installation must end with a real capability
probe:

```powershell
simple-ai-trading compute --backend auto
simple-ai-trading ai
```

An explicit probe such as `compute --backend directml` is saved only when the
request is actually satisfied. CPU operation remains supported, but local AI is
blocked because its separate model-residency and VRAM gates require measured GPU
execution.

## DirectML Status

`torch-directml` remains useful on compatible Windows and WSL AMD, Intel, and
NVIDIA devices, but it is a version-pinned public-preview adapter. Microsoft now
describes DirectML as sustained engineering and directs new Windows ONNX
deployments toward Windows ML, which dynamically selects execution providers.
No Windows ML/ONNX path is claimed until export, provider discovery, numerical
parity, and fault-isolation tests exist in this repository.

The retained Kronos benchmark used bounded DirectML worker processes because a
device fault must not freeze the app. It demonstrated recovery on one AMD host,
but failed its forecast-value gate and has no trading authority. See
[`docs/ai/foundation/latest`](ai/foundation/latest/README.md).

References:

- https://docs.pytorch.org/docs/stable/accelerator.html
- https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html
- https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html
- https://learn.microsoft.com/en-us/windows/ai/directml/dml-get-started
- https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview
- https://onnxruntime.ai/docs/execution-providers/
