# GPU Backend Policy

Simple AI Trading defaults to DirectML on Windows.

Reasoning:

- The app targets Windows operators first.
- DirectML supports AMD, NVIDIA, and Intel DirectX 12 GPUs through one backend.
- The repo already uses PyTorch for model training/scoring, and `torch-directml` provides the shortest working path for those operations.
- CUDA and ROCm remain supported as advanced explicit choices, but they are not the Windows default.

Current caveat:

- `torch-directml` is the selected backend for this repo's PyTorch training/scoring path.
- ONNX Runtime's DirectML execution provider remains useful for inference research, but its docs now describe it as sustained engineering and point Windows ONNX deployments toward WinML for future provider selection.
- Windows ML now publishes dynamically selected execution providers such as AMD MIGraphX, NVIDIA TensorRT RTX, Intel OpenVINO, and Qualcomm QNN for Windows 11 24H2+ systems. If the project adds packaged ONNX inference, evaluate WinML first and keep DirectML as the broad compatibility baseline.

Commands:

```powershell
simple-ai-trading compute
simple-ai-trading compute --backend directml
simple-ai-trading compute --backend cpu
simple-ai-trading ai
simple-ai-trading ai-review --report data/model_lab/model_lab_report.json
simple-ai-trading ai-forecast-benchmark --backend directml --model-size base
```

## Financial Foundation Forecast Gate

The optional Kronos benchmark is process-isolated because the live AMD host
exposed intermittent DirectML device failures during long inference sequences.
The benchmark never retries a failed DirectML request in the same interpreter.
It closes that worker, starts a fresh verified worker, checks immutable engine
identity, and replays the same batch once with the same seed. Planned rotation
after 20 batches bounds long-lived device state. A second fault in one batch,
an identity change, a deadline, non-finite output, shape mismatch, or seeded
output mismatch fails the run.

The retained 1,536-observation host run started 27 workers: one initial worker,
25 planned rotations, and one successful fault replacement. It recorded zero
in-process retries and exact same-seed output equality. Total supervised
inference time was 116.982 seconds across 512 batches. This proves that the
runtime can execute and recover on this AMD/DirectML host; it does not prove
predictive or trading value. The candidate failed the random-walk forecast gate
and remains research-only. See
[`docs/ai/foundation/latest`](ai/foundation/latest/README.md).

For new packaged ONNX deployment work, Windows ML is the preferred evaluation
path because Microsoft describes ONNX Runtime DirectML as sustained engineering.
The current PyTorch/Kronos experiment remains on `torch-directml` only because
that is the verified upstream model execution path; changing providers requires
a new parity and numerical benchmark.

CPU-only mode is allowed. When selected or when GPU probing fails:

- AI features are disabled.
- Training, retraining, tuning, base/advanced feature matrix generation,
  external-signal scoring, probability-temperature calibration, threshold
  calibration, backtest scoring, model-lab candidate generation, robust
  validation, and backtest-panel row generation use GPU-first `auto` when no
  explicit backend is passed. They continue on CPU only when the user selects
  CPU or every GPU probe fails, and artifacts record the requested backend,
  resolved backend, device, and fallback reason where that workflow has a
  persisted backend-evidence contract.
- Probability calibration reports include `calibration_backend_requested`,
  `calibration_backend_kind`, `calibration_backend_device`, and
  `calibration_backend_reason`. Promotion evidence should treat those fields as
  the proof of whether the calibration scan ran on DirectML/CUDA/ROCm/MPS or
  fell back to CPU.
- Signed live startup and `risk --live --model` require promoted
  `TrainedModel` artifacts to prove bounded multi-candidate model selection.
  When the resolved runtime backend is DirectML/CUDA/ROCm/MPS, the same gate
  also requires non-CPU training and probability-calibration backend evidence
  before orders can be submitted.
- Hybrid-candidate backtest scoring keeps Lorentzian nearest-neighbor,
  rational-quadratic kernel, and technical-confluence expert math on the tensor
  backend when the backend supports the required operations.
- Feature generation uses tensor prefix/window operations on the resolved
  backend for the 13 base features, then advanced rows expand from those
  backend-built base rows. If a backend lacks a required tensor operation, the
  workflow falls back to the original CPU feature builder rather than producing
  partial or divergent rows.
- CLI and Windows app warn that the run is slower.
- Structured local AI review cannot approve a model-lab artifact until the AI
  capability preflight passes again.

References:

- https://microsoft.github.io/DirectML/
- https://learn.microsoft.com/en-us/windows/ai/directml/dml-get-started
- https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers
