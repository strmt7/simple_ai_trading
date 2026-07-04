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
- If the project adds packaged ONNX inference, evaluate WinML first and keep DirectML as the compatibility baseline.

Commands:

```powershell
simple-ai-trading compute
simple-ai-trading compute --backend directml
simple-ai-trading compute --backend cpu
simple-ai-trading ai
simple-ai-trading ai-review --report data/model_lab/model_lab_report.json
```

CPU-only mode is allowed. When selected or when GPU probing fails:

- AI features are disabled.
- Training, retraining, tuning, and backtest scoring continue on CPU.
- CLI and Windows app warn that the run is slower.
- Structured local AI review cannot approve a model-lab artifact until the AI
  capability preflight passes again.

References:

- https://microsoft.github.io/DirectML/
- https://learn.microsoft.com/en-us/windows/ai/directml/dml-get-started
- https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
