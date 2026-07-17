# Round 68: Bounded AI context

**Status:** implementation and contract validation only. No model inference or market backtest was run, so this round makes no AI-uplift, profitability, ROI, drawdown, testnet, or live-trading claim. Performance tables and graphs are unchanged.

## Change

The live shadow reviewer used a 4,096-token Ollama context but previously admitted up to 16 KiB of structured evidence before prompt framing. Provider-side truncation could therefore remove instructions or evidence while still consuming GPU time. The `live-ai-entry-risk-review-v2` contract now:

- keeps only the causal structured case and a shorter, non-authoritative review instruction;
- binds the v2 prompt contract into the v2 case identity and audit hash chain;
- rejects message content above 3,584 UTF-8 bytes before constructing an HTTP request;
- reserves context headroom for the chat template and completion;
- reduces the structured completion ceiling from 180 to 128 tokens; and
- rejects telemetry when reported prompt plus completion tokens exceed 4,096.

The preflight is deterministic and runs before Ollama, so an oversized case consumes no provider tokens. Existing digest, GPU-residency, strict-JSON, finite-number, action, reason-code, risk-bound, timeout, asynchronous-exit, and append-only audit controls remain unchanged.

Ollama's official [context-length documentation](https://docs.ollama.com/context-length) defines context length as the maximum tokens available to the model and notes that larger contexts increase memory use. Its [FAQ](https://docs.ollama.com/faq#how-can-i-specify-the-context-window-size) documents `num_ctx` for API requests. The implementation therefore keeps the smaller GPU-efficient context and bounds evidence instead of increasing VRAM demand. This is an operational integrity control, not evidence that the language model improves trading decisions.

The frozen Qwen3 14B v9 governance benchmark, its preregistered cases, and all Round 9 action/ridge/MLP hashes are untouched. That benchmark still cannot run until the independent 15-hour recorder has finalized and the preregistered ML admission sequence permits it.

## Verification

- All 80 live AI-assist, AI-uplift, AI-runtime, and parity tests pass together.
- An oversized case is rejected before the mocked network transport is called.
- Provider requests carry `num_ctx=4096`, `num_predict=128`, deterministic temperature/seed controls, and the exact approved model digest.
- Individually valid prompt and output counts that exceed the combined context fail closed.
- Ruff and `git diff --check` pass on the changed implementation.

No graph was regenerated because no economic or AI-uplift experiment occurred.
