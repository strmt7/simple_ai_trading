# AI Model Selection And Benchmarking

Simple AI Trading treats local AI as a gated risk-review and signal-uplift
component, not as authority to override deterministic risk controls.

## Research Notes

- Ollama's official library publishes the local `qwen3.5:9b` tag used here as
  a structured-output challenger. Availability is not selection evidence:
  https://ollama.com/library/qwen3.5
- A 2026 prediction-market study found an LLM risk layer reduced loss magnitude
  mainly by filtering statistically plausible but semantically brittle event
  relationships. BTC/ETH/SOL five-minute contracts do not provide that diverse
  event-semantic channel, so this project permits no analogous benefit claim
  without prospective paired uplift:
  https://arxiv.org/abs/2602.07048
- FinGPT is an open-source financial LLM project that emphasizes data curation
  and lightweight adaptation for finance workflows, including algorithmic
  trading research: https://arxiv.org/abs/2306.06031
- FinMA-7B is a finance-instruction model from the PIXIU family; its model card
  describes finance NLP and prediction-task tuning: https://huggingface.co/TheFinAI/finma-7b-full
- DragonLLM's LLM Open Finance release describes finance-specialized 8B
  models for risk assessment, sentiment, retrieval-augmented workflows, and
  financial reporting tasks:
  https://huggingface.co/blog/DragonLLM/llm-open-finance-models
- Fin-R1 and Fin-o1 are finance-reasoning 7B/8B model families that are small
  enough for local benchmark experiments, but they still require the same
  structured risk benchmark and AI-vs-ML uplift proof before any autonomous
  effect:
  https://arxiv.org/abs/2503.16252 and https://arxiv.org/abs/2502.08127
- Agentar-Fin-R1 provides Qwen3-based 8B and 32B finance-reasoning candidates.
  Domain tuning is useful candidate evidence, not proof of safe risk decisions:
  https://arxiv.org/abs/2507.16802
- FinHarmBench found that finance-specialized LLMs can be less robust than
  general models on harmful and confusable-benign prompts. Finance tuning never
  bypasses adversarial safety or matched-uplift gates:
  https://aclanthology.org/2026.acl-industry.117/
- The CFA Institute practical LLM guide reports that finance-tuned models such
  as FinMA/FinGPT can be stronger for sentiment and headline classification,
  while broad general models may be stronger for numerical reasoning:
  https://rpc.cfainstitute.org/research/the-automation-ahead-content-series/practical-guide-for-llms-in-the-financial-industry
- Small 1.5B-class models can perform meaningful financial-analysis tasks in
  constrained settings, but they are treated here as fallback/latency options,
  not primary live-risk reviewers:
  https://www.mccormick.northwestern.edu/computer-science/documents/dong-shu-nu-cs-2025-14.pdf
- Kronos is a finance-native time-series foundation model pretrained on more
  than 12 billion K-line records. Its open 102.3M-parameter base model is a
  forecast-feature candidate, not the required multibillion risk reviewer:
  https://arxiv.org/abs/2508.02739
- A June 2026 return-forecasting comparison found that pretrained time-series
  models often ranked well but produced sparse, asset-specific gains over a
  random-walk baseline. Foundation forecasts therefore require rolling-origin,
  random-walk, after-cost, and ablation evidence here:
  https://arxiv.org/abs/2606.27100
- Time-MoE is a genuine 2.4B-parameter time-series MoE candidate, but its
  official implementation still lists covariate support as future work and its
  accelerated examples are CUDA-oriented. It is not treated as AMD/DirectML
  compatible until a pinned live-host worker proves that exact path:
  https://github.com/Time-MoE/Time-MoE
- Chronos-2 is a 120M-parameter multivariate/covariate forecaster. Its native
  cross-series contract is more relevant to BTC/ETH/SOL context than parameter
  count alone, but it is a specialized forecast candidate rather than the
  required multibillion risk reviewer:
  https://github.com/amazon-science/chronos-forecasting
- KTD-Fin shows that identifier/date leakage and passive factor exposure can
  make LLM trading backtests look more intelligent than they are. Historical
  LLM prompts must therefore mask symbol and calendar identity, use normalized
  causal factors only, and attribute uplift against the same-period ML path:
  https://arxiv.org/abs/2605.28359

## Implemented Policy

`simple-ai-trading ai-benchmark` compares installed local Ollama models against
structured finance-risk cases:

- veto failed AI-vs-ML uplift,
- cooldown during unpredictable or low-liquidity regimes,
- approve only clean positive-uplift evidence,
- veto missing or gapped second-level data evidence.

The command writes machine-readable evidence to `data/ai_model_benchmark.json`
by default. Passing this benchmark does not promote a trading model. It only
selects an LLM risk-assessment candidate. Actual AI use remains blocked unless
model-lab deterministic gates pass and accepted symbols include positive
AI-vs-ML uplift evidence.

The Polymarket model command does not contact the AI benchmark or Ollama
provider unless the frozen probability model first improves validation log
loss, untouched-test log loss, and untouched-test Brier score. A failed
prerequisite records `probability_model_gates_failed` and spends no AI tokens.
Zero after-fee proposals likewise record `no_positive_after_fee_proposals`
before benchmark-file, provenance, cache, or provider access.

Uplift evidence uses a common fixed-period return table rather than pairing
trades by list index. The baseline and AI strategies may enter different
trades, but every statistical observation covers the same contiguous market
period. Polymarket group P&L is divided by the exact common initial capital;
missing, duplicate, or unequal-capital equity periods are rejected rather than
filled with zero. Dataset, baseline, AI, local-model, and paired-table SHA-256
values are mandatory, as are finite P&L, return, drawdown, expectancy, profit
factor, trade count, win rate, liquidation, loss-streak, and downside-risk
metrics for both arms. The built-in minimum is 30 periods spanning at least 90
days, a
one-sided sign-test p-value no
greater than 0.05, and a positive 95% moving-block-bootstrap lower mean from at
least 2,000 deterministic resamples. Serialized policy can make these gates
stricter, never weaker.

The benchmark sends Ollama chat requests with `think: false`. Thinking traces
are useful for manual analysis, but they can consume the response budget and
leave empty `message.content`; benchmark decisions must be parseable JSON, not
hidden reasoning.

`ai-review` applies the same fail-closed boundary: one exact top-level object,
exact fields and JSON types, no duplicate keys, finite in-range scores, and no
wrapped prose. A compact report that exceeds the prompt budget is rejected
instead of being cut into incomplete or invalid financial evidence. The v3
artifact hash-binds the exact source report and prompt plus canonical request,
provider-response, capability, latency, and structured-decision evidence. An
approval also binds the installed Ollama weight digest from `/api/tags` and the
canonical `/api/show` metadata; missing or ambiguous provenance blocks the chat
request. These fields follow Ollama's official
[model-list](https://docs.ollama.com/api/tags) and
[model-details](https://docs.ollama.com/api-reference/show-model-details)
contracts.

Historical four-case provider telemetry:
`docs/ai_model_benchmark_legacy_20260710.json`.

The 2026-07-10 four-case host run is retained as historical provider telemetry
only. It is superseded by the current label-free gate and has no model-selection
or AI trading authority.

The earlier 11-case v6 result is revoked. Its prompt included descriptive case
IDs such as `veto_*` and `approve_*`, leaking the expected action. The fresh v7
run excludes case names and expected actions from model input and stores an exact
SHA-256 for every prompt. Qwen3 8B reached `9/11` actions (score `0.886818`, mean
latency `2.95173s`). Fin-R1 8B, Qwen3.5 9B, and Fino1 8B each reached `8/11` and
also failed semantic or risk-range checks. V7 is historical-only because its
response parser accepted wrapped JSON, duplicate keys, type coercion, missing
fields, and clamped out-of-range values. V8 preserves the 11 label-free prompts
but requires one exact top-level object with exact fields, types, enums, finite
ranges, and no duplicate keys. Qwen3 14B is preregistered for one v8 run only
after a fresh confirmation recorder finishes `complete`. No local model is
selected.

That one-shot rule is executable, not advisory. `ai-benchmark` requires the
frozen preregistration, confirmation DuckDB, and exact run ID for Qwen3 14B. It
audits terminal evidence before writing a durable claim. The exact
preregistration digest is code-pinned, and the same preregistration cannot be
claimed against a second confirmation in that ledger. A completed result is
digest-verified on reuse, while started or failed claims block another run.

AI therefore remains enabled-but-unavailable and fail-closed until a fresh model
passes the current gate. No LLM enters the 250 ms action scorer. The veto
evaluator immutably caches the first terminal response in the evidence DuckDB,
including hash-only provider/schema failure envelopes. Its key binds the causal
case, exact request, prompt and response-schema contracts, endpoint policy,
decision thresholds, and current Ollama model digest and metadata. Cache hits
retain the original measured model latency and post-inference `/api/ps` evidence.
Report v5 permits a valid response only when that evidence binds the exact digest
to positive VRAM residency. It also rejects contradictory action/reason pairs:
approval requires only `edge_after_fees`, veto requires an adverse reason, and
cooldown requires `cooldown_required`. CPU-only, missing, malformed, low-
confidence, or late output is stored as an immutable veto. A later action
experiment must remain veto-only and pass the separate 90-day matched-period
uplift contract.

The report separately binds measured inference time, single-GPU-worker queue
delay, and effective decision latency. Queue delay is recomputed from monotonic
case arrivals, so simultaneous BTC, ETH, and SOL reviews cannot each claim an
impossible zero-wait inference.
Before any veto prompt, Polymarket also requires the selected benchmark's sibling
provenance file to bind its exact SHA-256, Ollama manifest, verified multibillion
weight blob, and current installed digest. A changed tag or manifest fails before
generation instead of silently reusing stale governance evidence. Even a future
governance pass would not establish market edge; the separate 90-day paired
after-cost uplift gate remains mandatory.

### Kronos Forecast Evidence (Rejected)

`simple-ai-trading ai-forecast-benchmark` evaluates a separately gated
financial time-series foundation model. The implementation hash-pins the
Kronos source commit, model revision, tokenizer revision, file sizes, SHA-256
digests, and parameter counts before executable source or weights are loaded.
It then runs inference in a supervised child process with bounded deadlines,
planned DirectML worker rotation, one same-seed replay check, and no in-process
retry after a device fault.

The latest benchmark used the 102,310,592-parameter Kronos-base model on
DirectML and 1,536 deterministic BTCUSDT, ETHUSDT, and SOLUSDT decisions from
2024-07-01 through 2025-12-31. Each decision used 480 five-minute bars (40
hours) to predict the next four five-minute bars (20 minutes). The underlying
792,960 one-minute rows per symbol came only from checksum-verified Binance
USD-M archives. Data from 2026 onward remained sealed and was not accessed.

The candidate failed:

| Metric | Kronos-base | Zero-return random walk |
|---|---:|---:|
| Raw mean absolute error | 0.0042225031 | 0.0018330693 |
| Causally calibrated selection MAE | 0.0016923834 | 0.0016922277 |
| Raw information coefficient | -0.053405 | n/a |
| Raw direction accuracy | 51.987% | n/a |

Raw MAE improvement was `-130.3515%`. Earlier-half nonnegative amplitude
calibration assigned scales of `0.00773075`, `0`, and `0` to BTC, ETH, and SOL;
no symbol remained eligible. The later-half UTC-day block-bootstrap 95% interval
for paired MAE uplift was `[-0.0000019992, 0.0000014356]`, with only `42.35%`
positive probability. This is forecast-error evidence, not after-cost P&L,
backtesting, or permission to trade.

The exact latest artifacts are under
[`docs/ai/foundation/latest`](ai/foundation/latest/README.md). The CSV is the
replotting source, the report binds data/model/runtime evidence, the SVG is a
deterministic view, and the manifest hashes every promoted file. Promotion
re-parses all CSV rows, verifies temporal bounds and error identities, rejects
host-local paths or stale files, and publishes the manifest last.

The upstream Kronos paper reports crypto forecasting on Binance spot OHLC data
at five-minute and slower intervals; its investment simulation is a separate
daily, long-only Chinese A-share experiment. Those results do not establish an
edge for second-level leveraged futures day trading. A materially different
hypothesis must beat the causal random-walk gate before further integration:
https://arxiv.org/html/2508.02739 and
https://github.com/shiyu-coder/Kronos

## Why AI Stays Gated

DirectML is not treated as a universal current Windows answer. Microsoft marks
the project as maintenance mode and recommends Windows ML for ONNX inference
on Windows 11 24H2 and newer. This repository retains `torch-directml` only for
training operators that pass a live finite forward/backward/update preflight;
it captures framework warnings and rejects any hidden DirectML-to-CPU operator
fallback. Other hosts may resolve to CUDA, ROCm, MPS, or explicit CPU instead.

Local multibillion-parameter review uses Ollama, whose current Windows and GPU
documentation covers AMD Radeon and additional Vulkan support. Ollama model
execution and Windows ML/ONNX inference are separate runtime contracts; neither
proves that training used a GPU or that a model has financial edge.

GPU headroom is also an explicit fail-closed precondition. Legacy ROCm SMI is
accepted only when exact per-device total and used byte fields reconcile. On
Windows AMD hosts, the preflight reads the driver's 64-bit dedicated-memory
registry value and subtracts current WDDM dedicated usage; duplicate identical
registry views are collapsed, while conflicting totals or malformed counters
remain unknown and block required-GPU AI. It does not use
`Win32_VideoController.AdapterRAM`, whose documented type is only `uint32`.
The CLI freezes the resolved benchmark candidate set and applies this local
provider/model/GPU gate to every candidate before opening a confirmation
database. A failed preflight therefore cannot consume a preregistered one-shot
claim. The same gate runs immediately before an enabled Polymarket AI veto
ablation. The CLI uses an explicit `--enable-ai` or `--disable-ai` override when
present and otherwise inherits the saved runtime setting; the native Windows
control emits its visible state explicitly.

AI review v4 queries Ollama `/api/ps` immediately after inference. It binds the
requested model to one exact SHA-256 weight digest and records model bytes,
VRAM-resident bytes, and their ratio. An unloaded model, zero VRAM bytes,
ambiguous inventory, digest mismatch, malformed response, or provider failure
vetoes the review. The Windows indicator turns green only for this proved
post-inference GPU-resident state. This is execution evidence, not evidence of
forecast skill, profitability, or AI uplift.

- https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows
- https://github.com/microsoft/DirectML
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- https://learn.microsoft.com/windows/ai/new-windows-ml/supported-execution-providers
- https://docs.ollama.com/windows
- https://docs.ollama.com/gpu
- https://docs.ollama.com/api/ps
- https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-videocontroller
- https://rocm.docs.amd.com/projects/amdsmi/en/latest/doxygen/docBin/html/group__tagMemoryQuery.html

Open-source trading systems also argue for skepticism. LEAN and NautilusTrader
emphasize research-to-live parity, while Freqtrade warns that backtests can be
misleading and that dry-run/forward testing is a more reliable bridge to live
behavior. This repo therefore keeps AI as a review/uplift layer behind:

- deterministic risk and liquidity gates,
- second-level data-readiness checks for signed startup,
- accepted non-AI ML backtest evidence,
- paired AI-vs-ML uplift evidence after fees, slippage, drawdown, liquidation,
  and path-quality checks.
