# AI Model Selection And Benchmarking

Simple AI Trading treats local AI as a gated risk-review and signal-uplift
component, not as authority to override deterministic risk controls.

## Research Notes

- FinGPT is an open-source financial LLM project that emphasizes data curation
  and lightweight adaptation for finance workflows, including algorithmic
  trading research: https://arxiv.org/abs/2306.06031
- FinMA-7B is a finance-instruction model from the PIXIU family; its model card
  describes finance NLP and prediction-task tuning: https://huggingface.co/TheFinAI/finma-7b-full
- DragonLLM's LLM Open Finance release describes finance-specialized 8B
  models for risk assessment, sentiment, retrieval-augmented workflows, and
  financial reporting tasks:
  https://huggingface.co/blog/DragonLLM/llm-open-finance-models
- The CFA Institute practical LLM guide reports that finance-tuned models such
  as FinMA/FinGPT can be stronger for sentiment and headline classification,
  while broad general models may be stronger for numerical reasoning:
  https://rpc.cfainstitute.org/research/the-automation-ahead-content-series/practical-guide-for-llms-in-the-financial-industry
- Small 1.5B-class models can perform meaningful financial-analysis tasks in
  constrained settings, but they are treated here as fallback/latency options,
  not primary live-risk reviewers:
  https://www.mccormick.northwestern.edu/computer-science/documents/dong-shu-nu-cs-2025-14.pdf

## Implemented Policy

`simple-ai-trading ai-benchmark` compares installed local Ollama models against
structured finance-risk cases:

- veto failed AI-vs-ML uplift,
- cooldown during unpredictable or low-liquidity regimes,
- approve only clean positive-uplift evidence,
- veto missing or gapped second-level data evidence.

The command writes machine-readable evidence to `data/ai_model_benchmark.json`
by default. Passing this benchmark does not promote a trading model. It only
selects an AI reviewer candidate. Actual AI use remains blocked unless
model-lab deterministic gates pass and accepted symbols include positive
AI-vs-ML uplift evidence.

The benchmark sends Ollama chat requests with `think: false`. Thinking traces
are useful for manual analysis, but they can consume the response budget and
leave empty `message.content`; benchmark decisions must be parseable JSON, not
hidden reasoning.

Latest committed local benchmark summary:
`docs/ai_model_benchmark_latest.json`.

Current local priority order favors `qwen3:8b` as the installed structured
risk-review baseline, `deepseek-r1:8b` as a reasoning second opinion, and
smaller models only when they pass the same benchmark. Finance-specialized
DragonLLM, FinGPT, and FinMA models remain preferred candidates for future
local serving tests, especially for risk text, sentiment, and headline
classification, but they still must pass the same benchmark and uplift gates
before they can affect autonomous trading.

## Why AI Stays Gated

DirectML remains the Windows-first acceleration layer because Microsoft
documents `torch-directml` as the Windows PyTorch path and DirectML supports
DirectX 12 GPUs across AMD, Intel, NVIDIA, and Qualcomm. The repo still records
backend/fallback evidence because the DirectML project is in maintenance mode
and local AI behavior must be reproducible on the operator's host:

- https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows
- https://github.com/microsoft/DirectML

Open-source trading systems also argue for skepticism. LEAN and NautilusTrader
emphasize research-to-live parity, while Freqtrade warns that backtests can be
misleading and that dry-run/forward testing is a more reliable bridge to live
behavior. This repo therefore keeps AI as a review/uplift layer behind:

- deterministic risk and liquidity gates,
- second-level data-readiness checks for signed startup,
- accepted non-AI ML backtest evidence,
- paired AI-vs-ML uplift evidence after fees, slippage, drawdown, liquidation,
  and path-quality checks.
