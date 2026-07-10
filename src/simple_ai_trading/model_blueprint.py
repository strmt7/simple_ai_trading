"""Research-backed model and training blueprint for future model work.

The blueprint is deliberately structured data.  It lets the CLI, Windows app
contract, docs, and tests share the same model roadmap instead of relying on
free-form notes that future agents can easily miss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json


RISK_LEVELS = ("conservative", "regular", "aggressive")


@dataclass(frozen=True)
class RiskTrainingBlueprint:
    """How one risk level should consume the model stack."""

    name: str
    primary_focus: str
    acceptance_bias: str
    promoted_families: tuple[str, ...]
    veto_rules: tuple[str, ...]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ModelFamilyBlueprint:
    """One model family or AI component in the research roadmap."""

    family: str
    role: str
    status: str
    training_target: str
    gpu_path: str
    risk_levels: tuple[str, ...]
    execution_authority: str
    validation_gates: tuple[str, ...]
    sources: tuple[str, ...]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingLaneBlueprint:
    """A promotion lane that turns research inspiration into guarded work."""

    lane: str
    families: tuple[str, ...]
    research_takeaway: str
    next_build_step: str
    promotion_gates: tuple[str, ...]
    runtime_limit: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchSourceBlueprint:
    """Source-catalog entry with explicit usage policy."""

    source_id: str
    label: str
    url: str
    source_type: str
    applied_to: tuple[str, ...]
    usage_policy: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


MODEL_FAMILIES: tuple[ModelFamilyBlueprint, ...] = (
    ModelFamilyBlueprint(
        family="advanced_logistic",
        role="Primary supervised baseline for long/short/flat probabilities.",
        status="implemented",
        training_target=(
            "Forward-return and triple-barrier labels generated from the same "
            "execution assumptions used by backtests."
        ),
        gpu_path="PyTorch DirectML when available, CPU fallback for tests.",
        risk_levels=RISK_LEVELS,
        execution_authority="candidate_signal_after_objective_gates",
        validation_gates=(
            "purged walk-forward",
            "objective PnL/drawdown/trade-count gates",
            "temporal robustness",
            "path quality",
            "portfolio stress",
        ),
        sources=(
            "https://github.com/microsoft/qlib",
            "https://www.freqtrade.io/en/stable/freqai/",
            "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253",
            "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551",
        ),
    ),
    ModelFamilyBlueprint(
        family="adaptive_hybrid_model_zoo",
        role=(
            "Original hybrid overlay that blends supervised probabilities with "
            "Lorentzian neighbors, rational-quadratic kernels, and technical confluence."
        ),
        status="implemented",
        training_target="Improve the accepted supervised candidate on a separate chronological selection set.",
        gpu_path="Shares base-model scoring backend; stdlib expert scoring.",
        risk_levels=RISK_LEVELS,
        execution_authority="only_if_it_preserves_or_improves_objective_score",
        validation_gates=(
            "separate selection window",
            "final holdout replay",
            "full-sample replay",
            "objective acceptance",
        ),
        sources=(
            "https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/",
            "https://www.tradingview.com/script/AWNvbPRM-Nadaraya-Watson-Rational-Quadratic-Kernel-Non-Repainting/",
            "https://www.tradingview.com/support/solutions/43000614331-technical-ratings/",
        ),
    ),
    ModelFamilyBlueprint(
        family="regime_gate",
        role="Detect when market state is unsuitable for a candidate's learned edge.",
        status="implemented_evidence",
        training_target="Volatility, trend, reversal, autocorrelation, volume, and regime-specific expectancy evidence.",
        gpu_path="CPU deterministic math today; HMM/GMM candidates can use optional accelerated libraries later.",
        risk_levels=RISK_LEVELS,
        execution_authority="veto_or_cooldown_only",
        validation_gates=(
            "accepted-window share by regime",
            "dominant-regime concentration warning",
            "regime-specific expectancy",
        ),
        sources=(
            "https://www.twosigma.com/articles/a-machine-learning-approach-to-regime-modeling/",
            "https://developers.lseg.com/en/article-catalog/article/market-regime-detection",
            "https://macrosynergy.com/research/classifying-market-regimes/",
        ),
    ),
    ModelFamilyBlueprint(
        family="meta_label_gate",
        role="Learn and apply whether accepted primary signals should be taken, downsized, or skipped.",
        status="implemented_execution_gate",
        training_target="Simulated trade outcomes after fees, spread, slippage, stop, take-profit, and time barrier.",
        gpu_path="CPU deterministic policy today; future classifier may use DirectML or LightGBM OpenCL.",
        risk_levels=RISK_LEVELS,
        execution_authority="pre_entry_skip_or_downsize_only",
        validation_gates=(
            "precision target by objective",
            "trade-count sufficiency",
            "out-of-sample replay",
            "disabled policy preserves legacy behavior",
            "malformed enabled policy fails closed",
        ),
        sources=(
            "https://www.quantresearch.org/Innovations.htm",
            "https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/",
            "https://mlfinpy.readthedocs.io/en/latest/Labelling.html",
        ),
    ),
    ModelFamilyBlueprint(
        family="lightgbm_opencl",
        role="GPU-accelerated tabular model candidate for engineered technical, liquidity, and regime features.",
        status="research_candidate",
        training_target="Barrier-adjusted return direction and calibrated probability with repeated-seed checks.",
        gpu_path="LightGBM OpenCL for AMD/NVIDIA when installed; CPU fallback for parity tests.",
        risk_levels=RISK_LEVELS,
        execution_authority="none_until_native_vs_exported_prediction_parity_passes",
        validation_gates=(
            "repeated seed variance",
            "GPU capability artifact",
            "ONNX/native inference parity",
            "same model-lab objective gates as implemented models",
        ),
        sources=(
            "https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html",
            "https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html",
            "https://lightgbm.readthedocs.io/en/latest/GPU-Targets.html",
        ),
    ),
    ModelFamilyBlueprint(
        family="patch_transformer",
        role="DirectML sequence model for longer multi-symbol context without forcing an LLM to choose orders.",
        status="research_candidate",
        training_target="Quantile or class probabilities over barrier-adjusted returns and volatility/tail-risk horizons.",
        gpu_path="PyTorch DirectML first on Windows; compact model size for 8 GB VRAM.",
        risk_levels=RISK_LEVELS,
        execution_authority="forecast_feature_only_until_walk_forward_and_path_gates_pass",
        validation_gates=(
            "no-lookahead feature logs",
            "purged walk-forward",
            "calibration drift",
            "temporal robustness",
            "ablation versus tabular baseline",
        ),
        sources=(
            "https://arxiv.org/abs/2211.14730",
            "https://github.com/yuqinie98/patchtst",
            "https://arxiv.org/abs/1912.09363",
        ),
    ),
    ModelFamilyBlueprint(
        family="foundation_forecaster",
        role="Optional AI time-series forecast provider for probabilistic advisory features.",
        status="research_candidate",
        training_target="Timestamped return distribution, volatility, and tail-risk forecasts used as logged features.",
        gpu_path="Local GPU provider required for AI-enabled mode; CPU mode disables this family.",
        risk_levels=RISK_LEVELS,
        execution_authority="advisory_features_only_until_ai_uplift_gate_passes",
        validation_gates=(
            "forecast timestamp audit",
            "feature ablation",
            "AI-vs-ML holdout uplift",
            "same-cost backtest replay",
            "cannot override risk controls",
        ),
        sources=(
            "https://arxiv.org/abs/2508.02739",
            "https://github.com/shiyu-coder/Kronos",
            "https://arxiv.org/abs/2606.27100",
            "https://arxiv.org/abs/2403.07815",
            "https://github.com/amazon-science/chronos-forecasting",
            "https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/",
            "https://arxiv.org/abs/2310.10688",
            "https://arxiv.org/abs/2402.02592",
            "https://arxiv.org/abs/2510.13654",
        ),
    ),
    ModelFamilyBlueprint(
        family="ai_uplift_gate",
        role=(
            "Deterministic governance gate proving any AI-assisted candidate beats "
            "the non-AI ML baseline on holdout evidence before approval."
        ),
        status="implemented_governance_gate",
        training_target="No training; compares accepted AI-assisted backtest metrics against the non-AI baseline.",
        gpu_path="No GPU compute required; consumes artifacts produced by GPU-backed model runs.",
        risk_levels=RISK_LEVELS,
        execution_authority="approval_gate_only",
        validation_gates=(
            "minimum multibillion model parameter evidence",
            "common dataset plus baseline, AI, model, and paired-table SHA-256 bindings",
            "contiguous fixed-period returns instead of index-paired trade lists",
            "minimum 30 periods spanning 90 days and exact one-sided sign-test p-value at or below 0.05",
            "positive 95% moving-block-bootstrap lower mean from at least 2000 resamples",
            "AI realized PnL and expectancy exceed baseline",
            "AI drawdown does not exceed baseline",
            "minimum AI trade count",
            "missing evidence fails closed",
        ),
        sources=(
            "https://arxiv.org/abs/2303.17564",
            "https://arxiv.org/abs/2306.06031",
            "https://arxiv.org/abs/2403.07815",
            "https://arxiv.org/abs/2605.28359",
        ),
    ),
    ModelFamilyBlueprint(
        family="cross_asset_graph_sequence",
        role="Multi-asset graph/sequence model for correlation shocks, rank forecasts, and portfolio-context features.",
        status="research_candidate",
        training_target=(
            "No-lookahead cross-symbol return ranks, volatility regimes, dynamic correlation graphs, "
            "and diversification stress features."
        ),
        gpu_path="Compact PyTorch DirectML graph/sequence candidate; CPU fallback for contract tests.",
        risk_levels=RISK_LEVELS,
        execution_authority="portfolio_context_features_only_until_ablation_passes",
        validation_gates=(
            "point-in-time cross-symbol joins",
            "graph sparsity stability",
            "ablation versus independent-symbol baseline",
            "portfolio CVaR and correlation-cluster stress",
        ),
        sources=(
            "https://arxiv.org/html/2502.06707v2",
            "https://arxiv.org/abs/2402.18959",
            "https://arxiv.org/abs/1912.09363",
            "https://arxiv.org/abs/2606.27670",
        ),
    ),
    ModelFamilyBlueprint(
        family="tape_depth_gross_forecaster",
        role=(
            "Long-history local/peer trade-flow and coarse-depth forecaster with "
            "explicit gross-return labels and uncertainty bounds."
        ),
        status="implemented_evidence",
        training_target=(
            "Future real trade-reference return after a latency delay; never an "
            "executable fill or after-cost PnL label."
        ),
        gpu_path="LightGBM OpenCL on AMD/NVIDIA; bounded CPU fallback for parity tests.",
        risk_levels=RISK_LEVELS,
        execution_authority="none_gross_forecast_feature_only",
        validation_gates=(
            "checksummed official trade/depth manifests",
            "causal as-of depth join with age mask",
            "purged train/tune/calibration/evaluation split",
            "predictor profile selected independently from execution risk tolerance",
            "ordered core/tape-derived/cross-asset/full feature ablations",
            "point-in-time peer joins with exact source-manifest boundaries",
            "source-bound screening lock and untouched winner-only terminal confirmation",
            "timestamp-defined multi-year rolling folds with non-overlapping evaluations",
            "exact serialized-model replay against hash-bound float64 targets",
            "direction and regression baselines",
            "exact BBO or live shadow required for any execution claim",
        ),
        sources=(
            "https://arxiv.org/abs/1011.6402",
            "https://arxiv.org/abs/1803.06917",
            "https://arxiv.org/abs/2112.13213",
            "https://arxiv.org/abs/2606.27670",
            "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253",
            "https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html",
            "https://github.com/binance/binance-public-data",
        ),
    ),
    ModelFamilyBlueprint(
        family="deep_lob_microstructure",
        role="Limit-order-book model family for spread, depth, queue, and short-horizon mid-price dynamics.",
        status="blocked_until_depth_store",
        training_target="Top-of-book/depth tensors, microprice, imbalance, quote volatility, and fill-risk labels.",
        gpu_path="PyTorch DirectML for compact CNN/LSTM/attention candidates; CPU smoke model for tests.",
        risk_levels=("regular", "aggressive"),
        execution_authority="none_until_order_book_simulator_exists",
        validation_gates=(
            "symbol-specific spread/depth evidence",
            "latency/fill uncertainty simulation",
            "walk-forward by quote time",
            "market-impact stress",
        ),
        sources=(
            "https://arxiv.org/abs/1808.03668",
            "https://arxiv.org/abs/2405.18938",
            "https://arxiv.org/html/2403.09267v1",
            "https://arxiv.org/abs/2511.12563",
            (
                "https://developers.binance.com/docs/derivatives/usds-margined-futures/"
                "websocket-market-streams/How-to-manage-a-local-order-book-correctly"
            ),
        ),
    ),
    ModelFamilyBlueprint(
        family="rl_meta_controller",
        role="Sandboxed policy research for allocation, cooldown length, and execution style.",
        status="sandbox_only",
        training_target="Portfolio-level reward with transaction costs, liquidity, drawdown, and risk aversion embedded.",
        gpu_path="DirectML PyTorch if added; deterministic CPU baseline required before acceleration.",
        risk_levels=("aggressive",),
        execution_authority="none_for_raw_buy_sell_decisions",
        validation_gates=(
            "realistic simulator required",
            "deterministic baselines",
            "repeated seeds",
            "out-of-sample regime tests",
            "human-visible research artifact",
        ),
        sources=(
            "https://arxiv.org/abs/2011.09607",
            "https://github.com/AI4Finance-Foundation/FinRL",
            "https://nautilustrader.io/docs/latest/concepts/backtesting/",
        ),
    ),
    ModelFamilyBlueprint(
        family="ai_risk_reviewer",
        role="Structured local AI review of model-lab artifacts and risk evidence.",
        status="implemented_advisory",
        training_target="No training inside the repo; consumes compact deterministic evidence and returns schema-checked JSON.",
        gpu_path="Local GPU-capable model required unless explicitly disabled in settings.",
        risk_levels=RISK_LEVELS,
        execution_authority="veto_or_review_only",
        validation_gates=(
            "deterministic gates must already pass",
            "AI-vs-ML uplift evidence must pass when AI is enabled",
            "minimum multibillion local model check",
            "schema validation",
            "no credential access",
            "cannot approve failed model-lab outcome",
        ),
        sources=(
            "https://arxiv.org/abs/2503.16252",
            "https://arxiv.org/abs/2508.00828",
            "https://arxiv.org/abs/2510.11695",
            "https://huggingface.co/SUFE-AIFLM-Lab/Fin-R1",
            "https://airc.nist.gov/airmf-resources/airmf/5-sec-core/",
            "https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows",
            "https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html",
        ),
    ),
)


TRAINING_LANES: tuple[TrainingLaneBlueprint, ...] = (
    TrainingLaneBlueprint(
        lane="supervised_signal_baselines",
        families=("advanced_logistic", "lightgbm_opencl"),
        research_takeaway=(
            "Keep a strong tabular baseline before adding heavier sequence or "
            "foundation models; require calibrated probabilities and repeated "
            "candidate evidence after trading costs."
        ),
        next_build_step=(
            "Compare the implemented advanced logistic model against an optional "
            "LightGBM OpenCL candidate on the same purged walk-forward folds."
        ),
        promotion_gates=(
            "same feature signature and no-lookahead rows",
            "probability calibration artifact",
            "repeated-seed variance report",
            "objective and path-quality gates after execution stress",
        ),
        runtime_limit="Can propose candidate probabilities only after objective gates pass.",
    ),
    TrainingLaneBlueprint(
        lane="hybrid_confirmation",
        families=("adaptive_hybrid_model_zoo",),
        research_takeaway=(
            "Community indicator families are useful as independent confirmation "
            "signals, not as copied entry rules."
        ),
        next_build_step=(
            "Add ablation reports showing whether Lorentzian, kernel, and "
            "technical-confluence experts improve net performance per objective."
        ),
        promotion_gates=(
            "separate chronological selection window",
            "holdout replay cannot degrade accepted objective score",
            "community-source no-copy audit",
            "model artifact records expert weights and prototype counts",
        ),
        runtime_limit="Can only refine an already accepted base signal.",
    ),
    TrainingLaneBlueprint(
        lane="regime_and_meta_gates",
        families=("regime_gate", "meta_label_gate"),
        research_takeaway=(
            "The safer way to use ML in non-stationary markets is to learn when "
            "not to trade, when to downsize, and when to cool down."
        ),
        next_build_step=(
            "Promote regime-conditioned meta-label precision reports by symbol, "
            "risk level, and execution-stress scenario."
        ),
        promotion_gates=(
            "skip/downsize/take policy cannot create entries",
            "malformed enabled policy fails closed",
            "per-regime precision and expectancy evidence",
            "backtest and live entry paths consume the same policy fields",
        ),
        runtime_limit="Veto, cooldown, skip, or downsize only.",
    ),
    TrainingLaneBlueprint(
        lane="sequence_forecast_features",
        families=("patch_transformer", "foundation_forecaster", "cross_asset_graph_sequence"),
        research_takeaway=(
            "Patch-based, foundation, and cross-asset graph/sequence models are "
            "promising for longer context, but forecasts must first be logged features."
        ),
        next_build_step=(
            "Run the implemented timestamped forecast-feature store through the "
            "ordered tabular cross-asset ablation; graph/sequence forecasts remain "
            "blocked until they independently beat that baseline."
        ),
        promotion_gates=(
            "forecast timestamp audit proves no lookahead",
            "ablation beats tabular baseline after fees and slippage",
            "cross-symbol joins are point-in-time and portfolio-aware",
            "DirectML capability artifact for GPU runs",
            "CPU-only mode disables AI approval",
        ),
        runtime_limit="Advisory or feature-provider role until ablation and path gates pass.",
    ),
    TrainingLaneBlueprint(
        lane="microstructure_depth_research",
        families=("tape_depth_gross_forecaster", "deep_lob_microstructure"),
        research_takeaway=(
            "Long trade/depth history can support gross forecast research, but "
            "executable order-book models still need actual quote/depth tensors "
            "and a fill simulator; candle or percentage-band proxies are not enough."
        ),
        next_build_step=(
            "Benchmark the gross forecaster across the full official corpus, then "
            "add current L2 captures, quote update-rate evidence, microprice, and "
            "queue/fill uncertainty labels without relabeling coarse depth as BBO."
        ),
        promotion_gates=(
            "symbol-specific spread and depth data exists",
            "latency and fill-uncertainty simulation exists",
            "quote-time walk-forward split",
            "market-impact stress passes per objective",
        ),
        runtime_limit="Blocked from executable authority until depth simulator exists.",
    ),
    TrainingLaneBlueprint(
        lane="sandbox_meta_control",
        families=("rl_meta_controller",),
        research_takeaway=(
            "Financial RL belongs in portfolio/execution meta-control research "
            "before it can safely affect autonomous operation."
        ),
        next_build_step=(
            "Create a deterministic simulator benchmark after microstructure "
            "storage exists; compare RL against simple cooldown and sizing baselines."
        ),
        promotion_gates=(
            "deterministic baseline beats naive policy",
            "repeated-seed stability",
            "out-of-sample regime stress",
            "human-visible sandbox artifact",
        ),
        runtime_limit="No direct orders and no raw buy/sell decisions.",
    ),
    TrainingLaneBlueprint(
        lane="governance_and_ai_review",
        families=("ai_risk_reviewer", "ai_uplift_gate"),
        research_takeaway=(
            "LLM-style AI should review structured evidence, and AI-assisted "
            "alpha must prove uplift over the non-AI baseline before approval."
        ),
        next_build_step=(
            "Keep the deterministic AI-uplift precheck ahead of provider calls "
            "and require model-lab artifacts to include AI-vs-ML holdout metrics."
        ),
        promotion_gates=(
            "AI-vs-ML uplift artifact accepted for each AI-assisted symbol",
            "minimum multibillion local model check",
            "schema-validated JSON output",
            "no credential access",
            "cannot approve failed deterministic gates",
            "redacted report artifacts only",
        ),
        runtime_limit="Veto or review-only authority.",
    ),
)


RESEARCH_SOURCES: tuple[ResearchSourceBlueprint, ...] = (
    ResearchSourceBlueprint(
        source_id="pbo_cscv",
        label="Probability of Backtest Overfitting",
        url="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253",
        source_type="primary_research",
        applied_to=("advanced_logistic", "adaptive_hybrid_model_zoo", "lightgbm_opencl"),
        usage_policy="Use for validation design: record multiple trials, holdout behavior, and overfit risk.",
    ),
    ResearchSourceBlueprint(
        source_id="deflated_sharpe",
        label="Deflated Sharpe Ratio",
        url="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551",
        source_type="primary_research",
        applied_to=("advanced_logistic", "adaptive_hybrid_model_zoo", "lightgbm_opencl"),
        usage_policy="Use for statistical-edge skepticism; do not treat raw Sharpe or ROI as enough.",
    ),
    ResearchSourceBlueprint(
        source_id="freqai",
        label="FreqAI adaptive ML workflow",
        url="https://www.freqtrade.io/en/stable/freqai/",
        source_type="official_docs",
        applied_to=("advanced_logistic", "meta_label_gate", "regime_gate"),
        usage_policy="Use as workflow inspiration for retraining and feature boundaries, not copied strategy logic.",
    ),
    ResearchSourceBlueprint(
        source_id="patchtst",
        label="PatchTST",
        url="https://github.com/yuqinie98/patchtst",
        source_type="primary_research",
        applied_to=("patch_transformer",),
        usage_policy="Use as sequence-model inspiration; implement original compact DirectML candidate and ablation gates.",
    ),
    ResearchSourceBlueprint(
        source_id="tft",
        label="Temporal Fusion Transformer",
        url="https://arxiv.org/abs/1912.09363",
        source_type="primary_research",
        applied_to=("patch_transformer", "foundation_forecaster"),
        usage_policy="Use for interpretable multi-horizon forecast design and feature-selection ideas.",
    ),
    ResearchSourceBlueprint(
        source_id="chronos",
        label="Chronos time-series foundation models",
        url="https://github.com/amazon-science/chronos-forecasting",
        source_type="primary_research",
        applied_to=("foundation_forecaster",),
        usage_policy="Use only as timestamped probabilistic forecast features until ablation passes.",
    ),
    ResearchSourceBlueprint(
        source_id="chronos2_multivariate",
        label="Chronos-2 multivariate and covariate forecasting",
        url="https://arxiv.org/abs/2510.15821",
        source_type="primary_research",
        applied_to=("foundation_forecaster", "cross_asset_graph_sequence"),
        usage_policy="Use native cross-series context only as a forecast feature; 120M parameters do not satisfy the multibillion risk-review requirement.",
    ),
    ResearchSourceBlueprint(
        source_id="time_moe",
        label="Time-MoE billion-scale time-series foundation model",
        url="https://github.com/Time-MoE/Time-MoE",
        source_type="primary_research",
        applied_to=("foundation_forecaster",),
        usage_policy="Block until pinned DirectML compatibility, pretraining provenance, and post-cutoff baseline uplift pass.",
    ),
    ResearchSourceBlueprint(
        source_id="moirai",
        label="Moirai universal time-series transformer",
        url="https://arxiv.org/abs/2402.02592",
        source_type="primary_research",
        applied_to=("foundation_forecaster", "cross_asset_graph_sequence"),
        usage_policy="Use for universal forecast inspiration; promotion still requires AI-vs-ML uplift evidence.",
    ),
    ResearchSourceBlueprint(
        source_id="kronos_finance_foundation",
        label="Kronos finance-native time-series foundation model",
        url="https://arxiv.org/abs/2508.02739",
        source_type="primary_research",
        applied_to=("foundation_forecaster",),
        usage_policy=(
            "Benchmark open Kronos-base forecasts only as timestamped features; "
            "the upstream demonstration is not production trading evidence."
        ),
    ),
    ResearchSourceBlueprint(
        source_id="tsfm_finance_benchmark_2026",
        label="2026 pretrained TSFM financial-return benchmark",
        url="https://arxiv.org/abs/2606.27100",
        source_type="primary_research",
        applied_to=("foundation_forecaster", "patch_transformer"),
        usage_policy=(
            "Require rolling-origin and random-walk comparisons because reported "
            "forecast gains are model- and asset-specific rather than universal alpha."
        ),
    ),
    ResearchSourceBlueprint(
        source_id="ktd_fin",
        label="Memory-controlled LLM trading benchmark",
        url="https://arxiv.org/abs/2605.28359",
        source_type="primary_research",
        applied_to=("ai_risk_reviewer", "ai_uplift_gate"),
        usage_policy="Mask identifiers and calendar cues, use normalized causal factors, and attribute same-period uplift instead of accepting raw returns.",
    ),
    ResearchSourceBlueprint(
        source_id="fin_r1",
        label="Fin-R1 7B financial reasoning model",
        url="https://arxiv.org/abs/2503.16252",
        source_type="primary_research",
        applied_to=("ai_risk_reviewer", "ai_uplift_gate"),
        usage_policy=(
            "Benchmark schema-constrained risk reasoning locally; financial QA scores "
            "are not evidence of return prediction or permission to create orders."
        ),
    ),
    ResearchSourceBlueprint(
        source_id="bloomberggpt",
        label="BloombergGPT financial LLM",
        url="https://arxiv.org/abs/2303.17564",
        source_type="primary_research",
        applied_to=("ai_risk_reviewer", "ai_uplift_gate"),
        usage_policy="Use for financial-domain LLM context and multibillion-model expectations, not direct trading authority.",
    ),
    ResearchSourceBlueprint(
        source_id="fingpt",
        label="FinGPT open financial LLM framework",
        url="https://arxiv.org/abs/2306.06031",
        source_type="primary_research",
        applied_to=("ai_risk_reviewer", "ai_uplift_gate"),
        usage_policy="Use for financial data-curation and domain-adaptation inspiration; require local holdout uplift proof.",
    ),
    ResearchSourceBlueprint(
        source_id="hmm_regime_filter",
        label="Market regime filters with HMM-style state detection",
        url="https://arxiv.org/abs/2007.14874",
        source_type="primary_research",
        applied_to=("regime_gate", "meta_label_gate"),
        usage_policy="Use for abstention and cooldown design; regime models cannot create orders by themselves.",
    ),
    ResearchSourceBlueprint(
        source_id="deep_lob",
        label="DeepLOB",
        url="https://arxiv.org/abs/1808.03668",
        source_type="primary_research",
        applied_to=("deep_lob_microstructure",),
        usage_policy="Use for LOB tensor architecture inspiration after L2 depth storage exists.",
    ),
    ResearchSourceBlueprint(
        source_id="lobert",
        label="LOBERT message-level LOB foundation model",
        url="https://arxiv.org/abs/2511.12563",
        source_type="primary_research",
        applied_to=("deep_lob_microstructure",),
        usage_policy=(
            "Evaluate message tokenization only after synchronized L2 messages exist; "
            "coarse percentage bands are not a substitute for the paper's input."
        ),
    ),
    ResearchSourceBlueprint(
        source_id="hlob",
        label="HLOB order-book structure model",
        url="https://arxiv.org/abs/2405.18938",
        source_type="primary_research",
        applied_to=("deep_lob_microstructure",),
        usage_policy="Use for deeper LOB level-structure inspiration after symbol-specific depth data exists.",
    ),
    ResearchSourceBlueprint(
        source_id="binance_local_order_book",
        label="Binance local order book reconstruction",
        url=(
            "https://developers.binance.com/docs/derivatives/usds-margined-futures/"
            "websocket-market-streams/How-to-manage-a-local-order-book-correctly"
        ),
        source_type="official_docs",
        applied_to=("deep_lob_microstructure",),
        usage_policy="Use for live-like depth synchronization requirements before LOB model training.",
    ),
    ResearchSourceBlueprint(
        source_id="finmamba",
        label="FinMamba market-aware graph sequence model",
        url="https://arxiv.org/html/2502.06707v2",
        source_type="primary_research",
        applied_to=("cross_asset_graph_sequence",),
        usage_policy=(
            "Use for cross-asset dynamic graph inspiration; output remains a "
            "logged feature until ablation passes."
        ),
    ),
    ResearchSourceBlueprint(
        source_id="cross_impact_ofi",
        label="Cross-impact of order flow imbalance in equity markets",
        url="https://arxiv.org/abs/2112.13213",
        source_type="primary_research",
        applied_to=("tape_depth_gross_forecaster", "cross_asset_graph_sequence"),
        usage_policy=(
            "Test lagged cross-asset context at short horizons, but retain strong "
            "local-flow baselines and reject the added block without rolling uplift."
        ),
    ),
    ResearchSourceBlueprint(
        source_id="cryptogat",
        label="CryptoGAT cross-asset cryptocurrency forecasting",
        url="https://arxiv.org/abs/2606.27670",
        source_type="primary_research",
        applied_to=("cross_asset_graph_sequence",),
        usage_policy=(
            "Use as graph-ablation inspiration only; reported paper performance is "
            "not evidence for this repository or an execution claim."
        ),
    ),
    ResearchSourceBlueprint(
        source_id="tsfm_benchmark_integrity",
        label="Time-series foundation-model benchmark integrity requirements",
        url="https://arxiv.org/abs/2510.13654",
        source_type="primary_research",
        applied_to=("foundation_forecaster",),
        usage_policy=(
            "Treat obscure or overlapping pretraining corpora as contamination risk; "
            "require post-cutoff and genuinely future rolling evaluation."
        ),
    ),
    ResearchSourceBlueprint(
        source_id="finance_agent_benchmark",
        label="Finance Agent Benchmark",
        url="https://arxiv.org/abs/2508.00828",
        source_type="primary_research",
        applied_to=("ai_risk_reviewer", "ai_uplift_gate"),
        usage_policy=(
            "Use as evidence that finance-agent reasoning remains fallible; "
            "LLM output stays schema-constrained and veto/advisory only."
        ),
    ),
    ResearchSourceBlueprint(
        source_id="agent_market_arena",
        label="Agent Market Arena live multi-market benchmark",
        url="https://arxiv.org/abs/2510.11695",
        source_type="primary_research",
        applied_to=("ai_risk_reviewer", "rl_meta_controller"),
        usage_policy=(
            "Test coordinator, memory, and risk-style ablations independently from "
            "the LLM backbone; no direct order authority."
        ),
    ),
    ResearchSourceBlueprint(
        source_id="mambastock",
        label="MambaStock selective state-space model",
        url="https://arxiv.org/abs/2402.18959",
        source_type="primary_research",
        applied_to=("cross_asset_graph_sequence",),
        usage_policy=(
            "Use for efficient sequence-model inspiration; require repeated-seed "
            "and temporal robustness evidence."
        ),
    ),
    ResearchSourceBlueprint(
        source_id="finrl",
        label="FinRL reproducible reinforcement-learning workflow",
        url="https://arxiv.org/abs/2011.09607",
        source_type="primary_research",
        applied_to=("rl_meta_controller",),
        usage_policy=(
            "Use for environment/agent/evaluation structure, with costs, "
            "liquidity, and risk aversion embedded."
        ),
    ),
    ResearchSourceBlueprint(
        source_id="lightgbm_opencl",
        label="LightGBM GPU/OpenCL",
        url="https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html",
        source_type="official_docs",
        applied_to=("lightgbm_opencl",),
        usage_policy="Use optional OpenCL GPU training path for AMD/NVIDIA only after capability artifacts pass.",
    ),
    ResearchSourceBlueprint(
        source_id="tradingview_lorentzian",
        label="TradingView Lorentzian Classification",
        url="https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/",
        source_type="community_inspiration",
        applied_to=("adaptive_hybrid_model_zoo",),
        usage_policy="Do not copy Pine source; use only as conceptual inspiration for original neighbor features.",
    ),
    ResearchSourceBlueprint(
        source_id="tradingview_kernel",
        label="TradingView rational-quadratic kernel indicator",
        url="https://www.tradingview.com/script/AWNvbPRM-Nadaraya-Watson-Rational-Quadratic-Kernel-Non-Repainting/",
        source_type="community_inspiration",
        applied_to=("adaptive_hybrid_model_zoo",),
        usage_policy="Do not copy Pine source; use only as conceptual inspiration for original kernel confirmation.",
    ),
    ResearchSourceBlueprint(
        source_id="tradingview_technical_ratings",
        label="TradingView Technical Ratings",
        url="https://www.tradingview.com/support/solutions/43000614331-technical-ratings/",
        source_type="official_docs",
        applied_to=("adaptive_hybrid_model_zoo", "regime_gate"),
        usage_policy="Use as product/feature inspiration for compact confluence scoring, not as a copied ruleset.",
    ),
    ResearchSourceBlueprint(
        source_id="tradingview_strategies",
        label="TradingView strategy tester and order simulation",
        url="https://www.tradingview.com/pine-script-docs/concepts/strategies/",
        source_type="official_docs",
        applied_to=("adaptive_hybrid_model_zoo", "meta_label_gate"),
        usage_policy=(
            "Use for product/backtest UX expectations around order types and "
            "forward testing, not copied entries."
        ),
    ),
    ResearchSourceBlueprint(
        source_id="directml_pytorch",
        label="PyTorch with DirectML",
        url="https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows",
        source_type="official_docs",
        applied_to=("patch_transformer", "foundation_forecaster", "ai_risk_reviewer"),
        usage_policy="Use for Windows GPU capability detection and DirectML training/scoring paths.",
    ),
    ResearchSourceBlueprint(
        source_id="onnx_directml",
        label="ONNX Runtime DirectML execution provider",
        url="https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html",
        source_type="official_docs",
        applied_to=("foundation_forecaster", "ai_risk_reviewer", "cross_asset_graph_sequence"),
        usage_policy=(
            "Use for inference parity and packaging constraints; fixed shapes "
            "and session limits must be recorded."
        ),
    ),
    ResearchSourceBlueprint(
        source_id="boe_agentic_ai",
        label="Bank of England agentic AI financial-stability speech",
        url=(
            "https://www.bankofengland.co.uk/speech/2026/june/"
            "sarah-breeden-panel-at-the-european-central-bank-forum-on-central-banking-2026"
        ),
        source_type="governance",
        applied_to=("ai_risk_reviewer", "rl_meta_controller", "foundation_forecaster"),
        usage_policy=(
            "Use for governance inspiration: AI components must not bypass "
            "circuit breakers or accountability logs."
        ),
    ),
    ResearchSourceBlueprint(
        source_id="sec_market_access_rule",
        label="SEC market-access risk controls",
        url=(
            "https://www.sec.gov/rules-regulations/staff-guidance/"
            "trading-markets-frequently-asked-questions/divisionsmarketregfaq-0"
        ),
        source_type="governance",
        applied_to=("ai_risk_reviewer", "meta_label_gate", "rl_meta_controller"),
        usage_policy="Use for automated pre-trade control inspiration: limits must apply before any order is routed.",
    ),
)


RISK_BLUEPRINTS: tuple[RiskTrainingBlueprint, ...] = (
    RiskTrainingBlueprint(
        name="conservative",
        primary_focus="High precision, low trade count, small sizing, longer cooldowns, and stronger skip behavior.",
        acceptance_bias="Reject fragile path quality quickly; require high profit factor and short loss streaks.",
        promoted_families=(
            "advanced_logistic",
            "adaptive_hybrid_model_zoo",
            "regime_gate",
            "meta_label_gate",
            "ai_uplift_gate",
            "ai_risk_reviewer",
        ),
        veto_rules=(
            "no positive expectancy means no model",
            "regime concentration warning requires review",
            "meta-label evidence must not show weak take precision",
        ),
    ),
    RiskTrainingBlueprint(
        name="regular",
        primary_focus="Balanced trade frequency, calibrated thresholds, and diversified symbol acceptance.",
        acceptance_bias="Accept higher variance than conservative only when portfolio and temporal gates still pass.",
        promoted_families=(
            "advanced_logistic",
            "adaptive_hybrid_model_zoo",
            "regime_gate",
            "meta_label_gate",
            "lightgbm_opencl",
            "patch_transformer",
            "foundation_forecaster",
            "cross_asset_graph_sequence",
            "ai_uplift_gate",
            "ai_risk_reviewer",
        ),
        veto_rules=(
            "failed temporal windows block acceptance",
            "failed portfolio CVaR blocks acceptance",
            "forecast features need ablation evidence before promotion",
        ),
    ),
    RiskTrainingBlueprint(
        name="aggressive",
        primary_focus="More frequent signals and broader model exploration while preserving hard capital gates.",
        acceptance_bias="Higher drawdown tolerance, but no permission to bypass stops, CVaR, or loss-streak limits.",
        promoted_families=(
            "advanced_logistic",
            "adaptive_hybrid_model_zoo",
            "regime_gate",
            "meta_label_gate",
            "lightgbm_opencl",
            "patch_transformer",
            "foundation_forecaster",
            "cross_asset_graph_sequence",
            "deep_lob_microstructure",
            "rl_meta_controller",
            "ai_uplift_gate",
            "ai_risk_reviewer",
        ),
        veto_rules=(
            "RL cannot emit direct orders",
            "microstructure models require depth evidence",
            "leverage remains subordinate to risk controls",
        ),
    ),
)


def model_families(
    *,
    risk_level: str | None = None,
    include_research: bool = True,
) -> tuple[ModelFamilyBlueprint, ...]:
    """Return model-family blueprints, optionally filtered by risk level."""

    normalized_risk = _normalize_risk(risk_level) if risk_level else None
    values: list[ModelFamilyBlueprint] = []
    for family in MODEL_FAMILIES:
        if normalized_risk and normalized_risk not in family.risk_levels:
            continue
        if not include_research and family.status in {"research_candidate", "blocked_until_depth_store", "sandbox_only"}:
            continue
        values.append(family)
    return tuple(values)


def risk_blueprints() -> tuple[RiskTrainingBlueprint, ...]:
    """Return the risk-level training blueprints in operator order."""

    return RISK_BLUEPRINTS


def training_lanes() -> tuple[TrainingLaneBlueprint, ...]:
    """Return guarded model-promotion lanes in implementation order."""

    return TRAINING_LANES


def research_sources() -> tuple[ResearchSourceBlueprint, ...]:
    """Return the source catalog that anchors model/training inspiration."""

    return RESEARCH_SOURCES


def blueprint_payload(
    *,
    risk_level: str | None = None,
    include_research: bool = True,
) -> dict[str, object]:
    """Return a JSON-serializable blueprint payload."""

    normalized_risk = _normalize_risk(risk_level) if risk_level else None
    risks = [
        item.asdict()
        for item in RISK_BLUEPRINTS
        if normalized_risk is None or item.name == normalized_risk
    ]
    return {
        "purpose": (
            "Research-backed model/training roadmap. It is not a profitability "
            "claim and does not bypass deterministic risk gates."
        ),
        "risk_level": normalized_risk,
        "include_research": bool(include_research),
        "families": [item.asdict() for item in model_families(risk_level=normalized_risk, include_research=include_research)],
        "risk_blueprints": risks,
        "training_lanes": [item.asdict() for item in TRAINING_LANES],
        "research_sources": [item.asdict() for item in RESEARCH_SOURCES],
        "source_policy": (
            "Primary research and official docs can guide implementation. "
            "Community TradingView scripts are inspiration only; do not copy "
            "source code or promote them to direct order authority."
        ),
    }


def render_blueprint(
    *,
    risk_level: str | None = None,
    include_research: bool = True,
) -> str:
    """Render the blueprint as a concise operator-readable table."""

    payload = blueprint_payload(risk_level=risk_level, include_research=include_research)
    families = model_families(risk_level=risk_level, include_research=include_research)
    lines = [
        "Model training blueprint",
        "Purpose: research-guided roadmap; not a profitability claim.",
        "",
        f"{'family':<28} {'status':<24} {'authority':<38} role",
    ]
    for family in families:
        lines.append(
            f"{family.family:<28} {family.status:<24} "
            f"{family.execution_authority:<38} {family.role}"
        )
    lines.append("")
    lines.append("Risk-level training focus")
    for risk in payload["risk_blueprints"]:
        if not isinstance(risk, dict):
            continue
        lines.append(f"- {risk['name']}: {risk['primary_focus']}")
        lines.append(f"  acceptance: {risk['acceptance_bias']}")
    lines.append("")
    lines.append("Training lanes and promotion gates")
    for lane in TRAINING_LANES:
        lines.append(f"- {lane.lane}: {lane.next_build_step}")
        lines.append(f"  limit: {lane.runtime_limit}")
        if lane.promotion_gates:
            lines.append(f"  first gate: {lane.promotion_gates[0]}")
    lines.append("")
    lines.append("Source policy: research and docs guide design; community scripts are inspiration only.")
    return "\n".join(lines)


def validate_blueprint_contract() -> tuple[str, ...]:
    """Return contract violations that would make the roadmap unsafe."""

    errors: list[str] = []
    family_names = {family.family for family in MODEL_FAMILIES}
    if len(family_names) != len(MODEL_FAMILIES):
        errors.append("duplicate model family")
    for family in MODEL_FAMILIES:
        unknown_risks = sorted(set(family.risk_levels).difference(RISK_LEVELS))
        if unknown_risks:
            errors.append(f"{family.family} has unknown risk levels: {','.join(unknown_risks)}")
        if not family.sources or any(not source.startswith("https://") for source in family.sources):
            errors.append(f"{family.family} must cite https sources")
        if family.family in {"foundation_forecaster", "rl_meta_controller"}:
            if family.execution_authority in {"raw_buy_sell", "direct_order_authority"}:
                errors.append(f"{family.family} must not be a raw buy/sell authority")
        if family.status != "implemented" and family.execution_authority == "candidate_signal_after_objective_gates":
            errors.append(f"{family.family} is not implemented but can emit candidate signals")
    for risk in RISK_BLUEPRINTS:
        for family in risk.promoted_families:
            if family not in family_names:
                errors.append(f"{risk.name} references unknown family {family}")
    lane_family_names: set[str] = set()
    for lane in TRAINING_LANES:
        if not lane.families:
            errors.append(f"{lane.lane} has no families")
        if not lane.promotion_gates:
            errors.append(f"{lane.lane} has no promotion gates")
        if not lane.runtime_limit:
            errors.append(f"{lane.lane} has no runtime limit")
        for family in lane.families:
            if family not in family_names:
                errors.append(f"{lane.lane} references unknown family {family}")
            lane_family_names.add(family)
    missing_lane_families = sorted(family_names.difference(lane_family_names))
    if missing_lane_families:
        errors.append(f"model families missing training lane: {','.join(missing_lane_families)}")
    source_ids = [source.source_id for source in RESEARCH_SOURCES]
    if len(set(source_ids)) != len(source_ids):
        errors.append("duplicate research source id")
    allowed_source_types = {"primary_research", "official_docs", "community_inspiration", "vendor_docs", "governance"}
    for source in RESEARCH_SOURCES:
        if source.source_type not in allowed_source_types:
            errors.append(f"{source.source_id} has unknown source type {source.source_type}")
        if not source.url.startswith("https://"):
            errors.append(f"{source.source_id} must cite an https URL")
        for family in source.applied_to:
            if family not in family_names:
                errors.append(f"{source.source_id} references unknown family {family}")
        if source.source_type == "community_inspiration" and "do not copy" not in source.usage_policy.lower():
            errors.append(f"{source.source_id} community source must be no-copy")
    return tuple(errors)


def dumps_blueprint(
    *,
    risk_level: str | None = None,
    include_research: bool = True,
) -> str:
    """Serialize the blueprint with stable formatting for artifacts or CLI JSON."""

    return json.dumps(
        blueprint_payload(risk_level=risk_level, include_research=include_research),
        indent=2,
        sort_keys=True,
    )


def _normalize_risk(value: str | None) -> str:
    if not value:
        raise ValueError("risk level is required")
    normalized = str(value).strip().lower()
    aliases = {"balanced": "regular", "default": "conservative", "risky": "aggressive"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in RISK_LEVELS:
        raise ValueError(f"unknown risk level: {value}")
    return normalized


__all__ = [
    "MODEL_FAMILIES",
    "RISK_BLUEPRINTS",
    "RISK_LEVELS",
    "ModelFamilyBlueprint",
    "ResearchSourceBlueprint",
    "RiskTrainingBlueprint",
    "TrainingLaneBlueprint",
    "blueprint_payload",
    "dumps_blueprint",
    "model_families",
    "research_sources",
    "render_blueprint",
    "risk_blueprints",
    "training_lanes",
    "validate_blueprint_contract",
]
