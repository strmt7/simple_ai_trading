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
        execution_authority="advisory_features_only",
        validation_gates=(
            "forecast timestamp audit",
            "feature ablation",
            "same-cost backtest replay",
            "cannot override risk controls",
        ),
        sources=(
            "https://arxiv.org/abs/2403.07815",
            "https://github.com/amazon-science/chronos-forecasting",
            "https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/",
            "https://arxiv.org/abs/2310.10688",
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
            "https://arxiv.org/html/2403.09267v1",
            "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints",
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
            "https://arxiv.org/abs/2111.09395",
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
            "schema validation",
            "no credential access",
            "cannot approve failed model-lab outcome",
        ),
        sources=(
            "https://airc.nist.gov/airmf-resources/airmf/5-sec-core/",
            "https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows",
            "https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html",
        ),
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
            "deep_lob_microstructure",
            "rl_meta_controller",
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
    "RiskTrainingBlueprint",
    "blueprint_payload",
    "dumps_blueprint",
    "model_families",
    "render_blueprint",
    "risk_blueprints",
    "validate_blueprint_contract",
]
