"""Promotion/readiness checks for models before live-style execution."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .model import ModelLoadError, TrainedModel, load_model


@dataclass(frozen=True)
class ModelReadinessCheck:
    status: str
    label: str
    detail: str
    metric: float | int | str | None = None
    limit: float | int | str | None = None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ModelReadinessReport:
    checks: tuple[ModelReadinessCheck, ...]
    model_path: str | None = None

    @property
    def allowed(self) -> bool:
        return all(check.status != "block" for check in self.checks)

    @property
    def block_count(self) -> int:
        return sum(1 for check in self.checks if check.status == "block")

    @property
    def warning_count(self) -> int:
        return sum(1 for check in self.checks if check.status == "warn")

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["allowed"] = self.allowed
        payload["block_count"] = self.block_count
        payload["warning_count"] = self.warning_count
        return payload


class ModelPromotionError(ModelLoadError):
    """Raised when a readable model lacks required promotion evidence."""


def _check(
    status: str,
    label: str,
    detail: str,
    *,
    metric: float | int | str | None = None,
    limit: float | int | str | None = None,
) -> ModelReadinessCheck:
    return ModelReadinessCheck(status, label, detail, metric=metric, limit=limit)


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def build_model_readiness_report(
    model: TrainedModel,
    *,
    model_path: str | Path | None = None,
    require_selection_risk: bool = True,
    require_execution_validation: bool = True,
) -> ModelReadinessReport:
    checks: list[ModelReadinessCheck] = []
    selection_risk = getattr(model, "selection_risk", None)
    if require_selection_risk:
        if not isinstance(selection_risk, dict) or not selection_risk:
            checks.append(_check("block", "selection risk", "missing promotion evidence"))
        else:
            passed = selection_risk.get("passed") is True
            deflated_score = _finite(selection_risk.get("deflated_score"))
            selected_score = _finite(selection_risk.get("selected_score"))
            effective_trials = int(_finite(selection_risk.get("effective_trials")) or 0)
            if passed and deflated_score is not None and deflated_score > 0.0 and effective_trials > 0:
                checks.append(
                    _check(
                        "ok",
                        "selection risk",
                        f"passed deflated_score={deflated_score:+.4f} trials={effective_trials}",
                        metric=deflated_score,
                        limit=">0",
                    )
                )
            else:
                checks.append(
                    _check(
                        "block",
                        "selection risk",
                        (
                            "failed promotion evidence "
                            f"passed={passed} deflated_score={deflated_score} trials={effective_trials}"
                        ),
                        metric=deflated_score if deflated_score is not None else "missing",
                        limit=">0",
                    )
                )
            if selected_score is None or selected_score <= 0.0:
                checks.append(
                    _check(
                        "block",
                        "selected score",
                        f"non-positive selected_score={selected_score}",
                        metric=selected_score if selected_score is not None else "missing",
                        limit=">0",
                    )
                )

    execution_validation = getattr(model, "execution_validation", None)
    if require_execution_validation:
        if not isinstance(execution_validation, dict) or not execution_validation:
            checks.append(_check("block", "execution validation", "missing symbol-specific execution evidence"))
        else:
            passed = execution_validation.get("passed") is True
            stress = execution_validation.get("stress")
            temporal = execution_validation.get("temporal_robustness")
            stress_passed = isinstance(stress, dict) and stress.get("accepted") is True
            temporal_passed = isinstance(temporal, dict) and temporal.get("accepted") is True
            symbol = str(execution_validation.get("symbol") or "").strip().upper()
            if passed and stress_passed and temporal_passed and symbol:
                checks.append(_check("ok", "execution validation", f"{symbol} stress+temporal accepted"))
            else:
                checks.append(
                    _check(
                        "block",
                        "execution validation",
                        (
                            "failed symbol-specific execution evidence "
                            f"passed={passed} stress={stress_passed} temporal={temporal_passed} symbol={symbol or 'missing'}"
                        ),
                    )
                )

    policy = getattr(model, "meta_label_policy", None)
    if isinstance(policy, dict) and policy.get("enabled") is True:
        checks.append(_check("ok", "meta-label policy", str(policy.get("mode") or "enabled")))
    else:
        checks.append(_check("warn", "meta-label policy", "not enabled; entries use primary signal only"))

    warnings = [str(value) for value in getattr(model, "quality_warnings", []) or [] if str(value)]
    if any("meta_label_policy_unavailable" == warning for warning in warnings):
        checks.append(_check("warn", "model quality warnings", "meta-label policy unavailable"))
    elif warnings:
        checks.append(_check("warn", "model quality warnings", "; ".join(warnings[:3])))
    else:
        checks.append(_check("ok", "model quality warnings", "none"))

    return ModelReadinessReport(
        checks=tuple(checks),
        model_path=str(model_path) if model_path is not None else None,
    )


def assert_model_promoted(
    model: TrainedModel,
    *,
    model_path: str | Path | None = None,
    require_selection_risk: bool = True,
    require_execution_validation: bool = True,
) -> ModelReadinessReport:
    report = build_model_readiness_report(
        model,
        model_path=model_path,
        require_selection_risk=require_selection_risk,
        require_execution_validation=require_execution_validation,
    )
    if not report.allowed:
        reasons = "; ".join(f"{check.label}: {check.detail}" for check in report.checks if check.status == "block")
        raise ModelPromotionError(reasons or "model promotion evidence failed")
    return report


def load_model_readiness_report(
    model_path: str | Path,
    *,
    require_selection_risk: bool = True,
    require_execution_validation: bool = True,
) -> ModelReadinessReport:
    path = Path(model_path)
    model = load_model(path, expected_feature_version=None, expected_feature_dim=None, expected_feature_signature=None)
    return build_model_readiness_report(
        model,
        model_path=path,
        require_selection_risk=require_selection_risk,
        require_execution_validation=require_execution_validation,
    )
