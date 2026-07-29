"""Train-only feature-support gate for the frozen BTC Polymarket shadow model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .polymarket_historical_dataset import FEATURE_NAMES
from .polymarket_historical_model import HistoricalModelPanel


SUPPORT_SCHEMA_VERSION = "polymarket-historical-btc-feature-support-v1"
MAXIMUM_OUTSIDE_TRAINING_RANGE = 4
MAXIMUM_EXTREME_OUTLIERS = 0
OUTER_IQR_MULTIPLIER = 5.0
_MAX_ARTIFACT_BYTES = 256 * 1024


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("historical support JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"historical support JSON contains {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _float_text(value: float) -> str:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("historical support statistic is nonfinite")
    return format(parsed, ".17g")


def _sha(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return normalized


@dataclass(frozen=True, slots=True)
class HistoricalFeatureSupportReport:
    status: str
    outside_training_range_count: int
    extreme_outlier_count: int
    outside_training_range_features: tuple[str, ...]
    extreme_outlier_features: tuple[str, ...]
    profile_sha256: str
    trading_authority: bool = False

    def __post_init__(self) -> None:
        if self.status not in {"in_support", "abstain"}:
            raise ValueError("historical feature-support status is invalid")
        if self.trading_authority:
            raise ValueError("historical feature-support report cannot grant authority")


@dataclass(frozen=True, slots=True)
class HistoricalFeatureSupportProfile:
    artifact_sha256: str
    pretest_artifact_sha256: str
    dataset_sha256: str
    source_commit: str
    feature_names: tuple[str, ...]
    training_rows: int
    training_conditions: int
    minimum: np.ndarray
    maximum: np.ndarray
    outer_lower: np.ndarray
    outer_upper: np.ndarray
    maximum_outside_training_range: int
    maximum_extreme_outliers: int
    trading_authority: bool = False

    def __post_init__(self) -> None:
        width = len(FEATURE_NAMES)
        if (
            self.trading_authority
            or self.feature_names != FEATURE_NAMES
            or self.training_rows < 1
            or self.training_conditions < 1
            or self.minimum.shape != (width,)
            or self.maximum.shape != (width,)
            or self.outer_lower.shape != (width,)
            or self.outer_upper.shape != (width,)
            or not all(
                np.all(np.isfinite(values))
                for values in (
                    self.minimum,
                    self.maximum,
                    self.outer_lower,
                    self.outer_upper,
                )
            )
            or np.any(self.minimum > self.maximum)
            or np.any(self.outer_lower > self.minimum)
            or np.any(self.outer_upper < self.maximum)
            or self.maximum_outside_training_range < 0
            or self.maximum_extreme_outliers < 0
        ):
            raise ValueError("historical feature-support profile is invalid")
        _sha(self.artifact_sha256, name="support profile")
        _sha(self.pretest_artifact_sha256, name="pretest artifact")
        _sha(self.dataset_sha256, name="support dataset")
        commit = str(self.source_commit or "").strip().lower()
        if len(commit) != 40 or any(
            character not in "0123456789abcdef" for character in commit
        ):
            raise ValueError("historical feature-support source commit is invalid")

    def assess(self, features: np.ndarray) -> HistoricalFeatureSupportReport:
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (len(FEATURE_NAMES),) or np.any(~np.isfinite(vector)):
            raise ValueError("historical feature-support vector is invalid")
        outside = np.flatnonzero(
            (vector < self.minimum) | (vector > self.maximum)
        )
        extreme = np.flatnonzero(
            (vector < self.outer_lower) | (vector > self.outer_upper)
        )
        status = (
            "abstain"
            if len(outside) > self.maximum_outside_training_range
            or len(extreme) > self.maximum_extreme_outliers
            else "in_support"
        )
        return HistoricalFeatureSupportReport(
            status=status,
            outside_training_range_count=int(len(outside)),
            extreme_outlier_count=int(len(extreme)),
            outside_training_range_features=tuple(
                FEATURE_NAMES[int(index)] for index in outside
            ),
            extreme_outlier_features=tuple(
                FEATURE_NAMES[int(index)] for index in extreme
            ),
            profile_sha256=self.artifact_sha256,
        )


def freeze_historical_feature_support(
    train: HistoricalModelPanel,
    *,
    pretest_artifact_sha256: str,
    source_commit: str,
) -> tuple[Mapping[str, object], str]:
    """Freeze wide train-only support bounds without held-out-test access."""

    train.validate(expected_roles=("train",))
    pretest_sha = _sha(pretest_artifact_sha256, name="pretest artifact")
    commit = str(source_commit or "").strip().lower()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("historical feature-support source commit is invalid")
    matrix = np.asarray(train.features, dtype=np.float64)
    minimum = np.min(matrix, axis=0)
    maximum = np.max(matrix, axis=0)
    first_quartile, median, third_quartile = np.quantile(
        matrix,
        (0.25, 0.5, 0.75),
        axis=0,
        method="linear",
    )
    iqr = third_quartile - first_quartile
    outer_lower = np.minimum(
        minimum,
        first_quartile - OUTER_IQR_MULTIPLIER * iqr,
    )
    outer_upper = np.maximum(
        maximum,
        third_quartile + OUTER_IQR_MULTIPLIER * iqr,
    )
    source_root = Path(__file__).parent
    payload: dict[str, object] = {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "pretest_artifact_sha256": pretest_sha,
        "dataset_sha256": train.dataset_sha256,
        "source_commit": commit,
        "implementation_sha256": {
            "support": _file_sha256(Path(__file__)),
            "model": _file_sha256(
                source_root / "polymarket_historical_model.py"
            ),
            "dataset": _file_sha256(
                source_root / "polymarket_historical_dataset.py"
            ),
        },
        "feature_names": list(FEATURE_NAMES),
        "feature_names_sha256": _canonical_sha256(FEATURE_NAMES),
        "training_rows": len(matrix),
        "training_conditions": len(np.unique(train.condition_ids)),
        "statistics": {
            "minimum": [_float_text(value) for value in minimum],
            "first_quartile": [
                _float_text(value) for value in first_quartile
            ],
            "median": [_float_text(value) for value in median],
            "third_quartile": [
                _float_text(value) for value in third_quartile
            ],
            "maximum": [_float_text(value) for value in maximum],
            "outer_lower": [_float_text(value) for value in outer_lower],
            "outer_upper": [_float_text(value) for value in outer_upper],
        },
        "gate": {
            "maximum_outside_training_range": (
                MAXIMUM_OUTSIDE_TRAINING_RANGE
            ),
            "maximum_extreme_outliers": MAXIMUM_EXTREME_OUTLIERS,
            "outer_iqr_multiplier": _float_text(OUTER_IQR_MULTIPLIER),
            "action": "abstain",
            "authority": False,
            "test_features_used": False,
            "live_features_used": False,
        },
        "trading_authority": False,
        "predictive_improvement_claim": False,
    }
    artifact_sha = _canonical_sha256(payload)
    return {**payload, "artifact_sha256": artifact_sha}, artifact_sha


def load_historical_feature_support(
    path: str | Path,
    *,
    expected_pretest_artifact_sha256: str,
    expected_dataset_sha256: str,
) -> HistoricalFeatureSupportProfile:
    support_path = Path(path)
    if support_path.is_symlink():
        raise ValueError("historical feature-support artifact cannot be a symlink")
    raw = support_path.read_bytes()
    if not raw or len(raw) > _MAX_ARTIFACT_BYTES:
        raise ValueError("historical feature-support artifact size is invalid")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "historical feature-support artifact is not strict JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise ValueError("historical feature-support artifact is not an object")
    payload = dict(value)
    claimed = _sha(
        payload.pop("artifact_sha256", ""),
        name="support profile",
    )
    if _canonical_sha256(payload) != claimed:
        raise ValueError("historical feature-support artifact integrity failed")
    if (
        payload.get("schema_version") != SUPPORT_SCHEMA_VERSION
        or payload.get("pretest_artifact_sha256")
        != _sha(
            expected_pretest_artifact_sha256,
            name="expected pretest artifact",
        )
        or payload.get("dataset_sha256")
        != _sha(expected_dataset_sha256, name="expected support dataset")
        or payload.get("feature_names") != list(FEATURE_NAMES)
        or payload.get("feature_names_sha256") != _canonical_sha256(FEATURE_NAMES)
        or payload.get("trading_authority") is not False
        or payload.get("predictive_improvement_claim") is not False
    ):
        raise ValueError("historical feature-support identity differs")
    implementation = payload.get("implementation_sha256")
    source_root = Path(__file__).parent
    expected_implementation = {
        "support": _file_sha256(Path(__file__)),
        "model": _file_sha256(source_root / "polymarket_historical_model.py"),
        "dataset": _file_sha256(
            source_root / "polymarket_historical_dataset.py"
        ),
    }
    if implementation != expected_implementation:
        raise ValueError("historical feature-support implementation differs")
    gate = payload.get("gate")
    statistics = payload.get("statistics")
    if (
        not isinstance(gate, Mapping)
        or not isinstance(statistics, Mapping)
        or dict(gate)
        != {
            "maximum_outside_training_range": (
                MAXIMUM_OUTSIDE_TRAINING_RANGE
            ),
            "maximum_extreme_outliers": MAXIMUM_EXTREME_OUTLIERS,
            "outer_iqr_multiplier": _float_text(OUTER_IQR_MULTIPLIER),
            "action": "abstain",
            "authority": False,
            "test_features_used": False,
            "live_features_used": False,
        }
    ):
        raise ValueError("historical feature-support gate differs")

    def values(name: str) -> np.ndarray:
        raw_values = statistics.get(name)
        if not isinstance(raw_values, list) or len(raw_values) != len(FEATURE_NAMES):
            raise ValueError("historical feature-support statistics differ")
        output = np.asarray([float(item) for item in raw_values], dtype=np.float64)
        if np.any(~np.isfinite(output)):
            raise ValueError("historical feature-support statistics are nonfinite")
        return output

    return HistoricalFeatureSupportProfile(
        artifact_sha256=claimed,
        pretest_artifact_sha256=str(payload["pretest_artifact_sha256"]),
        dataset_sha256=str(payload["dataset_sha256"]),
        source_commit=str(payload["source_commit"]),
        feature_names=tuple(str(item) for item in payload["feature_names"]),
        training_rows=int(payload["training_rows"]),
        training_conditions=int(payload["training_conditions"]),
        minimum=values("minimum"),
        maximum=values("maximum"),
        outer_lower=values("outer_lower"),
        outer_upper=values("outer_upper"),
        maximum_outside_training_range=int(
            gate["maximum_outside_training_range"]
        ),
        maximum_extreme_outliers=int(gate["maximum_extreme_outliers"]),
    )


__all__ = [
    "HistoricalFeatureSupportProfile",
    "HistoricalFeatureSupportReport",
    "SUPPORT_SCHEMA_VERSION",
    "freeze_historical_feature_support",
    "load_historical_feature_support",
]
