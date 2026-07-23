"""Strict, host-independent validation for the Round 73 model freeze."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Callable

from .impact_absorption_model_features import (
    ROUND73_ACTION_ALIGNED_FEATURE_NAMES,
    ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256,
    ROUND73_MODEL_FEATURE_LAYERS,
)
from .impact_absorption_store import IMPACT_CAPTURE_SYMBOLS


_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40,64}")
_ARTIFACT_NAME = re.compile(r"[a-z0-9][a-z0-9._/-]{0,127}")
_MODEL_FAMILIES = frozenset(
    {"lightgbm", "logistic_regression", "histogram_gradient_boosting"}
)
_REQUIRED_ARTIFACT_KINDS = (
    "model",
    "preprocessor",
    "training_predictions",
    "tuning_predictions",
)

RepositoryStateFunction = Callable[[str | Path], Mapping[str, object]]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_bytes(repository_root: str | Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(repository_root).resolve()), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Round 73 repository identity could not be read") from exc
    return completed.stdout


def round73_repository_state(repository_root: str | Path) -> Mapping[str, object]:
    """Return the clean, location-independent Git identity used at model freeze."""

    status = _git_bytes(
        repository_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    return {
        "commit_sha": _git_bytes(repository_root, "rev-parse", "HEAD")
        .decode("ascii")
        .strip()
        .lower(),
        "tree_sha": _git_bytes(repository_root, "rev-parse", "HEAD^{tree}")
        .decode("ascii")
        .strip()
        .lower(),
        "clean": not status,
        "dirty": bool(status),
        "status_sha256": sha256_bytes(status),
    }


def validated_repository_state(value: Mapping[str, object]) -> dict[str, object]:
    commit = str(value.get("commit_sha", "")).lower()
    tree = str(value.get("tree_sha", "")).lower()
    clean = value.get("clean")
    dirty = value.get("dirty")
    status_hash = str(value.get("status_sha256", "")).lower()
    if (
        _GIT_OBJECT_ID.fullmatch(commit) is None
        or _GIT_OBJECT_ID.fullmatch(tree) is None
        or clean is not True
        or dirty is not False
        or status_hash != sha256_bytes(b"")
    ):
        raise ValueError("Round 73 model freeze requires a clean Git identity")
    return {
        "commit_sha": commit,
        "tree_sha": tree,
        "clean": True,
        "dirty": False,
        "status_sha256": status_hash,
    }


def validated_role_rows(raw: object, *, role: str) -> dict[str, object]:
    if not isinstance(raw, Mapping) or set(raw) != set(IMPACT_CAPTURE_SYMBOLS):
        raise ValueError(f"Round 73 {role} row identities must cover every symbol")
    output: dict[str, object] = {}
    for symbol in IMPACT_CAPTURE_SYMBOLS:
        item = raw[symbol]
        if not isinstance(item, Mapping):
            raise ValueError(f"Round 73 {role} row identity is invalid: {symbol}")
        count = item.get("row_count")
        rows_hash = str(item.get("rows_sha256", "")).lower()
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or _SHA256.fullmatch(rows_hash) is None
        ):
            raise ValueError(f"Round 73 {role} row identity differs: {symbol}")
        output[symbol] = {"row_count": count, "rows_sha256": rows_hash}
    return output


def validated_feature_schema(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError("Round 73 pretest feature schema is missing")
    names = raw.get("feature_names")
    if names != list(ROUND73_ACTION_ALIGNED_FEATURE_NAMES):
        raise ValueError("Round 73 pretest feature names differ")
    if (
        str(raw.get("feature_names_sha256", "")).lower()
        != ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256
    ):
        raise ValueError("Round 73 pretest feature-name hash differs")
    transforms = raw.get("transforms")
    dropped = raw.get("dropped_zero_iqr_columns")
    if not isinstance(transforms, Mapping) or not transforms:
        raise ValueError("Round 73 pretest feature transforms are missing")
    if not isinstance(dropped, Mapping) or set(dropped) != set(IMPACT_CAPTURE_SYMBOLS):
        raise ValueError("Round 73 dropped-column identities must cover every symbol")
    known = set(ROUND73_ACTION_ALIGNED_FEATURE_NAMES)
    normalized_dropped: dict[str, list[str]] = {}
    for symbol in IMPACT_CAPTURE_SYMBOLS:
        values = dropped[symbol]
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or value not in known for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"Round 73 dropped columns differ: {symbol}")
        normalized_dropped[symbol] = list(values)
    normalized = {
        "feature_names": list(ROUND73_ACTION_ALIGNED_FEATURE_NAMES),
        "feature_names_sha256": ROUND73_ACTION_ALIGNED_FEATURE_NAMES_SHA256,
        "transforms": dict(transforms),
        "dropped_zero_iqr_columns": normalized_dropped,
    }
    _canonical_json(normalized)
    return normalized


def validated_compute_backend(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError("Round 73 pretest compute identity is missing")
    backend = str(raw.get("resolved_backend", "")).strip().lower()
    device = str(raw.get("device_name", "")).strip()
    versions = raw.get("library_versions")
    if (
        backend not in {"cpu", "opencl", "directml", "cuda", "rocm", "xpu", "mps"}
        or not device
        or not isinstance(versions, Mapping)
        or not versions
        or any(
            not str(key).strip() or not str(value).strip()
            for key, value in versions.items()
        )
    ):
        raise ValueError("Round 73 pretest compute identity differs")
    output = {
        "resolved_backend": backend,
        "device_name": device,
        "platform_name": str(raw.get("platform_name", "")).strip(),
        "device_type": str(raw.get("device_type", "")).strip(),
        "gpu_accelerated": bool(raw.get("gpu_accelerated", False)),
        "library_versions": {
            str(key): str(value) for key, value in sorted(versions.items())
        },
    }
    if backend == "cpu" and output["gpu_accelerated"]:
        raise ValueError("Round 73 CPU fallback cannot be labeled GPU accelerated")
    opencl_device = raw.get("opencl_device")
    if backend == "opencl":
        if not isinstance(opencl_device, Mapping):
            raise ValueError("Round 73 OpenCL device identity is missing")
        platform_id = opencl_device.get("platform_id")
        device_id = opencl_device.get("device_id")
        global_memory = opencl_device.get("global_memory_bytes")
        compute_units = opencl_device.get("maximum_compute_units")
        if (
            isinstance(platform_id, bool)
            or not isinstance(platform_id, int)
            or platform_id < 0
            or isinstance(device_id, bool)
            or not isinstance(device_id, int)
            or device_id < 0
            or opencl_device.get("platform_name") != output["platform_name"]
            or opencl_device.get("display_name") != output["device_name"]
            or any(
                not str(opencl_device.get(name, "")).strip()
                for name in (
                    "platform_vendor",
                    "platform_version",
                    "device_vendor",
                    "device_version",
                    "driver_version",
                )
            )
            or isinstance(global_memory, bool)
            or not isinstance(global_memory, int)
            or global_memory <= 0
            or isinstance(compute_units, bool)
            or not isinstance(compute_units, int)
            or compute_units <= 0
        ):
            raise ValueError("Round 73 OpenCL device identity differs")
        output["opencl_device"] = {
            "platform_id": platform_id,
            "device_id": device_id,
            "platform_name": output["platform_name"],
            "platform_vendor": str(opencl_device["platform_vendor"]),
            "platform_version": str(opencl_device["platform_version"]),
            "device_name": output["device_name"],
            "raw_device_name": str(opencl_device["device_name"]),
            "board_name": str(opencl_device.get("board_name", "")),
            "device_vendor": str(opencl_device["device_vendor"]),
            "device_version": str(opencl_device["device_version"]),
            "driver_version": str(opencl_device["driver_version"]),
            "global_memory_bytes": global_memory,
            "maximum_compute_units": compute_units,
        }
    elif opencl_device is not None:
        raise ValueError("Round 73 non-OpenCL backend has an OpenCL identity")
    return output


def validated_action_policy(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError("Round 73 pretest action policy is missing")
    expected_grid = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    if (
        raw.get("candidate_probability_thresholds") != expected_grid
        or raw.get("one_active_position_per_symbol") is not True
        or raw.get("pre_entry_revalidation") is not True
        or raw.get("exact_side_score_tie_policy") != "no_trade"
        or raw.get("profit_reinvestment") is not False
        or float(raw.get("leverage", 0.0)) != 1.0
    ):
        raise ValueError("Round 73 pretest action policy differs")
    output = dict(raw)
    _canonical_json(output)
    return output


def validated_symbol_models(
    raw: object,
    *,
    artifacts: Mapping[str, bytes],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not isinstance(raw, Mapping) or set(raw) != set(IMPACT_CAPTURE_SYMBOLS):
        raise ValueError("Round 73 pretest symbol models must cover every symbol")
    output: dict[str, object] = {}
    artifact_rows: list[dict[str, object]] = []
    used_names: set[str] = set()
    enabled_count = 0
    allowed_thresholds = {0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9}
    for symbol in IMPACT_CAPTURE_SYMBOLS:
        item = raw[symbol]
        if not isinstance(item, Mapping):
            raise ValueError(f"Round 73 pretest symbol model is invalid: {symbol}")
        status = str(item.get("status", "")).strip().lower()
        if status == "disabled":
            reason = str(item.get("reason", "")).strip()
            if not reason:
                raise ValueError(f"Round 73 disabled symbol lacks a reason: {symbol}")
            output[symbol] = {"status": "disabled", "reason": reason}
            continue
        if status != "enabled":
            raise ValueError(f"Round 73 symbol model status differs: {symbol}")
        enabled_count += 1
        family = str(item.get("model_family", "")).strip().lower()
        layer = str(item.get("selected_feature_layer", "")).strip().lower()
        best_iteration = item.get("best_boosting_iteration")
        threshold = item.get("probability_threshold")
        artifact_names = item.get("artifact_names")
        if (
            family not in _MODEL_FAMILIES
            or layer not in ROUND73_MODEL_FEATURE_LAYERS
            or isinstance(best_iteration, bool)
            or not isinstance(best_iteration, int)
            or best_iteration <= 0
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or float(threshold) not in allowed_thresholds
            or not isinstance(artifact_names, Mapping)
            or set(artifact_names) != set(_REQUIRED_ARTIFACT_KINDS)
        ):
            raise ValueError(f"Round 73 enabled symbol model differs: {symbol}")
        normalized_names: dict[str, str] = {}
        for kind in _REQUIRED_ARTIFACT_KINDS:
            name = str(artifact_names[kind]).strip().lower()
            payload = artifacts.get(name)
            if (
                _ARTIFACT_NAME.fullmatch(name) is None
                or name in used_names
                or not isinstance(payload, bytes)
                or not payload
            ):
                raise ValueError(f"Round 73 pretest artifact differs: {name}")
            used_names.add(name)
            normalized_names[kind] = name
            artifact_rows.append(
                {
                    "artifact_name": name,
                    "artifact_kind": kind,
                    "symbol": symbol,
                    "media_type": "application/octet-stream",
                    "artifact_sha256": sha256_bytes(payload),
                    "byte_count": len(payload),
                }
            )
        output[symbol] = {
            "status": "enabled",
            "model_family": family,
            "selected_feature_layer": layer,
            "best_boosting_iteration": best_iteration,
            "probability_threshold": float(threshold),
            "artifact_names": normalized_names,
        }
    if not enabled_count or set(artifacts) != used_names:
        raise ValueError("Round 73 pretest artifacts are incomplete or unreferenced")
    artifact_rows.sort(key=lambda value: str(value["artifact_name"]))
    return output, artifact_rows


__all__ = [
    "RepositoryStateFunction",
    "round73_repository_state",
    "sha256_bytes",
    "validated_action_policy",
    "validated_compute_backend",
    "validated_feature_schema",
    "validated_repository_state",
    "validated_role_rows",
    "validated_symbol_models",
]
