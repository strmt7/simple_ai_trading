from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round27_model_amendment import (
    POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256,
    load_round27_model_amendment,
    validate_round27_model_amendment,
)


_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_LEDGER = (
    _ROOT / "docs/model-research/polymarket/round-027-effective-source-ledger-v7.json"
)
_STATIC_ANALYSIS_REMEDIATION = (
    _ROOT / "docs/model-research/polymarket/"
    "round-027-static-analysis-remediation-amendment-v17.json"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_round27_model_amendment_is_exact_and_pre_target() -> None:
    amendment = load_round27_model_amendment(_ROOT)

    assert amendment["amendment_sha256"] == POLYMARKET_ROUND27_MODEL_AMENDMENT_SHA256
    assert amendment["knowledge_at_freeze"] == {
        "ai_assist_economic_metrics_computed": False,
        "model_fitted_on_stage1": False,
        "official_outcomes_accessed": False,
        "performance_metrics_computed": False,
        "sealed_partition_accessed": False,
        "selection_partition_accessed": False,
        "stage1_capture_started": True,
        "stage1_feature_rows_accessed_or_materialized": False,
    }
    assert amendment["correction"] == {
        "all_primary_target_free_audits_required": True,
        "campaign_admission_artifact_required": True,
        "contingency_role_assignment_changed": False,
        "exact_role_feature_populations_bound": True,
        "feature_model_economic_or_ai_candidates_changed": False,
        "minimum_campaign_eligible_conditions": 300,
        "model_role_minima_required": True,
        "source_ledger_advanced": True,
        "target_access_schema_version_from": (
            "polymarket-round27-role-target-access-v1"
        ),
        "target_access_schema_version_to": ("polymarket-round27-role-target-access-v2"),
        "target_store_schema_version_from": (
            "polymarket-round27-role-gated-target-store-v1"
        ),
        "target_store_schema_version_to": (
            "polymarket-round27-role-gated-target-store-v2"
        ),
    }
    assert amendment["predecessor_amendment_sha256"] == (
        "754bec3c86d36a1f88feaa806780c65ecf71e815d90dd149be8cef2cb8c6367a"
    )
    assert amendment["source_ledger"] == {
        "relative_path": (
            "docs/model-research/polymarket/round-027-effective-source-ledger-v7.json"
        ),
        "sha256": ("f38396df1bb3f8dba662370401b562ab431f6514f0fad58210079e7d6a059581"),
    }
    assert set(amendment["superseded_source_text_sha256"]) == {
        "src/simple_ai_trading/polymarket_round27_ai_cases.py",
        "src/simple_ai_trading/polymarket_round27_ai_economics.py",
        "src/simple_ai_trading/polymarket_round27_economics.py",
        "src/simple_ai_trading/polymarket_round27_experiment.py",
        "src/simple_ai_trading/polymarket_round27_features.py",
        "src/simple_ai_trading/polymarket_round27_model.py",
        "src/simple_ai_trading/polymarket_round27_operator.py",
        "src/simple_ai_trading/polymarket_round27_target_store.py",
        "tools/collect_polymarket_round27_targets.py",
        "tools/run_polymarket_round27_ai_sealed.py",
        "tools/run_polymarket_round27_ai_selection.py",
    }
    assert (
        amendment["superseded_source_text_sha256"][
            "src/simple_ai_trading/polymarket_round27_economics.py"
        ]["corrected"]
        == "fd34be8bb07bf16a528d1daae67a46dedc62e98c8fc865a6c240471a3234ec24"
    )


def test_round27_model_amendment_rejects_tampering() -> None:
    amendment = load_round27_model_amendment(_ROOT)
    tampered = copy.deepcopy(amendment)
    tampered["authority"]["edge_claim"] = True

    with pytest.raises(ValueError, match="amendment differs"):
        validate_round27_model_amendment(tampered)


def test_round27_model_amendment_rejects_transitive_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_bytes = Path.read_bytes

    def _tampered_read_bytes(path: Path) -> bytes:
        value = original_read_bytes(path)
        if path.name == "polymarket_fees.py":
            return value + b"\n"
        return value

    monkeypatch.setattr(Path, "read_bytes", _tampered_read_bytes)

    with pytest.raises(ValueError, match="source ledger file differs"):
        load_round27_model_amendment(_ROOT)


def test_round27_static_analysis_remediation_is_canonical_and_exact() -> None:
    amendment = json.loads(_STATIC_ANALYSIS_REMEDIATION.read_text(encoding="ascii"))
    claimed = amendment.pop("amendment_sha256")
    ledger = json.loads(_SOURCE_LEDGER.read_text(encoding="ascii"))

    assert claimed == "cd505ac97623af8e52c255a6b3bf09c1f8cc8be5129498f1522721c5919a6c66"
    assert claimed == _canonical_sha256(amendment)
    assert (
        amendment["predecessor_source_ledger_sha256"] == ledger["source_ledger_sha256"]
    )
    assert amendment["knowledge_at_amendment"] == {
        "model_fitted_on_stage1": False,
        "official_outcomes_accessed": False,
        "performance_metrics_computed": False,
        "stage1_feature_rows_accessed_or_materialized": False,
    }
    for relative, replacement in amendment["source_text_sha256"].items():
        assert replacement["frozen"] == ledger["files_sha256"][relative]
        assert replacement["corrected"] == _text_sha256(_ROOT / relative)


@pytest.mark.parametrize("name", [".gitattributes", "pyproject.toml", "uv.lock"])
def test_round27_historical_provenance_does_not_freeze_runtime_metadata(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_bytes = Path.read_bytes

    def _updated_read_bytes(path: Path) -> bytes:
        value = original_read_bytes(path)
        return value + b"\n" if path.name == name else value

    monkeypatch.setattr(Path, "read_bytes", _updated_read_bytes)

    assert load_round27_model_amendment(_ROOT)["source_ledger"]["sha256"] == (
        "f38396df1bb3f8dba662370401b562ab431f6514f0fad58210079e7d6a059581"
    )


def test_round27_effective_source_ledger_covers_static_import_closure() -> None:
    ledger = json.loads(_SOURCE_LEDGER.read_text(encoding="ascii"))
    assert ledger["scope"]["hash_normalization"] == (
        "replace_crlf_with_lf_before_sha256"
    )
    assert ledger["scope"]["operator_entrypoints_included"] is True
    locked = set(ledger["files_sha256"])
    excluded = set(ledger["excluded_files"])
    pending = [Path(relative) for relative in ledger["scope"]["entrypoint_files"]]
    closure: set[str] = set()

    while pending:
        relative = pending.pop()
        normalized = relative.as_posix()
        if normalized in closure or normalized in excluded:
            continue
        closure.add(normalized)
        tree = ast.parse((_ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.level == 1:
                    dependency = Path("src/simple_ai_trading") / f"{node.module}.py"
                    if (_ROOT / dependency).is_file():
                        pending.append(dependency)
                    continue
                if node.level == 0:
                    modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            for module in modules:
                parts = module.split(".")
                if parts[0] == "simple_ai_trading":
                    dependency = Path("src", *parts).with_suffix(".py")
                elif parts[0] == "tools":
                    dependency = Path(*parts).with_suffix(".py")
                else:
                    continue
                if (_ROOT / dependency).is_file():
                    pending.append(dependency)

    assert closure | {".gitattributes", "pyproject.toml", "uv.lock"} == locked
