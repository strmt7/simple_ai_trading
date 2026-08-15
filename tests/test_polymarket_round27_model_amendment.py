from __future__ import annotations

import ast
import copy
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
    _ROOT
    / "docs/model-research/polymarket/round-027-effective-source-ledger-v3.json"
)


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
    assert amendment["correction"]["lag_zero_excluded_from_insignificance_scan"] is True
    assert amendment["correction"]["fixed_expected_block_lengths_conditions_retained"] == [
        1,
        4,
        12,
    ]
    assert amendment["correction"][
        "economic_or_prediction_gate_numeric_thresholds_changed"
    ] is False
    assert amendment["predecessor_amendment_sha256"] == (
        "5dc338fbd521e02bdecd6e90df185e6d8276276556fa6c4d2425faccf809c731"
    )
    assert amendment["source_ledger"] == {
        "relative_path": (
            "docs/model-research/polymarket/"
            "round-027-effective-source-ledger-v3.json"
        ),
        "sha256": (
            "972ef3e49f16ced1706a3ff0b91dae72033ae48dee9f4b15794585de26fa9493"
        ),
    }
    assert set(amendment["superseded_source_text_sha256"]) == {
        "src/simple_ai_trading/polymarket_round27_ai_cases.py",
        "src/simple_ai_trading/polymarket_round27_ai_economics.py",
        "src/simple_ai_trading/polymarket_round27_economics.py",
        "src/simple_ai_trading/polymarket_round27_experiment.py",
        "src/simple_ai_trading/polymarket_round27_features.py",
        "src/simple_ai_trading/polymarket_round27_model.py",
    }
    assert (
        amendment["superseded_source_text_sha256"][
            "src/simple_ai_trading/polymarket_round27_model.py"
        ]["corrected"]
        == "76360e4541ab7118e9ea29561d20d18dcc97dd32fffff07fce6d11af2452d4bf"
    )
    assert amendment["superseded_source_text_sha256"][
        "src/simple_ai_trading/polymarket_round27_features.py"
    ]["corrected"] == (
        "d74d97b9bab0dba46d2b207b845da1d4b8028972bc636e0674f759cecb22f027"
    )
    assert amendment["superseded_source_text_sha256"][
        "src/simple_ai_trading/polymarket_round27_experiment.py"
    ]["corrected"] == (
        "51b9077781cabb6d3f8fd7033894b41a0b5ed2d7cf911eb4b573df6f902c63c1"
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


def test_round27_effective_source_ledger_covers_static_import_closure() -> None:
    ledger = json.loads(_SOURCE_LEDGER.read_text(encoding="ascii"))
    assert ledger["scope"]["hash_normalization"] == (
        "replace_crlf_with_lf_before_sha256"
    )
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
            if not isinstance(node, ast.ImportFrom) or node.level != 1 or not node.module:
                continue
            dependency = Path("src/simple_ai_trading") / f"{node.module}.py"
            if (_ROOT / dependency).is_file():
                pending.append(dependency)

    assert closure | {".gitattributes", "pyproject.toml", "uv.lock"} == locked
