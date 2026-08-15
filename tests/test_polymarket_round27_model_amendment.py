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
    / "docs/model-research/polymarket/round-027-effective-source-ledger-v5.json"
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
    assert amendment["correction"] == {
        "economic_report_schema_version_from": (
            "polymarket-round27-economic-replay-v4"
        ),
        "economic_report_schema_version_to": (
            "polymarket-round27-economic-replay-v5"
        ),
        "feature_model_ai_candidates_changed": False,
        "minimum_new_entry_time_to_settlement_ms": 60_000,
        "settlement_hazard_gate_may_be_overridden": False,
        "source_ledger_advanced": True,
    }
    assert amendment["predecessor_amendment_sha256"] == (
        "759bedfbc395dab37f32d78c54433b6441ff4a231a71f9a018d1e9e80d922369"
    )
    assert amendment["source_ledger"] == {
        "relative_path": (
            "docs/model-research/polymarket/"
            "round-027-effective-source-ledger-v5.json"
        ),
        "sha256": (
            "700d89d8220f4a888d38ce67546fa7726083547c9507b7fd15f47beaef9472f2"
        ),
    }
    assert set(amendment["superseded_source_text_sha256"]) == {
        "src/simple_ai_trading/polymarket_round27_ai_cases.py",
        "src/simple_ai_trading/polymarket_round27_ai_economics.py",
        "src/simple_ai_trading/polymarket_round27_economics.py",
        "src/simple_ai_trading/polymarket_round27_experiment.py",
        "src/simple_ai_trading/polymarket_round27_features.py",
        "src/simple_ai_trading/polymarket_round27_model.py",
        "src/simple_ai_trading/polymarket_round27_operator.py",
        "tools/run_polymarket_round27_ai_sealed.py",
        "tools/run_polymarket_round27_ai_selection.py",
    }
    assert (
        amendment["superseded_source_text_sha256"][
            "src/simple_ai_trading/polymarket_round27_economics.py"
        ]["corrected"]
        == "a9f78fac647caf5eafcc7221d498545f39a7bb1fa8d3ad5bc9458f3bb2c861db"
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
