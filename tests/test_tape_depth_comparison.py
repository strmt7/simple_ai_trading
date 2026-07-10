from __future__ import annotations

import hashlib
import json

import pytest

from simple_ai_trading.tape_depth_comparison import (
    compare_tape_depth_reports,
    load_and_compare_tape_depth_reports,
)
from simple_ai_trading.tape_depth_prequential import (
    TAPE_DEPTH_PREQUENTIAL_REPORT_VERSION,
)


_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def _metrics(edge: float) -> dict[str, object]:
    return {
        "rows": 10_000,
        "direction_auc": 0.5 + edge,
        "direction_brier": 0.25 - edge * 0.2,
        "prevalence_brier": 0.25,
        "mean_absolute_error_bps": 4.0 - edge * 10.0,
        "zero_baseline_mae_bps": 4.0,
        "spearman_information_coefficient": edge,
        "top_decile_mean_signed_gross_bps": edge * 20.0,
    }


def _report(
    model_profile: str,
    feature_set: str,
    *,
    selection_edge: float,
    confirmation_edge: float,
) -> dict[str, object]:
    folds = []
    for symbol_index, symbol in enumerate(_SYMBOLS):
        for fold_index in range(4):
            identity = f"{symbol}:{fold_index}".encode("ascii")
            edge = selection_edge if fold_index < 2 else confirmation_edge
            folds.append(
                {
                    "symbol": symbol,
                    "fold_index": fold_index,
                    "evaluation_start_ms": 1_700_000_000_000
                    + fold_index * 10_000_000
                    + symbol_index,
                    "evaluation_end_ms": 1_700_009_999_999
                    + fold_index * 10_000_000
                    + symbol_index,
                    "dataset_fingerprint": hashlib.sha256(identity).hexdigest(),
                    "status": "research_candidate",
                    "metrics": _metrics(edge),
                }
            )
    report = {
        "schema_version": TAPE_DEPTH_PREQUENTIAL_REPORT_VERSION,
        "status": "research_candidate",
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "config": {
            "symbols": list(_SYMBOLS),
            "model_profile": model_profile,
            "feature_set": feature_set,
            "horizon_seconds": 60,
            "risk_level": "conservative",
        },
        "plan_fingerprints": {
            symbol: hashlib.sha256(symbol.encode("ascii")).hexdigest()
            for symbol in _SYMBOLS
        },
        "folds": folds,
    }
    report["total_folds"] = len(folds)
    report["completed_folds"] = len(folds)
    return report


def test_tape_depth_comparison_confirms_single_trial_on_later_folds() -> None:
    comparison = compare_tape_depth_reports(
        [
            _report(
                "regularized",
                "core",
                selection_edge=0.02,
                confirmation_edge=0.015,
            )
        ],
        selection_fraction=0.5,
    )

    assert comparison["status"] == "confirmed_forecast_candidate"
    assert comparison["declared_trial_count"] == 1
    assert comparison["selected_trial"] == "regularized/core"
    assert comparison["confirmation"]["passed"] is True  # type: ignore[index]
    assert comparison["trading_authority"] is False
    assert comparison["profitability_claim"] is False


def test_tape_depth_comparison_does_not_fallback_after_winner_fails_confirmation() -> None:
    comparison = compare_tape_depth_reports(
        [
            _report(
                "regularized",
                "core",
                selection_edge=0.015,
                confirmation_edge=0.02,
            ),
            _report(
                "expressive",
                "full",
                selection_edge=0.03,
                confirmation_edge=-0.01,
            ),
        ],
        selection_fraction=0.5,
    )

    assert comparison["selected_trial"] == "expressive/full"
    assert comparison["status"] == "rejected"
    assert comparison["rejection_reasons"] == [
        "selected_trial_failed_later_confirmation_folds"
    ]
    assert comparison["confirmation"]["trial"] == "expressive/full"  # type: ignore[index]


def test_tape_depth_comparison_rejects_dataset_drift() -> None:
    first = _report(
        "regularized",
        "core",
        selection_edge=0.02,
        confirmation_edge=0.02,
    )
    second = _report(
        "balanced",
        "tape_derived",
        selection_edge=0.03,
        confirmation_edge=0.03,
    )
    second["folds"][0]["dataset_fingerprint"] = "f" * 64  # type: ignore[index]

    with pytest.raises(ValueError, match="identical folds and data"):
        compare_tape_depth_reports([first, second], selection_fraction=0.5)


def test_tape_depth_comparison_file_output_binds_source_reports(tmp_path) -> None:
    source = tmp_path / "report.json"
    source.write_text(
        json.dumps(
            _report(
                "regularized",
                "full",
                selection_edge=0.02,
                confirmation_edge=0.02,
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "comparison.json"

    comparison = load_and_compare_tape_depth_reports(
        [source],
        output=output,
        selection_fraction=0.5,
    )

    assert output.is_file()
    assert comparison["source_reports"][0]["sha256"] == hashlib.sha256(  # type: ignore[index]
        source.read_bytes()
    ).hexdigest()


def test_tape_depth_comparison_rejects_incomplete_fold_run() -> None:
    report = _report(
        "regularized",
        "core",
        selection_edge=0.02,
        confirmation_edge=0.02,
    )
    report["completed_folds"] = int(report["completed_folds"]) - 1

    with pytest.raises(ValueError, match="not a complete fold run"):
        compare_tape_depth_reports([report], selection_fraction=0.5)


def test_tape_depth_comparison_does_not_overwrite_source(tmp_path) -> None:
    source = tmp_path / "report.json"
    source.write_text(
        json.dumps(
            _report(
                "regularized",
                "core",
                selection_edge=0.02,
                confirmation_edge=0.02,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot overwrite"):
        load_and_compare_tape_depth_reports([source], output=source)
