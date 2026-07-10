from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from simple_ai_trading.tape_depth_comparison import (
    confirm_tape_depth_report,
    load_and_confirm_tape_depth_report,
    load_and_select_tape_depth_reports,
    load_verified_tape_depth_selection,
    select_tape_depth_screening_reports,
    validate_tape_depth_confirmation_request,
)
from simple_ai_trading.tape_depth_prequential import (
    TAPE_DEPTH_PREQUENTIAL_REPORT_VERSION,
)


_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
_AVAILABLE_FOLDS = 8
_SCREENING_FOLDS = 4


def _refingerprint(payload: dict[str, object], field: str) -> None:
    payload.pop(field, None)
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    payload[field] = hashlib.sha256(canonical).hexdigest()


@pytest.fixture
def stub_prequential_evidence(monkeypatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(
        "simple_ai_trading.tape_depth_comparison.verify_tape_depth_prequential_report",
        lambda path, _report: calls.append(str(path)),
    )
    return calls


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
    stage: str,
    edge: float,
    selection_lock_sha256: str | None = None,
) -> dict[str, object]:
    fold_start = 0 if stage == "screening" else _SCREENING_FOLDS
    fold_indices = (
        range(_SCREENING_FOLDS)
        if stage == "screening"
        else range(_SCREENING_FOLDS, _AVAILABLE_FOLDS)
    )
    folds = []
    for symbol_index, symbol in enumerate(_SYMBOLS):
        for fold_index in fold_indices:
            identity = f"{symbol}:{fold_index}".encode("ascii")
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
    config = {
        "symbols": list(_SYMBOLS),
        "training_window_days": 730,
        "tuning_window_days": 30,
        "calibration_window_days": 30,
        "evaluation_window_days": 90,
        "horizon_seconds": 60,
        "total_latency_ms": 750,
        "decision_cadence_seconds": 20,
        "maximum_depth_age_ms": 60_000,
        "maximum_rows": 5_000_000,
        "maximum_cached_rows": 15_000_000,
        "dataset_cache": True,
        "study_stage": stage,
        "fold_start": fold_start,
        "max_folds": _SCREENING_FOLDS if stage == "screening" else 0,
        "risk_level": "conservative",
        "model_profile": model_profile,
        "feature_set": feature_set,
        "compute_backend": "auto",
        "minimum_segment_rows": 10_000,
        "selection_lock_sha256": selection_lock_sha256,
    }
    report = {
        "schema_version": TAPE_DEPTH_PREQUENTIAL_REPORT_VERSION,
        "status": "research_candidate",
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "config": config,
        "plan_fingerprints": {
            symbol: hashlib.sha256(f"{symbol}:{stage}".encode("ascii")).hexdigest()
            for symbol in _SYMBOLS
        },
        "coverage_fingerprints": {
            symbol: hashlib.sha256(f"{symbol}:coverage".encode("ascii")).hexdigest()
            for symbol in _SYMBOLS
        },
        "available_fold_counts": {
            symbol: _AVAILABLE_FOLDS for symbol in _SYMBOLS
        },
        "folds": folds,
    }
    report["total_folds"] = len(folds)
    report["completed_folds"] = len(folds)
    return report


def _selection(*, winning_edge: float = 0.03) -> dict[str, object]:
    return select_tape_depth_screening_reports(
        [
            _report("regularized", "core", stage="screening", edge=0.02),
            _report("expressive", "full", stage="screening", edge=winning_edge),
        ]
    )


def test_screening_freezes_winner_without_confirmation_metrics() -> None:
    selection = _selection()

    assert selection["status"] == "winner_frozen"
    assert selection["selected_trial"] == "expressive/full"
    assert selection["confirmation_fold_start"] == _SCREENING_FOLDS
    assert "confirmation" not in selection
    assert len(str(selection["selection_fingerprint"])) == 64
    diagnostic = selection["forecast_selection_overfit_diagnostic"]
    assert diagnostic["estimated_probability"] == 0.0  # type: ignore[index]
    assert diagnostic["symmetric_splits"] == 6  # type: ignore[index]
    assert selection["trading_authority"] is False
    assert selection["profitability_claim"] is False


def test_screening_rejects_fold_unstable_leaderboard_as_overfit() -> None:
    early = _report("regularized", "core", stage="screening", edge=0.02)
    late = _report("expressive", "full", stage="screening", edge=0.02)
    for fold in early["folds"]:  # type: ignore[union-attr]
        fold["metrics"] = _metrics(0.04 if int(fold["fold_index"]) < 2 else 0.005)
    for fold in late["folds"]:  # type: ignore[union-attr]
        fold["metrics"] = _metrics(0.005 if int(fold["fold_index"]) < 2 else 0.04)

    selection = select_tape_depth_screening_reports([early, late])

    assert selection["status"] == "rejected"
    assert selection["ranked_winner_trial"] == "regularized/core"
    assert selection["selected_trial"] is None
    assert selection["rejection_reasons"] == [
        "forecast_selection_pbo_above_0_20"
    ]
    diagnostic = selection["forecast_selection_overfit_diagnostic"]
    assert diagnostic["estimated_probability"] == pytest.approx(2 / 6)  # type: ignore[index]
    assert diagnostic["passed"] is False  # type: ignore[index]


def test_confirmation_rejects_failed_winner_without_runner_up() -> None:
    selection = _selection()
    lock_hash = "a" * 64
    confirmation = confirm_tape_depth_report(
        selection,
        _report(
            "expressive",
            "full",
            stage="confirmation",
            edge=-0.01,
            selection_lock_sha256=lock_hash,
        ),
        selection_lock_sha256=lock_hash,
    )

    assert confirmation["selected_trial"] == "expressive/full"
    assert confirmation["status"] == "rejected"
    assert confirmation["rejection_reasons"] == [
        "frozen_winner_failed_confirmation"
    ]


def test_screening_prefers_simpler_feature_set_on_exact_tie() -> None:
    selection = select_tape_depth_screening_reports(
        [
            _report("regularized", "cross_asset", stage="screening", edge=0.02),
            _report("regularized", "full", stage="screening", edge=0.02),
        ]
    )

    assert selection["selected_trial"] == "regularized/cross_asset"


def test_screening_rejects_dataset_drift() -> None:
    first = _report("regularized", "core", stage="screening", edge=0.02)
    second = _report("balanced", "tape_derived", stage="screening", edge=0.03)
    second["folds"][0]["dataset_fingerprint"] = "f" * 64  # type: ignore[index]

    with pytest.raises(ValueError, match="identical folds and data"):
        select_tape_depth_screening_reports([first, second])


def test_selection_file_recomputes_all_source_reports(
    tmp_path,
    stub_prequential_evidence,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(_report("regularized", "core", stage="screening", edge=0.02)),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(_report("expressive", "full", stage="screening", edge=0.03)),
        encoding="utf-8",
    )
    output = tmp_path / "selection.json"

    selection = load_and_select_tape_depth_reports([first, second], output=output)
    loaded, file_hash = load_verified_tape_depth_selection(output)

    assert loaded == selection
    assert file_hash == hashlib.sha256(output.read_bytes()).hexdigest()
    assert selection["source_reports"][0]["sha256"] == hashlib.sha256(  # type: ignore[index]
        first.read_bytes()
    ).hexdigest()
    assert len(stub_prequential_evidence) == 4


def test_selection_lock_rejects_changed_source_report(
    tmp_path,
    stub_prequential_evidence,
) -> None:
    source = tmp_path / "screening.json"
    source.write_text(
        json.dumps(_report("regularized", "core", stage="screening", edge=0.02)),
        encoding="utf-8",
    )
    output = tmp_path / "selection.json"
    load_and_select_tape_depth_reports([source], output=output)
    source.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="source report changed"):
        load_verified_tape_depth_selection(output)


def test_confirmation_file_flow_binds_lock_and_report(
    tmp_path,
    stub_prequential_evidence,
) -> None:
    source = tmp_path / "screening.json"
    source.write_text(
        json.dumps(_report("expressive", "full", stage="screening", edge=0.03)),
        encoding="utf-8",
    )
    selection_path = tmp_path / "selection.json"
    load_and_select_tape_depth_reports([source], output=selection_path)
    lock_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    confirmation_report = tmp_path / "confirmation-report.json"
    confirmation_report.write_text(
        json.dumps(
            _report(
                "expressive",
                "full",
                stage="confirmation",
                edge=0.02,
                selection_lock_sha256=lock_hash,
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "confirmation.json"

    confirmation = load_and_confirm_tape_depth_report(
        selection_path=selection_path,
        report_path=confirmation_report,
        output=output,
    )

    assert confirmation["status"] == "confirmed_forecast_candidate"
    assert confirmation["selection_lock_sha256"] == lock_hash
    assert output.is_file()


def test_confirmation_rejects_non_winner_report() -> None:
    selection = _selection()
    lock_hash = "b" * 64

    with pytest.raises(ValueError, match="frozen winner contract"):
        confirm_tape_depth_report(
            selection,
            _report(
                "regularized",
                "core",
                stage="confirmation",
                edge=0.02,
                selection_lock_sha256=lock_hash,
            ),
            selection_lock_sha256=lock_hash,
        )


def test_confirmation_request_requires_exact_untouched_plan_suffix() -> None:
    selection = _selection()
    config = {
        **dict(selection["modeling_config"]),  # type: ignore[arg-type]
        "dataset_cache": True,
        "maximum_cached_rows": 15_000_000,
        "study_stage": "confirmation",
        "fold_start": _SCREENING_FOLDS,
        "max_folds": 0,
        "model_profile": "expressive",
        "feature_set": "full",
        "selection_lock_sha256": "d" * 64,
    }
    coverage = dict(selection["coverage_fingerprints"])  # type: ignore[arg-type]
    boundaries = dict(selection["screening_boundaries_ms"])  # type: ignore[arg-type]
    plans = []
    for symbol_index, symbol in enumerate(_SYMBOLS):
        folds = tuple(
            SimpleNamespace(
                fold_index=index,
                evaluation_start_ms=1_700_000_000_000
                + index * 10_000_000
                + symbol_index,
            )
            for index in range(_SCREENING_FOLDS, _AVAILABLE_FOLDS)
        )
        plans.append(
            SimpleNamespace(
                symbol=symbol,
                coverage_fingerprint=coverage[symbol],
                available_fold_count=_AVAILABLE_FOLDS,
                folds=folds,
            )
        )

    validate_tape_depth_confirmation_request(selection, config=config, plans=plans)
    plans[0].folds[0].evaluation_start_ms = int(boundaries["BTCUSDT"])
    with pytest.raises(ValueError, match="untouched frozen suffix"):
        validate_tape_depth_confirmation_request(
            selection,
            config=config,
            plans=plans,
        )


def test_confirmation_rejects_incomplete_terminal_suffix() -> None:
    selection = _selection()
    lock_hash = "c" * 64
    report = _report(
        "expressive",
        "full",
        stage="confirmation",
        edge=0.02,
        selection_lock_sha256=lock_hash,
    )
    report["folds"] = report["folds"][:-1]  # type: ignore[index]
    report["total_folds"] = len(report["folds"])  # type: ignore[arg-type]
    report["completed_folds"] = len(report["folds"])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="untouched terminal suffix"):
        confirm_tape_depth_report(
            selection,
            report,
            selection_lock_sha256=lock_hash,
        )


def test_selection_rejects_incomplete_fold_run() -> None:
    report = _report("regularized", "core", stage="screening", edge=0.02)
    report["completed_folds"] = int(report["completed_folds"]) - 1

    with pytest.raises(ValueError, match="not a complete fold run"):
        select_tape_depth_screening_reports([report])


def test_selection_and_confirmation_do_not_overwrite_inputs(
    tmp_path,
    stub_prequential_evidence,
) -> None:
    source = tmp_path / "screening.json"
    source.write_text(
        json.dumps(_report("regularized", "core", stage="screening", edge=0.02)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot overwrite"):
        load_and_select_tape_depth_reports([source], output=source)

    selection_path = tmp_path / "selection.json"
    load_and_select_tape_depth_reports([source], output=selection_path)
    lock_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    report_path = tmp_path / "confirmation.json"
    report_path.write_text(
        json.dumps(
            _report(
                "regularized",
                "core",
                stage="confirmation",
                edge=0.02,
                selection_lock_sha256=lock_hash,
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot overwrite"):
        load_and_confirm_tape_depth_report(
            selection_path=selection_path,
            report_path=report_path,
            output=report_path,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("plan_not_mapping", "plan fingerprints are invalid"),
        ("plan_bad_hash", "plan fingerprints are invalid"),
        ("available_not_mapping", "available fold counts are invalid"),
        ("available_bad_int", "available fold counts are invalid"),
        ("available_zero", "available fold counts are invalid"),
        ("bad_stage", "study stage is invalid"),
        ("bad_fold_type", "fold window is invalid"),
        ("negative_fold", "fold window is invalid"),
        ("unexpected_lock", "selection-lock binding is invalid"),
        ("fold_exceeds_available", "exceed available coverage"),
    ],
)
def test_screening_report_rejects_malformed_evidence_maps_and_windows(
    case: str,
    message: str,
) -> None:
    report = _report("regularized", "core", stage="screening", edge=0.02)
    config = report["config"]
    assert isinstance(config, dict)
    if case == "plan_not_mapping":
        report["plan_fingerprints"] = None
    elif case == "plan_bad_hash":
        report["plan_fingerprints"]["BTCUSDT"] = "bad"  # type: ignore[index]
    elif case == "available_not_mapping":
        report["available_fold_counts"] = None
    elif case == "available_bad_int":
        report["available_fold_counts"]["BTCUSDT"] = "bad"  # type: ignore[index]
    elif case == "available_zero":
        report["available_fold_counts"]["BTCUSDT"] = 0  # type: ignore[index]
    elif case == "bad_stage":
        config["study_stage"] = "invalid"
    elif case == "bad_fold_type":
        config["fold_start"] = "invalid"
    elif case == "negative_fold":
        config["max_folds"] = -1
    elif case == "unexpected_lock":
        config["selection_lock_sha256"] = "a" * 64
    elif case == "fold_exceeds_available":
        report["available_fold_counts"]["BTCUSDT"] = 1  # type: ignore[index]

    with pytest.raises(ValueError, match=message):
        select_tape_depth_screening_reports([report])


def test_screening_selection_rejects_invalid_trial_sets_and_seal_depth() -> None:
    report = _report("regularized", "core", stage="screening", edge=0.02)
    with pytest.raises(ValueError, match="at least one"):
        select_tape_depth_screening_reports([])
    with pytest.raises(ValueError, match="unique profile"):
        select_tape_depth_screening_reports([report, copy.deepcopy(report)])

    development = copy.deepcopy(report)
    development["config"]["study_stage"] = "development"  # type: ignore[index]
    with pytest.raises(ValueError, match="screening-stage"):
        select_tape_depth_screening_reports([development])

    one_declared = copy.deepcopy(report)
    one_declared["config"]["max_folds"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="4, 6, 8, or 10"):
        select_tape_depth_screening_reports([one_declared])

    shallow = copy.deepcopy(report)
    shallow["available_fold_counts"] = {symbol: 5 for symbol in _SYMBOLS}
    with pytest.raises(ValueError, match="fewer than two sealed folds"):
        select_tape_depth_screening_reports([shallow])

    uneven = copy.deepcopy(report)
    uneven["folds"] = [
        fold
        for fold in uneven["folds"]  # type: ignore[union-attr]
        if not (fold["symbol"] == "SOLUSDT" and fold["fold_index"] == 3)
    ]
    uneven["total_folds"] = len(uneven["folds"])  # type: ignore[arg-type]
    uneven["completed_folds"] = len(uneven["folds"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fold count differs for SOLUSDT"):
        select_tape_depth_screening_reports([uneven])


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("fingerprint", "immutable evidence contract"),
        ("symbols", "symbols are invalid"),
        ("winner", "winner is invalid"),
        ("boundary_type", "fold boundary is invalid"),
        ("shallow_suffix", "preserve two confirmation folds"),
        ("overfit_diagnostic", "overfit diagnostic is invalid"),
    ],
)
def test_selection_lock_rejects_malformed_contract(case: str, message: str) -> None:
    selection = _selection()
    if case == "fingerprint":
        selection["status"] = "rejected"
    elif case == "symbols":
        selection["symbols"] = []
        _refingerprint(selection, "selection_fingerprint")
    elif case == "winner":
        selection["selected_model_profile"] = "unknown"
        _refingerprint(selection, "selection_fingerprint")
    elif case == "boundary_type":
        selection["confirmation_fold_start"] = "bad"
        _refingerprint(selection, "selection_fingerprint")
    elif case == "shallow_suffix":
        selection["available_fold_counts"] = {symbol: 5 for symbol in _SYMBOLS}
        _refingerprint(selection, "selection_fingerprint")
    elif case == "overfit_diagnostic":
        selection["forecast_selection_overfit_diagnostic"]["passed"] = False  # type: ignore[index]
        _refingerprint(selection, "selection_fingerprint")

    with pytest.raises(ValueError, match=message):
        confirm_tape_depth_report(
            selection,
            _report(
                "expressive",
                "full",
                stage="confirmation",
                edge=0.02,
                selection_lock_sha256="e" * 64,
            ),
            selection_lock_sha256="e" * 64,
        )


def test_confirmation_rejects_invalid_file_hash_and_request_identity() -> None:
    selection = _selection()
    report = _report(
        "expressive",
        "full",
        stage="confirmation",
        edge=0.02,
        selection_lock_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="file hash is invalid"):
        confirm_tape_depth_report(
            selection,
            report,
            selection_lock_sha256="bad",
        )

    config = {
        **dict(selection["modeling_config"]),  # type: ignore[arg-type]
        "dataset_cache": True,
        "maximum_cached_rows": 15_000_000,
        "study_stage": "development",
        "fold_start": _SCREENING_FOLDS,
        "max_folds": 0,
        "model_profile": "expressive",
        "feature_set": "full",
        "selection_lock_sha256": "f" * 64,
    }
    with pytest.raises(ValueError, match="frozen winner contract"):
        validate_tape_depth_confirmation_request(selection, config=config, plans=[])

    config["study_stage"] = "confirmation"
    config["symbols"] = list(reversed(_SYMBOLS))
    with pytest.raises(ValueError, match="frozen winner contract"):
        validate_tape_depth_confirmation_request(selection, config=config, plans=[])

    config["symbols"] = list(_SYMBOLS)
    with pytest.raises(ValueError, match="plans differ"):
        validate_tape_depth_confirmation_request(selection, config=config, plans=[])


def test_selection_file_loader_rejects_invalid_json_sources_and_recomputation(
    tmp_path,
    stub_prequential_evidence,
) -> None:
    with pytest.raises(ValueError, match="at least one screening report path"):
        load_and_select_tape_depth_reports([], output=tmp_path / "none.json")

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="is unreadable"):
        load_and_select_tape_depth_reports(
            [invalid_json],
            output=tmp_path / "invalid-selection.json",
        )

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        load_and_select_tape_depth_reports(
            [list_json],
            output=tmp_path / "list-selection.json",
        )

    oversized = tmp_path / "oversized.json"
    with oversized.open("wb") as handle:
        handle.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="exceeds the evidence size limit"):
        load_and_select_tape_depth_reports(
            [oversized],
            output=tmp_path / "oversized-selection.json",
        )

    no_sources = tmp_path / "no-sources.json"
    no_sources.write_text(json.dumps(_selection()), encoding="utf-8")
    with pytest.raises(ValueError, match="omits its screening source reports"):
        load_verified_tape_depth_selection(no_sources)

    invalid_source = _selection()
    invalid_source["source_reports"] = ["bad"]
    _refingerprint(invalid_source, "selection_fingerprint")
    invalid_source_path = tmp_path / "invalid-source.json"
    invalid_source_path.write_text(json.dumps(invalid_source), encoding="utf-8")
    with pytest.raises(ValueError, match="source report is invalid"):
        load_verified_tape_depth_selection(invalid_source_path)

    source = tmp_path / "screening.json"
    source.write_text(
        json.dumps(_report("expressive", "full", stage="screening", edge=0.03)),
        encoding="utf-8",
    )
    selection_path = tmp_path / "selection.json"
    load_and_select_tape_depth_reports([source], output=selection_path)
    changed = json.loads(selection_path.read_text(encoding="utf-8"))
    changed["limitations"] = ["altered but internally rehashed"]
    _refingerprint(changed, "selection_fingerprint")
    selection_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from recomputed"):
        load_verified_tape_depth_selection(selection_path)
