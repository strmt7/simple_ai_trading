from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from simple_ai_trading import ai_runtime
from simple_ai_trading import api
from simple_ai_trading import backtest
from simple_ai_trading import execution_simulation
from simple_ai_trading import financial_sanity
from simple_ai_trading import lightgbm_backend
from simple_ai_trading import polymarket_action_pipeline
from simple_ai_trading import polymarket_action_value
from simple_ai_trading import polymarket_coverage
from simple_ai_trading import polymarket_features
from simple_ai_trading import polymarket_model
from simple_ai_trading import polymarket_recorder
from simple_ai_trading import polymarket_replay
from simple_ai_trading import polymarket_repricing
from simple_ai_trading import polymarket_resolution
from simple_ai_trading import polymarket_round13
from simple_ai_trading import polymarket_round21_ablation
from simple_ai_trading import polymarket_round21_model
from simple_ai_trading import polymarket_round21_sidecar_campaign
from simple_ai_trading import polymarket_round21_tcn
from simple_ai_trading import polymarket_round27_features
from simple_ai_trading import polymarket_round27_model_amendment
from simple_ai_trading import polymarket_round27_model_contract
from simple_ai_trading import polymarket_round27_stage1_capture
from simple_ai_trading import polymarket_round28_book_ticker
from simple_ai_trading.compute import BackendInfo


ROOT = Path(__file__).resolve().parents[1]


def test_model_evaluation_numeric_guards_are_typed_and_fail_closed() -> None:
    assert backtest._parse_finite_float(Decimal("1.25")) == 1.25
    assert backtest._parse_finite_float(object()) is None
    assert backtest._parse_finite_float(float("inf")) is None
    assert backtest._parse_int("3") == 3
    assert backtest._parse_int(object()) is None

    assert execution_simulation._safe_finite(Decimal("2.5"), -1.0) == 2.5
    assert execution_simulation._safe_finite(object(), -1.0) == -1.0
    assert financial_sanity._finite(Decimal("3.75")) == 3.75
    assert financial_sanity._finite(object()) is None
    assert financial_sanity._primitive_metric(None) == "missing"
    assert financial_sanity._primitive_metric(object()) == "unsupported:object"


def test_numeric_and_sequence_guards_fail_closed() -> None:
    assert ai_runtime._safe_float(object()) is None

    with pytest.raises(ValueError, match="not an array"):
        polymarket_action_pipeline._string_tuple(object(), name="values")
    with pytest.raises(ValueError, match="not an array"):
        polymarket_action_pipeline._integer_tuple(object(), name="values")
    with pytest.raises(ValueError, match="non-integer"):
        polymarket_action_pipeline._integer_tuple([float("nan")], name="values")

    with pytest.raises(ValueError, match="integer is invalid"):
        polymarket_recorder._required_int(object())
    with pytest.raises(ValueError, match="integer is invalid"):
        polymarket_recorder._required_int(float("nan"))
    with pytest.raises(ValueError, match="binary value is invalid"):
        polymarket_recorder._required_bytes(object())
    with pytest.raises(RuntimeError, match="query returned no value"):
        polymarket_recorder._required_query_value(None, name="test")

    with pytest.raises(ValueError, match="must be an integer"):
        polymarket_replay._integer(object(), name="value")
    with pytest.raises(ValueError, match="must be an integer"):
        polymarket_replay._integer(float("nan"), name="value")
    with pytest.raises(ValueError, match="below its minimum"):
        polymarket_replay._integer(0, name="value", minimum=1)

    with pytest.raises(ValueError, match="must be an integer"):
        polymarket_resolution._integer(object(), name="value")
    with pytest.raises(ValueError, match="must be an integer"):
        polymarket_resolution._integer(float("nan"), name="value")

    with pytest.raises(ValueError, match="not numeric"):
        polymarket_round27_features._finite_positive(object(), name="value")
    with pytest.raises(ValueError, match="not an integer"):
        polymarket_round27_features._integer(object(), name="value")
    with pytest.raises(ValueError, match="not an integer"):
        polymarket_round27_features._integer(float("nan"), name="value")

    with pytest.raises(ValueError, match="not an integer"):
        polymarket_round28_book_ticker._integer(object(), name="value")
    with pytest.raises(ValueError, match="not an integer"):
        polymarket_round28_book_ticker._integer(float("nan"), name="value")


def test_round13_scalar_guards_are_strict() -> None:
    with pytest.raises(ValueError, match="not a string"):
        polymarket_round13._string(1, name="value")
    assert polymarket_round13._optional_string(None, name="value") is None
    with pytest.raises(ValueError, match="not an integer"):
        polymarket_round13._integer("1", name="value")
    assert polymarket_round13._optional_integer(None, name="value") is None
    with pytest.raises(ValueError, match="not finite"):
        polymarket_round13._finite_float("1", name="value")
    with pytest.raises(ValueError, match="not finite"):
        polymarket_round13._finite_float(float("inf"), name="value")
    with pytest.raises(ValueError, match="not a boolean"):
        polymarket_round13._boolean(0, name="value")


def test_round21_numeric_and_interval_guards_are_strict() -> None:
    with pytest.raises(ValueError, match="is invalid"):
        polymarket_round21_ablation._finite_float(True, label="value")
    with pytest.raises(ValueError, match="is invalid"):
        polymarket_round21_ablation._finite_float(object(), label="value")
    with pytest.raises(ValueError, match="is invalid"):
        polymarket_round21_ablation._finite_float(float("inf"), label="value")
    assert polymarket_round21_ablation._positive_paired_lower_bounds(None) is False
    assert (
        polymarket_round21_ablation._positive_paired_lower_bounds(
            {
                "log_loss": {"lower_95": object()},
                "brier": {"lower_95": 1.0},
            }
        )
        is False
    )

    with pytest.raises(ValueError, match="is invalid"):
        polymarket_round21_model._finite_float("1", name="value")
    with pytest.raises(ValueError, match="is invalid"):
        polymarket_round21_model._finite_float(float("inf"), name="value")
    with pytest.raises(ValueError, match="is invalid"):
        polymarket_round21_tcn._integer(True, name="value")
    with pytest.raises(ValueError, match="is invalid"):
        polymarket_round21_tcn._finite_float("1", name="value")
    with pytest.raises(ValueError, match="is invalid"):
        polymarket_round21_tcn._finite_float(float("inf"), name="value")


def test_nvidia_free_memory_parser_ignores_invalid_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ai_runtime.shutil, "which", lambda _name: "nvidia-smi")
    monkeypatch.setattr(ai_runtime, "_run_capture", lambda _command: "bad\n2048\n")
    assert ai_runtime._nvidia_free_vram_gb() == 2.0

    monkeypatch.setattr(ai_runtime, "_run_capture", lambda _command: "bad\n")
    assert ai_runtime._nvidia_free_vram_gb() is None


def test_binance_type_and_commission_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        api.BinanceClient._bracket_leverage(
            {"maxLeverage": object(), "initialLeverage": 5}
        )
        == 5
    )
    client = api.BinanceClient("", "", market_type="spot")
    monkeypatch.setattr(
        client,
        "_request_dict",
        lambda *_args, **_kwargs: {
            "symbol": "BTCUSDT",
            "standardCommission": object(),
        },
    )
    with pytest.raises(api.BinanceAPIError, match="commission payload"):
        client.get_commission_rates("BTCUSDT")

    component = {"maker": 0.2, "taker": 0.2, "buyer": 0.2, "seller": 0.2}
    monkeypatch.setattr(
        client,
        "_request_dict",
        lambda *_args, **_kwargs: {
            "symbol": "BTCUSDT",
            "standardCommission": component,
            "specialCommission": component,
            "taxCommission": component,
        },
    )
    with pytest.raises(api.BinanceAPIError, match="aggregate spot"):
        client.get_commission_rates("BTCUSDT")


def test_opencl_numeric_and_device_identity_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NonIntegerNumber:
        value = "invalid"

    monkeypatch.setattr(lightgbm_backend.ctypes, "sizeof", lambda _value: 1)
    monkeypatch.setattr(lightgbm_backend.ctypes, "byref", lambda value: value)
    with pytest.raises(RuntimeError, match="non-integer"):
        lightgbm_backend._opencl_number(
            lambda *_args: 0,
            lightgbm_backend.ctypes.c_void_p(),
            0,
            NonIntegerNumber,
        )

    class IntegerNumber:
        value = 7

    assert (
        lightgbm_backend._opencl_number(
            lambda *_args: 0,
            lightgbm_backend.ctypes.c_void_p(),
            0,
            IntegerNumber,
        )
        == 7
    )

    monkeypatch.setattr(
        lightgbm_backend,
        "_opencl_device_override",
        lambda: (0, None, "opencl:0"),
    )
    resolved = BackendInfo(
        requested="auto",
        kind="cpu",
        device="cpu",
        vendor="host",
        reason="test",
    )
    parameters, backend, _label = lightgbm_backend.lightgbm_backend_parameters(
        "auto",
        1,
        resolved_backend=resolved,
        pin_opencl_device=True,
    )
    assert backend == "cpu"
    assert parameters["device_type"] == "cpu"


def test_lightgbm_probe_executes_one_fake_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = SimpleNamespace(
        Dataset=lambda features, **_kwargs: calls.append(features) or features,
        train=lambda *_args, **_kwargs: calls.append("trained"),
    )
    monkeypatch.setitem(sys.modules, "lightgbm", fake)
    lightgbm_backend._probe_lightgbm_target.cache_clear()

    available, reason = lightgbm_backend._probe_lightgbm_target("cpu", None, None)

    assert available is True
    assert reason == "one real tree update completed"
    assert len(calls) == 2


def test_posix_lock_adapter_validates_runtime_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="is invalid"):
        polymarket_round21_sidecar_campaign._integer(True, label="value")

    monkeypatch.setattr(
        polymarket_round21_sidecar_campaign.importlib,
        "import_module",
        lambda _name: SimpleNamespace(),
    )
    with pytest.raises(RuntimeError, match="locking is unavailable"):
        polymarket_round21_sidecar_campaign._posix_file_lock(3, unlock=False)

    calls: list[tuple[int, int]] = []
    module = SimpleNamespace(
        flock=lambda descriptor, operation: calls.append((descriptor, operation)),
        LOCK_UN=1,
        LOCK_EX=2,
        LOCK_NB=4,
    )
    monkeypatch.setattr(
        polymarket_round21_sidecar_campaign.importlib,
        "import_module",
        lambda _name: module,
    )
    polymarket_round21_sidecar_campaign._posix_file_lock(3, unlock=False)
    polymarket_round21_sidecar_campaign._posix_file_lock(3, unlock=True)
    assert calls == [(3, 6), (3, 1)]


def test_sidecar_file_lock_and_immutable_segment_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[bool] = []
    with monkeypatch.context() as lock_patch:
        lock_patch.setattr(polymarket_round21_sidecar_campaign.os, "name", "posix")
        lock_patch.setattr(
            polymarket_round21_sidecar_campaign,
            "_posix_file_lock",
            lambda _descriptor, *, unlock: operations.append(unlock),
        )
        with polymarket_round21_sidecar_campaign._CampaignFileLock(
            tmp_path / "campaign.lock"
        ):
            pass
        assert operations == [False, True]

        def lock_failure(_descriptor: int, *, unlock: bool) -> None:
            del unlock
            raise OSError("locked")

        lock_patch.setattr(
            polymarket_round21_sidecar_campaign,
            "_posix_file_lock",
            lock_failure,
        )
        with pytest.raises(RuntimeError, match="already running"):
            polymarket_round21_sidecar_campaign._CampaignFileLock(
                tmp_path / "campaign-failure.lock"
            ).__enter__()

    state = tmp_path / "state"
    plan = SimpleNamespace(plan_sha256="a" * 64)
    polymarket_round21_sidecar_campaign._write_segment_result(
        state,
        plan=plan,
        segment_index=0,
        status="complete",
        details={},
    )
    with pytest.raises(FileExistsError, match="already exists"):
        polymarket_round21_sidecar_campaign._write_segment_result(
            state,
            plan=plan,
            segment_index=0,
            status="complete",
            details={},
        )


class _QueryConnection:
    def __init__(self, rows: list[object]) -> None:
        self.rows = iter(rows)

    def execute(self, *_args: object, **_kwargs: object) -> _QueryConnection:
        return self

    def fetchone(self) -> object:
        return next(self.rows)

    def fetchall(self) -> list[object]:
        return []


def test_empty_query_results_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    feed_connection = _QueryConnection([("complete",), None])
    feed_store = SimpleNamespace(
        connect=lambda: feed_connection,
        resume_integrity_errors=lambda _run_id: (),
    )

    def invalid_gaps(*_args: object, **_kwargs: object) -> None:
        raise ValueError("gap")

    monkeypatch.setattr(
        polymarket_coverage.PolymarketEvidenceReplay,
        "validate_stream_gaps",
        invalid_gaps,
    )
    with pytest.raises(ValueError, match="stream-gap count is unavailable"):
        polymarket_coverage.inspect_polymarket_feed_coverage(
            feed_store,
            run_id="run",
        )

    counted_store = SimpleNamespace(
        connect=lambda: _QueryConnection([("complete",), (2,)]),
        resume_integrity_errors=lambda _run_id: (),
        iter_public_events=lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        polymarket_coverage, "load_official_resolutions", lambda *_args, **_kwargs: ()
    )
    counted = polymarket_coverage.inspect_polymarket_feed_coverage(
        counted_store,
        run_id="run",
    )
    assert counted.stream_gap_count == 2

    verified_connection = _QueryConnection([("complete",), None])
    verified_store = SimpleNamespace(
        connect=lambda: verified_connection,
        resume_integrity_errors=lambda _run_id: (),
    )
    with pytest.raises(ValueError, match="stream-gap count is unavailable"):
        polymarket_coverage.inspect_polymarket_verified_source_coverage(
            verified_store,
            run_id="run",
            condition_ids=("condition",),
            clob_baseline_condition_ids=("condition",),
            source_counts={},
        )

    action_store = SimpleNamespace(connect=lambda: _QueryConnection([None]))
    with pytest.raises(ValueError, match="count is unavailable"):
        polymarket_action_pipeline._require_no_round13_resolution_evidence(
            action_store,
            "run",
        )


def test_round13_resolution_query_results_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = "a" * 64
    monkeypatch.setattr(
        polymarket_resolution,
        "_round13_capture_contract",
        lambda _store, _run_id: contract,
    )
    missing_table_store = SimpleNamespace(
        connect=lambda: _QueryConnection([None]),
    )
    with pytest.raises(ValueError, match="table check is unavailable"):
        polymarket_resolution._require_round13_resolution_authority(
            missing_table_store,
            run_id="run",
            supplied_contract_sha256=contract,
        )

    claim_row = (
        "polymarket-round13-evaluation-claim-v1",
        "b" * 64,
        "c" * 64,
        json.dumps(["d" * 64]),
        1,
        "opened",
        "",
        "",
    )
    missing_horizon_connection = _QueryConnection([(1,), claim_row, None])
    missing_horizon_store = SimpleNamespace(connect=lambda: missing_horizon_connection)
    with pytest.raises(ValueError, match="market horizon is unavailable"):
        polymarket_resolution._require_round13_resolution_authority(
            missing_horizon_store,
            run_id="run",
            supplied_contract_sha256=contract,
        )


def test_model_and_materialization_type_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference = object.__new__(polymarket_model.PolymarketInferenceInput)
    object.__setattr__(
        inference,
        "feature_values",
        (0.0,) * len(polymarket_model.POLYMARKET_MODEL_FEATURE_NAMES),
    )
    assert set(inference.feature_map()) == set(
        polymarket_model.POLYMARKET_MODEL_FEATURE_NAMES
    )

    execution = SimpleNamespace(
        entry_result=object(),
        exit_result=object(),
        entry_filled=False,
        entry_cost_quote=None,
        exit_filled=True,
        exit_proceeds_quote=None,
        net_quote=None,
    )
    monkeypatch.setattr(
        polymarket_action_value,
        "_paper_result_reconciles",
        lambda _value: True,
    )
    with pytest.raises(ValueError, match="entry cost is missing"):
        polymarket_action_value._validate_execution_economics(execution)

    opportunity = object.__new__(polymarket_repricing.PolymarketRepricingOpportunity)
    for name in (
        "best_entry_cost_quote",
        "best_exit_proceeds_quote",
        "best_net_quote",
        "best_net_bps_on_entry_cost",
        "best_decision_received_wall_ms",
        "best_entry_received_wall_ms",
        "best_exit_decision_received_wall_ms",
        "best_exit_received_wall_ms",
        "best_entry_execution_target_wall_ms",
        "best_exit_decision_target_wall_ms",
        "best_exit_execution_target_wall_ms",
        "best_decision_received_monotonic_ns",
        "best_entry_received_monotonic_ns",
        "best_exit_decision_received_monotonic_ns",
        "best_exit_received_monotonic_ns",
        "best_entry_execution_target_monotonic_ns",
        "best_exit_decision_target_monotonic_ns",
        "best_exit_execution_target_monotonic_ns",
        "best_entry_venue_taker_delay_ms",
        "best_exit_venue_taker_delay_ms",
    ):
        object.__setattr__(opportunity, name, None)
    assert opportunity._complete_value_state_valid() is False
    assert opportunity._complete_clock_state_valid() is False
    with pytest.raises(ValueError, match="requires replay books and markets"):
        polymarket_repricing.evaluate_polymarket_repricing_ceiling(
            SimpleNamespace(markets=(), books=()),
        )

    monkeypatch.setattr(
        polymarket_features,
        "validate_polymarket_feature_source_scope",
        lambda *_args, **_kwargs: {"condition_ids": object()},
    )
    with pytest.raises(ValueError, match="source scope conditions are invalid"):
        polymarket_features.materialize_polymarket_feature_dataset(
            object(),
            SimpleNamespace(source_scope={}, run_id="run"),
        )

    with pytest.raises(ValueError, match="report counts differ"):
        polymarket_round28_book_ticker.validate_round28_book_ticker_report(
            {
                "report_sha256": "a" * 64,
                "base_decision_count": 1,
                "accepted_decision_count": 1,
                "rejected_decision_count": 0,
                "accepted_fraction": object(),
                "sidecar_raw_message_count": 1,
                "sidecar_stream_gap_count": 0,
            }
        )


def test_round27_static_remediation_rejects_invalid_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacements = polymarket_round27_model_amendment.load_round27_static_analysis_source_replacements(
        ROOT
    )
    extra = dict(replacements)
    extra["outside.py"] = ("0" * 64, "0" * 64)
    monkeypatch.setattr(
        polymarket_round27_model_amendment,
        "_load_static_analysis_source_replacements",
        lambda _root: extra,
    )
    with pytest.raises(ValueError, match="remediation scope differs"):
        polymarket_round27_model_amendment.load_round27_model_amendment(ROOT)

    wrong = dict(replacements)
    first = next(iter(wrong))
    wrong[first] = ("0" * 64, wrong[first][1])
    monkeypatch.setattr(
        polymarket_round27_model_amendment,
        "_load_static_analysis_source_replacements",
        lambda _root: wrong,
    )
    with pytest.raises(ValueError, match="remediation predecessor differs"):
        polymarket_round27_model_amendment.load_round27_model_amendment(ROOT)


def test_round27_static_remediation_parser_rejects_malformed_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = polymarket_round27_model_amendment
    artifact_path = (
        ROOT / "docs/model-research/polymarket/"
        "round-027-static-analysis-remediation-amendment-v17.json"
    )
    original = json.loads(artifact_path.read_text(encoding="ascii"))
    original_path = module._STATIC_ANALYSIS_REMEDIATION_RELATIVE_PATH

    monkeypatch.setattr(
        module,
        "_STATIC_ANALYSIS_REMEDIATION_RELATIVE_PATH",
        Path("missing-remediation.json"),
    )
    with pytest.raises(ValueError, match="remediation is unavailable"):
        module._load_static_analysis_source_replacements(ROOT)
    monkeypatch.setattr(
        module,
        "_STATIC_ANALYSIS_REMEDIATION_RELATIVE_PATH",
        original_path,
    )

    monkeypatch.setattr(
        module,
        "_load_strict",
        lambda _path: {"amendment_sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="source remediation differs"):
        module._load_static_analysis_source_replacements(ROOT)

    def install(payload: dict[str, object]) -> None:
        body = deepcopy(payload)
        body.pop("amendment_sha256", None)
        claimed = module._canonical_sha256(body)
        body["amendment_sha256"] = claimed
        monkeypatch.setattr(module, "_STATIC_ANALYSIS_REMEDIATION_SHA256", claimed)
        monkeypatch.setattr(module, "_load_strict", lambda _path: deepcopy(body))

    malformed = deepcopy(original)
    first_source = next(iter(malformed["source_text_sha256"]))
    malformed["source_text_sha256"][first_source] = []
    install(malformed)
    with pytest.raises(ValueError, match="source remediation differs"):
        module._load_static_analysis_source_replacements(ROOT)

    colliding = deepcopy(original)
    sources = colliding["source_text_sha256"]
    first_source = next(iter(sources))
    second_source = next(iter(key for key in sources if key != first_source))
    replacement = sources.pop(second_source)
    sources[f"./{first_source}"] = replacement
    install(colliding)
    with pytest.raises(ValueError, match="source remediation differs"):
        module._load_static_analysis_source_replacements(ROOT)


def test_round27_contract_layers_reject_mismatched_replacements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remediation = polymarket_round27_model_amendment.load_round27_static_analysis_source_replacements(
        ROOT
    )
    relative = "src/simple_ai_trading/polymarket_round27_features.py"
    frozen, corrected = remediation[relative]
    monkeypatch.setattr(
        polymarket_round27_model_contract,
        "load_round27_static_analysis_source_replacements",
        lambda _root: {relative: ("0" * 64, corrected)},
    )
    with pytest.raises(ValueError, match="remediation binding differs"):
        polymarket_round27_model_contract.load_round27_model_contract(ROOT)

    class StatefulReplacements(dict[str, tuple[str, str]]):
        calls = 0

        def get(
            self,
            key: str,
            default: tuple[str, str] | None = None,
        ) -> tuple[str, str] | None:
            if key != relative:
                return default
            self.calls += 1
            return (frozen, corrected) if self.calls == 1 else ("0" * 64, corrected)

    monkeypatch.setattr(
        polymarket_round27_model_contract,
        "load_round27_static_analysis_source_replacements",
        lambda _root: StatefulReplacements(),
    )
    with pytest.raises(ValueError, match="remediation binding differs"):
        polymarket_round27_model_contract.load_round27_model_contract(ROOT)


def test_round27_stage1_contract_rejects_unknown_source_hashes() -> None:
    artifact = (
        ROOT / "docs/model-research/polymarket/"
        "round-027-stage1-campaign-contract-v1.json"
    )
    original = json.loads(artifact.read_text(encoding="ascii"))

    malformed = deepcopy(original)
    malformed["source_text_sha256"]["src/simple_ai_trading/polymarket_recorder.py"] = (
        "invalid"
    )
    malformed_body = dict(malformed)
    malformed_body.pop("contract_sha256")
    malformed["contract_sha256"] = polymarket_round27_stage1_capture._canonical_sha256(
        malformed_body
    )
    with pytest.raises(ValueError, match="Stage 1 source differs"):
        polymarket_round27_stage1_capture.validate_round27_stage1_contract(
            malformed,
            repository=ROOT,
        )

    unknown = deepcopy(original)
    unknown["source_text_sha256"]["src/simple_ai_trading/polymarket_recorder.py"] = (
        "f" * 64
    )
    unknown_body = dict(unknown)
    unknown_body.pop("contract_sha256")
    unknown["contract_sha256"] = polymarket_round27_stage1_capture._canonical_sha256(
        unknown_body
    )
    with pytest.raises(ValueError, match="Stage 1 source differs"):
        polymarket_round27_stage1_capture.validate_round27_stage1_contract(
            unknown,
            repository=ROOT,
        )
