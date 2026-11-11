"""Changed payoff-cost laws veto entry without taking away owned closing."""

from types import SimpleNamespace
from dataclasses import replace

import pytest

import simple_ai_trading.cli as cli
from simple_ai_trading.model import (
    HybridExpert,
    TrainedModel,
    load_model,
    serialize_model,
)
from simple_ai_trading.execution_simulation import EXECUTION_MODEL_VERSION
from simple_ai_trading.model_execution_admission import payoff_execution_version_error
from simple_ai_trading.autonomous import (
    AutonomousControl,
    Decision,
    Heartbeat,
    PositionsStore,
    STATE_RUNNING,
    STATE_STOPPING,
    _entry_gate,
    run_loop,
)
from simple_ai_trading.logging_ext import reset as reset_logger
from simple_ai_trading.objective import get_objective
from simple_ai_trading.types import RuntimeConfig, StrategyConfig
import test_autonomous as loop_fixture


@pytest.fixture(autouse=True)
def _restore_logging():
    reset_logger()
    yield
    reset_logger()


def _payoff_model(version="causal-adv-square-root-v2"):
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=0,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    model.hybrid_experts = [
        HybridExpert(
            name="fixture-payoff",
            kind="signed_payoff_ranker",
            weight=1.0,
            feature_count=1,
            params={"execution_model_version": version, "weights": [1.0], "bias": 0.0},
        )
    ]
    return model


def test_stale_payoff_model_is_blocked_before_autonomous_inference(
    monkeypatch, tmp_path
):
    model = _payoff_model()
    calls = {"candles": 0, "price": 0}

    def forbidden_prediction(_features):
        raise AssertionError("stale payoff model reached numerical inference")

    monkeypatch.setattr(model, "predict_proba", forbidden_prediction)
    monkeypatch.setattr(
        cli, "_load_live_start_model", lambda *a, **k: (model, None, None)
    )
    monkeypatch.setattr(
        cli,
        "_live_rows_for_model",
        lambda *a, **k: [SimpleNamespace(features=(0.0,), close=100.0, timestamp=123)],
    )

    class Client:
        def get_klines(self, *args, **kwargs):
            calls["candles"] += 1
            return []

        def get_symbol_price(self, symbol):
            calls["price"] += 1
            return 100.0, 123

    strategy = StrategyConfig(external_signals_enabled=False)
    decide, error, _notice = cli._build_autonomous_decision_fn(
        model_path=tmp_path / "historical.json",
        strategy=strategy,
        effective_dry_run=True,
    )
    assert error is None and decide is not None
    decision = decide(Client(), RuntimeConfig(testnet=True), strategy, None)
    assert decision.side == "FLAT" and decision.mark_price == 100.0
    assert decision.observed_at_ms == 123
    assert decision.model_entry_ready is False
    assert decision.model_entry_reason == "model-execution-cost-version-mismatch"
    assert calls == {"candles": 0, "price": 1}


@pytest.mark.parametrize(
    "kind",
    [
        "signed_payoff_ranker",
        "signed_payoff_mlp_ranker",
        "signed_payoff_lightgbm_ranker",
    ],
)
@pytest.mark.parametrize(
    ("version", "reason"),
    [
        (None, "model-execution-cost-version-missing"),
        ("", "model-execution-cost-version-missing"),
        ("causal-adv-square-root-v2", "model-execution-cost-version-mismatch"),
        (EXECUTION_MODEL_VERSION + " ", "model-execution-cost-version-mismatch"),
        (True, "model-execution-cost-metadata-invalid"),
        (3, "model-execution-cost-metadata-invalid"),
        ([], "model-execution-cost-metadata-invalid"),
        (EXECUTION_MODEL_VERSION, None),
    ],
)
def test_payoff_metadata_is_exact_and_not_a_profitability_gate(kind, version, reason):
    model = _payoff_model(version)
    model.hybrid_experts[0].kind = kind
    assert payoff_execution_version_error(model) == reason


@pytest.mark.parametrize(
    "model",
    [
        object(),
        SimpleNamespace(hybrid_experts=None),
        SimpleNamespace(hybrid_experts={}),
        SimpleNamespace(hybrid_experts=[object()]),
        SimpleNamespace(
            hybrid_experts=[SimpleNamespace(kind="signed_payoff_ranker", params=None)]
        ),
    ],
)
def test_unknown_model_metadata_is_an_entry_veto(model):
    assert (
        payoff_execution_version_error(model) == "model-execution-cost-metadata-invalid"
    )


def test_unrelated_experts_retain_their_existing_gates():
    model = _payoff_model()
    model.hybrid_experts[0].kind = "technical_confluence"
    assert payoff_execution_version_error(model) is None


def test_all_dependent_experts_must_match_including_immutable_collections():
    current = _payoff_model(EXECUTION_MODEL_VERSION).hybrid_experts[0]
    stale = _payoff_model().hybrid_experts[0]
    model = SimpleNamespace(hybrid_experts=(current, stale))
    assert (
        payoff_execution_version_error(model) == "model-execution-cost-version-mismatch"
    )
    model.hybrid_experts = (current,)
    assert payoff_execution_version_error(model) is None
    model.hybrid_experts = []
    assert payoff_execution_version_error(model) is None


def test_historical_model_remains_readable_and_is_never_retagged(tmp_path):
    path = tmp_path / "historical-model.json"
    serialize_model(_payoff_model(), path)
    original = path.read_bytes()
    loaded = load_model(path)
    assert (
        payoff_execution_version_error(loaded)
        == "model-execution-cost-version-mismatch"
    )
    assert (
        loaded.hybrid_experts[0].params["execution_model_version"]
        == "causal-adv-square-root-v2"
    )
    assert path.read_bytes() == original


@pytest.mark.parametrize("readiness", [False, None, "true", 1])
def test_opening_gate_requires_explicit_model_readiness(tmp_path, readiness):
    cfg = loop_fixture._make_config(tmp_path)
    decision = Decision(side="LONG", confidence=0.99, mark_price=100.0)
    decision.model_entry_ready = readiness
    decision.model_entry_reason = "model-execution-cost-version-mismatch"
    gate = _entry_gate(
        PositionsStore(root=cfg.positions_root),
        decision,
        loop_fixture._strategy(),
        cfg,
        get_objective("default"),
        now_ms_value=1000,
        symbol="BTCUSDC",
        market_type="spot",
    )
    assert not gate.allowed and gate.reason == "model-execution-cost-version-mismatch"


@pytest.mark.parametrize("mode", ["automatic", "operator_stop"])
def test_rejected_payoff_model_preserves_owned_closing(monkeypatch, tmp_path, mode):
    model = _payoff_model()
    monkeypatch.setattr(
        model, "predict_proba", lambda _features: pytest.fail("stale inference")
    )
    monkeypatch.setattr(
        cli, "_load_live_start_model", lambda *a, **k: (model, None, None)
    )
    cfg = loop_fixture._make_config(
        tmp_path,
        stop_after_iterations=1 if mode == "automatic" else None,
        max_unrealized_close_pct=0.05 if mode == "automatic" else None,
    )
    client = loop_fixture.FakeClient()
    loop_fixture._stage_paper_position(cfg, client, loop_fixture._make_position())
    client._price = 123.0
    strategy = replace(
        loop_fixture._strategy(), stop_loss_pct=10.0, take_profit_pct=10.0
    )
    heartbeats = []
    original_write = Heartbeat.write

    def retain_heartbeat(heartbeat, path):
        heartbeats.append((heartbeat.state, heartbeat.message))
        return original_write(heartbeat, path)

    monkeypatch.setattr(Heartbeat, "write", retain_heartbeat)
    decide, error, _notice = cli._build_autonomous_decision_fn(
        model_path=tmp_path / "historical-model.json",
        strategy=strategy,
        effective_dry_run=True,
    )
    assert error is None and decide is not None
    seen = []

    def closing_decision(*args):
        decision = decide(*args)
        seen.append(decision)
        if mode == "operator_stop":
            AutonomousControl(path=cfg.control_path).write(STATE_STOPPING)
        return decision

    result = run_loop(
        client,
        loop_fixture._runtime(),
        strategy,
        cfg,
        decision_fn=closing_decision,
        sleep=lambda _seconds: None,
        clock=loop_fixture._tick_clock(),
    )
    assert seen and all(item.model_entry_ready is False for item in seen)
    assert result.opened_trades == 0 and result.closed_trades == 1
    assert PositionsStore(root=cfg.positions_root).load_open() == []
    assert (STATE_RUNNING, "model-execution-cost-version-mismatch") in heartbeats
    if mode == "operator_stop":
        assert result.exit_reason == "operator-stop"


@pytest.mark.parametrize("phase", ["loaded_model", "paper_fallback_fit"])
def test_version_rechecked_after_feature_work_or_fallback_fit(
    monkeypatch, tmp_path, phase
):
    model = _payoff_model(EXECUTION_MODEL_VERSION)
    monkeypatch.setattr(
        model, "predict_proba", lambda _features: pytest.fail("revoked inference")
    )
    monkeypatch.setattr(
        cli,
        "_load_live_start_model",
        lambda *a, **k: (model if phase == "loaded_model" else None, None, None),
    )
    row = SimpleNamespace(features=(0.0,), close=100.0, timestamp=123)

    def build_rows(*args, **kwargs):
        model.hybrid_experts[0].params["execution_model_version"] = (
            "causal-adv-square-root-v2"
        )
        return [row]

    monkeypatch.setattr(cli, "_live_rows_for_model", build_rows)
    monkeypatch.setattr(cli, "_build_model_rows", lambda *a, **k: [row])
    monkeypatch.setattr(cli, "train", lambda *a, **k: model)
    client = SimpleNamespace(
        get_klines=lambda *a, **k: [], get_symbol_price=lambda symbol: (101.0, 456)
    )
    strategy = StrategyConfig(external_signals_enabled=False)
    decide, error, _notice = cli._build_autonomous_decision_fn(
        model_path=tmp_path / "model.json", strategy=strategy, effective_dry_run=True
    )
    assert error is None
    decision = decide(client, RuntimeConfig(testnet=True), strategy, None)
    assert decision.model_entry_ready is False and decision.mark_price == 101.0


def test_missing_veto_reason_has_an_explicit_operator_message(tmp_path):
    cfg = loop_fixture._make_config(tmp_path)
    decision = Decision(
        side="FLAT", confidence=0.0, mark_price=100.0, model_entry_ready=False
    )
    gate = _entry_gate(
        PositionsStore(root=cfg.positions_root),
        decision,
        loop_fixture._strategy(),
        cfg,
        get_objective("default"),
        now_ms_value=1000,
        symbol="BTCUSDC",
        market_type="spot",
    )
    assert not gate.allowed and gate.reason == "model-opening-evidence-unavailable"
