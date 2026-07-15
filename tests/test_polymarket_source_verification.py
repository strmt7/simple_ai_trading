"""Focused tests for independent Polymarket source reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from simple_ai_trading.polymarket_features import PolymarketFeatureConfig
from simple_ai_trading.polymarket_model import PolymarketModelConfig
from simple_ai_trading.polymarket_model_execution import (
    PolymarketExecutionResearchConfig,
)
from simple_ai_trading import polymarket_publication
from simple_ai_trading import polymarket_source_verification as verification
from simple_ai_trading import cli


@dataclass(frozen=True)
class _Sample:
    condition_id: str = "condition-1"
    baseline_up_probability: float = 0.55

    def asdict(self) -> dict[str, object]:
        return {
            "sample_id": "sample-1",
            "condition_id": self.condition_id,
            "baseline_up_probability": "0.55000000000000004",
        }


@dataclass(frozen=True)
class _Execution:
    payload: dict[str, object]
    report_sha256: str
    trades: tuple[object, ...] = (object(),)
    filled_order_count: int = 1

    def asdict(self) -> dict[str, object]:
        return self.payload


class _Connection:
    def execute(self, _query: str, _parameters: list[object]):
        return self

    @staticmethod
    def fetchone() -> tuple[str, str]:
        return "complete", "1" * 64


class _Store:
    def __init__(self, *_args, **kwargs: object) -> None:
        assert kwargs["read_only"] is True
        self.connection = _Connection()

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def connect(self) -> _Connection:
        return self.connection


def test_source_verifier_reconstructs_every_latency_scenario_and_fails_on_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sample = _Sample()
    feature_config = PolymarketFeatureConfig().asdict()
    model_config = PolymarketModelConfig().asdict()
    execution_config = PolymarketExecutionResearchConfig().asdict()
    feature_summary = {
        "config": feature_config,
        "dataset_sha256": "2" * 64,
    }
    model_summary = {
        "config": model_config,
        "dataset_sha256": "3" * 64,
    }
    split_summary = {"split_sha256": "4" * 64}
    model_payload = {"model_sha256": "5" * 64}
    probability_payload = {"report_sha256": "6" * 64}
    prediction = {
        **sample.asdict(),
        "model_up_probability": "0.59999999999999998",
    }

    def execution_payload(report_sha256: str) -> dict[str, object]:
        return {
            "config": execution_config,
            "market_permissions": {sample.condition_id: True},
            "decision_delay_ms_by_condition": {sample.condition_id: 0},
            "report_sha256": report_sha256,
            "trades": [{"trade_id": report_sha256}],
        }

    baseline_payload = execution_payload("7" * 64)
    model_execution_payload = execution_payload("8" * 64)
    payload = {
        "run_id": "run-1",
        "feature_dataset": feature_summary,
        "model_dataset": model_summary,
        "split": split_summary,
        "model": model_payload,
        "probability_report": probability_payload,
        "execution_latency_sensitivity": {
            "policies": {
                "baseline": {"100": baseline_payload},
                "model": {"100": model_execution_payload},
            }
        },
    }
    validated = SimpleNamespace(
        payload=payload,
        artifact_sha256="9" * 64,
        predictions=(prediction,),
    )
    feature_dataset = SimpleNamespace(
        rows=(object(), object()),
        dataset_sha256="2" * 64,
        summary=lambda: feature_summary,
    )
    model_dataset = SimpleNamespace(
        samples=(sample,),
        dataset_sha256="3" * 64,
        summary=lambda: model_summary,
    )
    split = SimpleNamespace(
        test=(sample,),
        split_sha256="4" * 64,
        summary=lambda: split_summary,
    )
    model = SimpleNamespace(
        model_sha256="5" * 64,
        asdict=lambda: model_payload,
    )
    probability = SimpleNamespace(
        report_sha256="6" * 64,
        asdict=lambda: probability_payload,
    )
    executions = {
        "7" * 64: _Execution(baseline_payload, "7" * 64),
        "8" * 64: _Execution(model_execution_payload, "8" * 64),
    }

    monkeypatch.setattr(
        polymarket_publication,
        "validate_polymarket_model_artifact",
        lambda _path: validated,
    )
    monkeypatch.setattr(verification, "PolymarketEvidenceStore", _Store)
    monkeypatch.setattr(
        verification,
        "build_polymarket_feature_dataset",
        lambda *_args, **_kwargs: feature_dataset,
    )
    monkeypatch.setattr(
        verification.PolymarketEvidenceReplay,
        "load_markets",
        lambda *_args, **_kwargs: (object(),),
    )
    monkeypatch.setattr(
        verification,
        "build_polymarket_model_dataset",
        lambda *_args, **_kwargs: model_dataset,
    )
    monkeypatch.setattr(
        verification,
        "split_polymarket_model_dataset",
        lambda _dataset: split,
    )
    monkeypatch.setattr(
        verification,
        "fit_polymarket_offset_model",
        lambda *_args: (model, probability),
    )
    monkeypatch.setattr(
        verification,
        "predict_polymarket_probabilities",
        lambda *_args: [0.6],
    )
    monkeypatch.setattr(
        verification.PolymarketEvidenceReplay,
        "load",
        lambda *_args, **_kwargs: object(),
    )

    drift = {"enabled": False}

    def evaluate(*_args, **kwargs: object) -> _Execution:
        permissions = kwargs["market_permissions"]
        report = (
            executions["7" * 64]
            if permissions == {sample.condition_id: True}
            and _args[1][0] == sample.baseline_up_probability
            else executions["8" * 64]
        )
        if drift["enabled"]:
            return _Execution(
                {**report.payload, "trades": []},
                report.report_sha256,
                trades=(),
                filled_order_count=0,
            )
        return report

    monkeypatch.setattr(
        verification,
        "evaluate_polymarket_execution_policy",
        evaluate,
    )
    report = verification.verify_polymarket_model_artifact_source(
        tmp_path / "artifact.json",
        tmp_path / "evidence.duckdb",
    )

    assert report.status == "verified"
    assert report.verified_execution_scenario_count == 2
    assert report.verified_execution_trade_count == 2
    assert report.verified_filled_order_count == 2
    verification.validate_polymarket_source_verification(
        report.asdict(),
        artifact_sha256="9" * 64,
        run_id="run-1",
    )

    drift["enabled"] = True
    with pytest.raises(ValueError, match="execution report"):
        verification.verify_polymarket_model_artifact_source(
            tmp_path / "artifact.json",
            tmp_path / "evidence.duckdb",
        )


def test_publish_cli_has_no_source_verification_bypass(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, object] = {}
    source_report = SimpleNamespace(
        report_sha256="a" * 64,
        asdict=lambda: {"report_sha256": "a" * 64},
    )

    def verify(artifact: str, database: str, **kwargs: object):
        calls["verify"] = (artifact, database, kwargs)
        return source_report

    def publish(artifact: str, root: str, **kwargs: object):
        calls["publish"] = (artifact, root, kwargs)
        assert kwargs["source_verification"] == source_report.asdict()
        return SimpleNamespace(
            asdict=lambda: {
                "artifact_sha256": "b" * 64,
                "manifest_sha256": "c" * 64,
            }
        )

    monkeypatch.setattr(cli, "verify_polymarket_model_artifact_source", verify)
    monkeypatch.setattr(cli, "publish_polymarket_model_artifact", publish)
    status = cli.main(
        [
            "polymarket-publish",
            "--artifact",
            "artifact.json",
            "--database",
            "source.duckdb",
            "--research-root",
            "docs/research",
            "--json",
        ]
    )

    assert status == 0
    assert json.loads(capsys.readouterr().out)["manifest_sha256"] == "c" * 64
    assert calls["verify"][0:2] == ("artifact.json", "source.duckdb")
    assert calls["publish"][0:2] == ("artifact.json", "docs/research")
