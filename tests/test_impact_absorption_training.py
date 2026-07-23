from __future__ import annotations

from pathlib import Path

from simple_ai_trading import impact_absorption_training as subject
from simple_ai_trading.impact_absorption_holdout_store import (
    Round73PretestManifestReport,
)
from simple_ai_trading.impact_absorption_shallow_model import (
    Round73PreparedPretestArtifacts,
)
from simple_ai_trading.impact_absorption_target_store_v3 import (
    Round73DevelopmentTargetStudyReport,
)


STUDY_ID = "a" * 32
DEVELOPMENT_SHA = "b" * 64


def test_training_reuses_one_deep_development_seal(monkeypatch, tmp_path: Path) -> None:
    development = Round73DevelopmentTargetStudyReport(
        study_id=STUDY_ID,
        source_run_count=168,
        selected_anchor_count=1_000,
        option_count=36_000,
        eligible_option_count=20_000,
        positive_option_count=2_000,
        role_run_manifests_sha256="c" * 64,
        development_study_manifest_sha256=DEVELOPMENT_SHA,
    )
    prepared = Round73PreparedPretestArtifacts(
        model_manifest={"model": "test-only"},
        artifacts={"artifact.bin": b"test-only"},
        symbol_reports=({"symbol": "BTCUSDT"},),
    )
    publication = Round73PretestManifestReport(
        study_id=STUDY_ID,
        development_study_manifest_sha256=DEVELOPMENT_SHA,
        pretest_manifest_sha256="d" * 64,
        artifact_manifest_sha256="e" * 64,
        artifact_count=1,
        artifact_bytes=9,
    )
    seal_calls = 0
    reused_seals = 0

    def seal(*_args, **_kwargs):
        nonlocal seal_calls
        seal_calls += 1
        return development

    def slices(*_args, **kwargs):
        nonlocal reused_seals
        reused = kwargs["development_seal_function"](
            tmp_path / "test.duckdb",
            study_id=STUDY_ID,
        )
        reused_seals += 1
        assert reused is development
        return iter(("test-only-slice",))

    def prepare(datasets, **_kwargs):
        assert tuple(datasets) == ("test-only-slice",)
        return prepared

    def publish(*_args, **kwargs):
        nonlocal reused_seals
        reused = kwargs["development_seal_function"](
            tmp_path / "test.duckdb",
            study_id=STUDY_ID,
        )
        reused_seals += 1
        assert reused is development
        return publication

    monkeypatch.setattr(subject, "seal_round73_development_targets", seal)
    monkeypatch.setattr(subject, "iter_round73_staged_symbol_slices", slices)
    monkeypatch.setattr(subject, "prepare_round73_pretest_artifacts", prepare)
    monkeypatch.setattr(subject, "publish_round73_pretest_manifest", publish)
    progress: list[str] = []

    report = subject.train_and_publish_round73_pretest(
        tmp_path / "test.duckdb",
        study_id=STUDY_ID,
        repository_root=tmp_path,
        progress_callback=lambda event, _payload: progress.append(event),
    )

    assert seal_calls == 1
    assert reused_seals == 2
    assert report.development is development
    assert report.prepared is prepared
    assert report.publication is publication
    assert progress == [
        "development_seal_started",
        "development_seal_completed",
        "pretest_publication_started",
        "pretest_publication_completed",
    ]
    assert report.as_dict()["development_exact_wire_audit_count"] == 1
