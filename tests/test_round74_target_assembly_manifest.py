from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import simple_ai_trading.round74_target_assembly_manifest as subject


RUN_ID = "1" * 32
TARGET_ENVIRONMENT = "binance_usdm_mainnet"
SCENARIO_SHA256 = "2" * 64


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


class _Assembly:
    assembly_sha256 = "3" * 64

    def __init__(self) -> None:
        self.spec = SimpleNamespace(
            execution_environment=TARGET_ENVIRONMENT,
            quantity_rules_evidence=SimpleNamespace(evidence_sha256="4" * 64),
            commission_evidence=SimpleNamespace(evidence_sha256="5" * 64),
            funding_schedule_evidence=SimpleNamespace(evidence_sha256="6" * 64),
            entry_exit_latency_evidence=SimpleNamespace(evidence_sha256="7" * 64),
            slippage_evidence=SimpleNamespace(evidence_sha256="8" * 64),
        )

    def as_dict(self) -> dict[str, object]:
        return {"assembly_sha256": self.assembly_sha256}

    @classmethod
    def from_dict(cls, value: object) -> _Assembly:
        if value != {"assembly_sha256": cls.assembly_sha256}:
            raise ValueError("fake assembly differs")
        return cls()


def _artifact_payload(
    *,
    label: str,
    environment: str,
    evidence: dict[str, str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": f"test-{label}-v1",
        "environment": environment,
        "authority": {"profitability_claim": False},
    }
    if label in {"cohort_capture", "funding", "execution_scenario"}:
        payload["run_id"] = RUN_ID
    if label == "execution_scenario":
        payload["scenario_contract_sha256"] = SCENARIO_SHA256
    payload["evidence"] = [
        {
            "kind": kind,
            "evidence_sha256": digest,
        }
        for kind, digest in evidence.items()
    ]
    return payload


def _write_artifact(
    root: Path,
    *,
    label: str,
    environment: str,
    evidence: dict[str, str],
) -> subject.Round74TargetSourceArtifactBinding:
    payload = _artifact_payload(
        label=label,
        environment=environment,
        evidence=evidence,
    )
    payload["artifact_sha256"] = _canonical_sha256(payload)
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    path = root / f"{label}.json"
    path.write_bytes(encoded)
    return subject.Round74TargetSourceArtifactBinding(
        label=label,
        relative_path=path.name,
        artifact_file_sha256=hashlib.sha256(encoded).hexdigest(),
        artifact_sha256=str(payload["artifact_sha256"]),
        environment=environment,
        evidence_sha256_by_kind=tuple(evidence.items()),
        run_id_bound=label in {"cohort_capture", "funding", "execution_scenario"},
    )


def _manifest(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> subject.Round74TargetAssemblyManifest:
    monkeypatch.setattr(subject, "Round74SourceTargetAssembly", _Assembly)
    evidence_by_label = {
        "cohort_capture": {},
        "exchange_info": {"quantity_rules": "4" * 64},
        "commission": {"commission": "5" * 64},
        "funding": {"funding_schedule": "6" * 64},
        "execution_calibration": {
            "entry_exit_latency": "9" * 64,
            "residual_slippage": "a" * 64,
        },
        "execution_scenario": {
            "entry_exit_latency": "7" * 64,
            "residual_slippage": "8" * 64,
        },
    }
    sources = tuple(
        _write_artifact(
            root,
            label=label,
            environment=(
                "binance_usdm_testnet"
                if label == "execution_calibration"
                else TARGET_ENVIRONMENT
            ),
            evidence=evidence_by_label[label],
        )
        for label in subject.ROUND74_TARGET_SOURCE_LABELS
    )
    return subject.Round74TargetAssemblyManifest(
        run_id=RUN_ID,
        cohort_binding_sha256="b" * 64,
        scenario_contract_sha256=SCENARIO_SHA256,
        assembly=_Assembly(),
        source_artifacts=sources,
    )


def test_manifest_reopens_every_source_and_binds_assembly_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    manifest = _manifest(source_root, monkeypatch)
    restored = subject.Round74TargetAssemblyManifest.from_dict(manifest.as_dict())

    assembly = subject.audit_round74_target_assembly_manifest(
        restored,
        source_artifact_root=source_root,
    )

    assert isinstance(assembly, _Assembly)
    assert restored.manifest_sha256 == manifest.manifest_sha256
    assert len(restored.source_artifacts) == 6
    assert (
        dict(restored.source_artifacts[-1].evidence_sha256_by_kind)[
            "entry_exit_latency"
        ]
        == restored.assembly.spec.entry_exit_latency_evidence.evidence_sha256
    )


def test_source_file_tamper_fails_deep_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    manifest = _manifest(source_root, monkeypatch)
    target = source_root / "funding.json"
    target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(ValueError, match="file digest differs"):
        subject.audit_round74_target_assembly_manifest(
            manifest,
            source_artifact_root=source_root,
        )


def test_mixed_environment_and_oversized_source_fail_deep_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    manifest = _manifest(source_root, monkeypatch)
    target = source_root / "exchange_info.json"
    payload = json.loads(target.read_text(encoding="ascii"))
    payload.pop("artifact_sha256")
    payload["contaminating_source"] = {"environment": "binance_usdm_testnet"}
    payload["artifact_sha256"] = _canonical_sha256(payload)
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    target.write_bytes(encoded)
    exchange = replace(
        manifest.source_artifacts[1],
        artifact_file_sha256=hashlib.sha256(encoded).hexdigest(),
        artifact_sha256=str(payload["artifact_sha256"]),
    )
    contaminated = replace(
        manifest,
        source_artifacts=(
            manifest.source_artifacts[0],
            exchange,
            *manifest.source_artifacts[2:],
        ),
    )

    with pytest.raises(ValueError, match="environment differs"):
        subject.audit_round74_target_assembly_manifest(
            contaminated,
            source_artifact_root=source_root,
        )

    monkeypatch.setattr(subject, "_MAXIMUM_SOURCE_ARTIFACT_BYTES", 1)
    with pytest.raises(ValueError, match="size differs"):
        subject.audit_round74_target_assembly_manifest(
            manifest,
            source_artifact_root=source_root,
        )


def test_manifest_rejects_missing_scenario_or_wrong_calibration_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    manifest = _manifest(source_root, monkeypatch)

    with pytest.raises(ValueError, match="identity differs"):
        subject.Round74TargetAssemblyManifest(
            run_id=manifest.run_id,
            cohort_binding_sha256=manifest.cohort_binding_sha256,
            scenario_contract_sha256=manifest.scenario_contract_sha256,
            assembly=manifest.assembly,
            source_artifacts=manifest.source_artifacts[:-1],
        ).validate()
    calibration = manifest.source_artifacts[4]
    with pytest.raises(ValueError, match="binding differs"):
        subject.Round74TargetSourceArtifactBinding(
            label=calibration.label,
            relative_path=calibration.relative_path,
            artifact_file_sha256=calibration.artifact_file_sha256,
            artifact_sha256=calibration.artifact_sha256,
            environment=TARGET_ENVIRONMENT,
            evidence_sha256_by_kind=calibration.evidence_sha256_by_kind,
            run_id_bound=False,
        ).validate()


def test_manifest_loader_rejects_bare_assembly_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    monkeypatch.setattr(subject, "Round74SourceTargetAssembly", _Assembly)
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"assembly_sha256": _Assembly.assembly_sha256}),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="manifest payload differs"):
        subject.load_and_audit_round74_target_assembly_manifest(
            manifest_path=path,
            source_artifact_root=source_root,
        )
