from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import simple_ai_trading.round74_event_development_inputs as subject


def _digest(value: int) -> str:
    return f"{value:064x}"


def _run_id(value: int) -> str:
    return f"{value:032x}"


class _Plan:
    training_slots = 120
    tuning_slots = 24
    test_slots = 24
    total_slots = 168
    plan_sha256 = subject.ROUND74_EVENT_COHORT_PLAN_SHA256

    def validate(self) -> None:
        return None

    @staticmethod
    def role_for_ordinal(ordinal: int) -> str:
        if ordinal < 120:
            return "training"
        if ordinal < 144:
            return "tuning"
        return "test"


@dataclass(frozen=True)
class _Binding:
    slot_ordinal: int

    @property
    def run_id(self) -> str:
        return _run_id(self.slot_ordinal + 1)

    @property
    def plan_sha256(self) -> str:
        return _Plan.plan_sha256

    @property
    def role(self) -> str:
        return _Plan.role_for_ordinal(self.slot_ordinal)

    @property
    def binding_sha256(self) -> str:
        return _digest(self.slot_ordinal + 1)

    def validate(self) -> None:
        return None


@dataclass(frozen=True)
class _Entry:
    run_id: str
    role: str


class _Partition:
    cohort_plan_sha256 = _Plan.plan_sha256
    partition_sha256 = _digest(901)
    entries = tuple(
        _Entry(_run_id(ordinal + 1), _Plan.role_for_ordinal(ordinal))
        for ordinal in range(_Plan.total_slots)
    )

    def validate(self) -> None:
        return None


class _Assembly:
    loaded_run_ids: list[str] = []

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.assembly_sha256 = _digest(int(run_id, 16) + 1_000)
        self.spec = SimpleNamespace(
            execution_environment=subject.ROUND74_DEVELOPMENT_EXECUTION_ENVIRONMENT
        )


class _Manifest:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.cohort_binding_sha256 = _digest(int(run_id, 16))
        self.manifest_sha256 = _digest(int(run_id, 16) + 2_000)
        self.assembly = _Assembly(run_id)


def _write_panel(root: Path) -> tuple[Path, Path, Path, Path]:
    plan = root / "plan.json"
    bindings = root / "bindings"
    assemblies = root / "assemblies"
    sources = root / "sources"
    bindings.mkdir()
    assemblies.mkdir()
    sources.mkdir()
    plan.write_text("{}\n", encoding="ascii")
    for ordinal in range(_Plan.total_slots):
        (bindings / f"{ordinal:03d}.json").write_text(
            json.dumps({"ordinal": ordinal}) + "\n",
            encoding="ascii",
        )
    for entry in _Partition.entries[:144]:
        (assemblies / f"{entry.run_id}.json").write_text(
            json.dumps({"run_id": entry.run_id}) + "\n",
            encoding="ascii",
        )
    return plan, bindings, assemblies, sources


def test_round74_development_input_loader_never_reads_sealed_assemblies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_path, bindings, assemblies, sources = _write_panel(tmp_path)
    _Assembly.loaded_run_ids = []
    monkeypatch.setattr(subject, "load_round74_event_cohort_plan", lambda _raw: _Plan())
    monkeypatch.setattr(
        subject,
        "load_round74_event_cohort_binding",
        lambda raw: _Binding(json.loads(raw)["ordinal"]),
    )
    monkeypatch.setattr(
        subject,
        "build_round74_event_run_partition",
        lambda _plan, _bindings: _Partition(),
    )
    monkeypatch.setattr(subject, "Round74SourceTargetAssembly", _Assembly)

    def _load_manifest(*, manifest_path: Path, **_kwargs: object) -> _Manifest:
        run_id = manifest_path.stem
        _Assembly.loaded_run_ids.append(run_id)
        return _Manifest(run_id)

    monkeypatch.setattr(
        subject,
        "load_and_audit_round74_target_assembly_manifest",
        _load_manifest,
    )

    result = subject.load_round74_development_inputs(
        plan_path=plan_path,
        binding_directory=bindings,
        target_assembly_directory=assemblies,
        source_artifact_root=sources,
    )

    expected = [_run_id(ordinal + 1) for ordinal in range(144)]
    assert _Assembly.loaded_run_ids == expected
    assert list(result.target_assembly_by_run_id()) == expected
    assert result.as_dict()["sealed_test_target_assemblies_read"] is False
    assert len(result.inputs_sha256) == 64

    sealed_run_id = _Partition.entries[144].run_id
    (assemblies / f"{sealed_run_id}.json").write_text(
        json.dumps({"run_id": sealed_run_id}) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="manifest file panel differs"):
        subject.load_round74_development_inputs(
            plan_path=plan_path,
            binding_directory=bindings,
            target_assembly_directory=assemblies,
            source_artifact_root=sources,
        )
    assert _Assembly.loaded_run_ids == expected
