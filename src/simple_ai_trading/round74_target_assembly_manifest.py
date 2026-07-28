"""Source-artifact manifest for one Round 74 executable target assembly."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from .impact_absorption_target_assembly import Round74SourceTargetAssembly
from .impact_absorption_execution_scenario import (
    ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT_SHA256,
    ROUND74_PUBLIC_EXECUTION_SCENARIO_LATENCY_SOURCE_ID,
    ROUND74_PUBLIC_EXECUTION_SCENARIO_SCHEMA_VERSION,
    ROUND74_PUBLIC_EXECUTION_SCENARIO_SELECTED_NAME,
    ROUND74_PUBLIC_EXECUTION_SCENARIO_SLIPPAGE_SOURCE_ID,
)


ROUND74_TARGET_ASSEMBLY_MANIFEST_SCHEMA_VERSION = (
    "round-074-target-assembly-manifest-v1"
)
ROUND74_TARGET_SOURCE_LABELS = (
    "cohort_capture",
    "exchange_info",
    "commission",
    "funding",
    "execution_calibration",
    "execution_scenario",
)
_EVIDENCE_KINDS_BY_LABEL = {
    "cohort_capture": (),
    "exchange_info": ("quantity_rules",),
    "commission": ("commission",),
    "funding": ("funding_schedule",),
    "execution_calibration": ("entry_exit_latency", "residual_slippage"),
    "execution_scenario": ("entry_exit_latency", "residual_slippage"),
}
_RUN_BOUND_LABELS = frozenset({"cohort_capture", "funding", "execution_scenario"})
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_MAXIMUM_SOURCE_ARTIFACT_BYTES = 16 * 1024 * 1024


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Round 74 target assembly manifest is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _strict_json_object(path: Path, *, label: str) -> dict[str, object]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        selected: dict[str, object] = {}
        for key, value in pairs:
            if key in selected:
                raise ValueError(f"duplicate key: {key}")
            selected[key] = value
        return selected

    def reject_nonfinite(value: str) -> object:
        raise ValueError(f"non-finite value: {value}")

    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Round 74 {label} JSON differs") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Round 74 {label} root differs")
    return value


def _find_key_values(value: object, key: str) -> tuple[object, ...]:
    selected: list[object] = []
    if isinstance(value, Mapping):
        for nested_key, nested in value.items():
            if nested_key == key:
                selected.append(nested)
            selected.extend(_find_key_values(nested, key))
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for nested in value:
            selected.extend(_find_key_values(nested, key))
    return tuple(selected)


def _evidence_sha256_by_kind(value: object) -> dict[str, str]:
    selected: dict[str, str] = {}
    if isinstance(value, Mapping):
        kind = value.get("kind")
        evidence_sha256 = value.get("evidence_sha256")
        if isinstance(kind, str) and _is_sha256(evidence_sha256):
            prior = selected.setdefault(kind, str(evidence_sha256))
            if prior != evidence_sha256:
                raise ValueError(
                    "Round 74 source artifact evidence digest is ambiguous"
                )
        for nested in value.values():
            for nested_kind, digest in _evidence_sha256_by_kind(nested).items():
                prior = selected.setdefault(nested_kind, digest)
                if prior != digest:
                    raise ValueError(
                        "Round 74 source artifact evidence digest is ambiguous"
                    )
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for nested in value:
            for nested_kind, digest in _evidence_sha256_by_kind(nested).items():
                prior = selected.setdefault(nested_kind, digest)
                if prior != digest:
                    raise ValueError(
                        "Round 74 source artifact evidence digest is ambiguous"
                    )
    return selected


def _audit_execution_scenario_artifact(
    value: Mapping[str, object],
    *,
    run_id: str,
) -> None:
    scenario = dict(value)
    scenario_sha256 = str(scenario.pop("scenario_sha256", ""))
    authority = scenario.get("authority")
    upstream = scenario.get("upstream_testnet_calibration")
    panel = scenario.get("scenario_panel")
    source_ids = {
        str(item)
        for item in _find_key_values(scenario, "source_id")
        if isinstance(item, str)
    }
    selected_rows = (
        [
            row
            for row in panel
            if isinstance(row, Mapping) and row.get("selected") is True
        ]
        if isinstance(panel, list)
        else []
    )
    if (
        scenario.get("schema_version")
        != ROUND74_PUBLIC_EXECUTION_SCENARIO_SCHEMA_VERSION
        or scenario.get("environment") != "binance_usdm_mainnet"
        or scenario.get("run_id") != run_id
        or scenario.get("scenario_contract_sha256")
        != ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT_SHA256
        or scenario.get("selected_scenario")
        != ROUND74_PUBLIC_EXECUTION_SCENARIO_SELECTED_NAME
        or scenario_sha256 != _canonical_sha256(scenario)
        or source_ids
        != {
            ROUND74_PUBLIC_EXECUTION_SCENARIO_LATENCY_SOURCE_ID,
            ROUND74_PUBLIC_EXECUTION_SCENARIO_SLIPPAGE_SOURCE_ID,
        }
        or not isinstance(authority, Mapping)
        or authority.get("testnet_execution_equivalence") is not False
        or authority.get("mainnet_fill_evidence") is not False
        or authority.get("profitability_claim") is not False
        or authority.get("orders_submitted") is not False
        or authority.get("live_trading_authority") is not False
        or not isinstance(upstream, Mapping)
        or upstream.get("source_venue") != "binance_usdm_testnet"
        or upstream.get("mainnet_execution_equivalence") is not False
        or upstream.get("mainnet_transfer_permitted") is not False
        or len(selected_rows) != 1
        or selected_rows[0].get("name")
        != ROUND74_PUBLIC_EXECUTION_SCENARIO_SELECTED_NAME
        or selected_rows[0].get("exact_future_public_l2_replay_required") is not True
        or selected_rows[0].get("mainnet_fill_estimate") is not False
    ):
        raise ValueError("Round 74 target source scenario artifact differs")


@dataclass(frozen=True)
class Round74TargetSourceArtifactBinding:
    """Expected identity and evidence carried by one source artifact."""

    label: str
    relative_path: str
    artifact_file_sha256: str
    artifact_sha256: str
    environment: str
    evidence_sha256_by_kind: tuple[tuple[str, str], ...]
    run_id_bound: bool

    def validate(self) -> None:
        relative = PurePosixPath(self.relative_path)
        expected_kinds = _EVIDENCE_KINDS_BY_LABEL.get(self.label)
        evidence = dict(self.evidence_sha256_by_kind)
        if (
            expected_kinds is None
            or relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.suffix != ".json"
            or "\\" in self.relative_path
            or not _is_sha256(self.artifact_file_sha256)
            or not _is_sha256(self.artifact_sha256)
            or not self.environment.strip()
            or tuple(evidence) != expected_kinds
            or len(evidence) != len(self.evidence_sha256_by_kind)
            or any(not _is_sha256(digest) for digest in evidence.values())
            or self.run_id_bound != (self.label in _RUN_BOUND_LABELS)
            or (
                self.label == "execution_calibration"
                and self.environment != "binance_usdm_testnet"
            )
        ):
            raise ValueError("Round 74 target source artifact binding differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "label": self.label,
            "relative_path": self.relative_path,
            "artifact_file_sha256": self.artifact_file_sha256,
            "artifact_sha256": self.artifact_sha256,
            "environment": self.environment,
            "evidence_sha256_by_kind": dict(self.evidence_sha256_by_kind),
            "run_id_bound": self.run_id_bound,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74TargetSourceArtifactBinding:
        evidence = value.get("evidence_sha256_by_kind")
        if set(value) != {
            "label",
            "relative_path",
            "artifact_file_sha256",
            "artifact_sha256",
            "environment",
            "evidence_sha256_by_kind",
            "run_id_bound",
        } or not isinstance(evidence, Mapping):
            raise ValueError("Round 74 target source artifact payload differs")
        selected = cls(
            label=str(value["label"]),
            relative_path=str(value["relative_path"]),
            artifact_file_sha256=str(value["artifact_file_sha256"]),
            artifact_sha256=str(value["artifact_sha256"]),
            environment=str(value["environment"]),
            evidence_sha256_by_kind=tuple(
                (str(kind), str(digest)) for kind, digest in evidence.items()
            ),
            run_id_bound=value["run_id_bound"] is True,
        )
        selected.validate()
        if selected.as_dict() != dict(value):
            raise ValueError("Round 74 target source artifact payload differs")
        return selected


@dataclass(frozen=True)
class Round74TargetAssemblyManifest:
    """One target assembly rebound to its cohort and source artifacts."""

    run_id: str
    cohort_binding_sha256: str
    scenario_contract_sha256: str
    assembly: Round74SourceTargetAssembly
    source_artifacts: tuple[Round74TargetSourceArtifactBinding, ...]
    schema_version: str = ROUND74_TARGET_ASSEMBLY_MANIFEST_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != ROUND74_TARGET_ASSEMBLY_MANIFEST_SCHEMA_VERSION
            or len(self.run_id) != 32
            or any(character not in _SHA256_CHARACTERS for character in self.run_id)
            or not _is_sha256(self.cohort_binding_sha256)
            or not _is_sha256(self.scenario_contract_sha256)
            or not isinstance(self.assembly, Round74SourceTargetAssembly)
            or tuple(binding.label for binding in self.source_artifacts)
            != ROUND74_TARGET_SOURCE_LABELS
        ):
            raise ValueError("Round 74 target assembly manifest identity differs")
        for binding in self.source_artifacts:
            binding.validate()
        by_label = {binding.label: binding for binding in self.source_artifacts}
        target_environment = self.assembly.spec.execution_environment
        if any(
            by_label[label].environment != target_environment
            for label in (
                "exchange_info",
                "commission",
                "funding",
                "execution_scenario",
            )
        ):
            raise ValueError("Round 74 target assembly source environment differs")
        expected_assembly_evidence = {
            "quantity_rules": (
                self.assembly.spec.quantity_rules_evidence.evidence_sha256
            ),
            "commission": self.assembly.spec.commission_evidence.evidence_sha256,
            "funding_schedule": (
                self.assembly.spec.funding_schedule_evidence.evidence_sha256
            ),
            "entry_exit_latency": (
                self.assembly.spec.entry_exit_latency_evidence.evidence_sha256
            ),
            "residual_slippage": (self.assembly.spec.slippage_evidence.evidence_sha256),
        }
        observed_assembly_evidence = {
            **dict(by_label["exchange_info"].evidence_sha256_by_kind),
            **dict(by_label["commission"].evidence_sha256_by_kind),
            **dict(by_label["funding"].evidence_sha256_by_kind),
            **dict(by_label["execution_scenario"].evidence_sha256_by_kind),
        }
        if observed_assembly_evidence != expected_assembly_evidence:
            raise ValueError("Round 74 target assembly source evidence differs")

    @property
    def manifest_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "cohort_binding_sha256": self.cohort_binding_sha256,
            "scenario_contract_sha256": self.scenario_contract_sha256,
            "assembly": self.assembly.as_dict(),
            "source_artifacts": [
                binding.as_dict() for binding in self.source_artifacts
            ],
        }
        if include_sha256:
            value["manifest_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74TargetAssemblyManifest:
        payload = dict(value)
        claimed = str(payload.pop("manifest_sha256", ""))
        raw_assembly = payload.get("assembly")
        raw_sources = payload.get("source_artifacts")
        if (
            set(payload)
            != {
                "schema_version",
                "run_id",
                "cohort_binding_sha256",
                "scenario_contract_sha256",
                "assembly",
                "source_artifacts",
            }
            or not isinstance(raw_assembly, Mapping)
            or not isinstance(raw_sources, list)
            or any(not isinstance(row, Mapping) for row in raw_sources)
        ):
            raise ValueError("Round 74 target assembly manifest payload differs")
        selected = cls(
            run_id=str(payload["run_id"]),
            cohort_binding_sha256=str(payload["cohort_binding_sha256"]),
            scenario_contract_sha256=str(payload["scenario_contract_sha256"]),
            assembly=Round74SourceTargetAssembly.from_dict(raw_assembly),
            source_artifacts=tuple(
                Round74TargetSourceArtifactBinding.from_dict(row) for row in raw_sources
            ),
            schema_version=str(payload["schema_version"]),
        )
        selected.validate()
        if (
            claimed != selected.manifest_sha256
            or selected.as_dict(include_sha256=False) != payload
        ):
            raise ValueError("Round 74 target assembly manifest digest differs")
        return selected


def audit_round74_target_assembly_manifest(
    manifest: Round74TargetAssemblyManifest,
    *,
    source_artifact_root: str | Path,
) -> Round74SourceTargetAssembly:
    """Reopen every source artifact and verify its exact manifest binding."""

    manifest.validate()
    selected_root = Path(source_artifact_root)
    if selected_root.is_symlink() or not selected_root.is_dir():
        raise ValueError("Round 74 target source artifact root differs")
    root = selected_root.resolve()
    for binding in manifest.source_artifacts:
        path = root.joinpath(*PurePosixPath(binding.relative_path).parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Round 74 target source artifact file differs")
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ValueError("Round 74 target source artifact escapes root")
        if resolved.stat().st_size > _MAXIMUM_SOURCE_ARTIFACT_BYTES:
            raise ValueError("Round 74 target source artifact size differs")
        raw = resolved.read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding.artifact_file_sha256:
            raise ValueError("Round 74 target source artifact file digest differs")
        payload = _strict_json_object(
            resolved,
            label=f"{binding.label} source artifact",
        )
        claimed = str(payload.pop("artifact_sha256", ""))
        if claimed != binding.artifact_sha256 or claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 target source artifact digest differs")
        observed_evidence = _evidence_sha256_by_kind(payload)
        if observed_evidence != dict(binding.evidence_sha256_by_kind):
            raise ValueError("Round 74 target source artifact evidence differs")
        environments = {
            str(value)
            for value in _find_key_values(payload, "environment")
            if isinstance(value, str)
        }
        if environments != {binding.environment}:
            raise ValueError("Round 74 target source artifact environment differs")
        run_ids = {
            str(value)
            for value in _find_key_values(payload, "run_id")
            if isinstance(value, str)
        }
        if binding.run_id_bound and run_ids != {manifest.run_id}:
            raise ValueError("Round 74 target source artifact run differs")
        if binding.label == "execution_scenario" and (
            manifest.scenario_contract_sha256
            not in {
                str(value)
                for value in _find_key_values(payload, "scenario_contract_sha256")
                if isinstance(value, str)
            }
        ):
            raise ValueError("Round 74 target source scenario contract differs")
        if binding.label == "execution_scenario":
            _audit_execution_scenario_artifact(
                payload,
                run_id=manifest.run_id,
            )
    return manifest.assembly


def load_and_audit_round74_target_assembly_manifest(
    *,
    manifest_path: str | Path,
    source_artifact_root: str | Path,
) -> Round74TargetAssemblyManifest:
    """Load one strict manifest, then reopen and audit every source."""

    selected_path = Path(manifest_path)
    if selected_path.is_symlink() or not selected_path.is_file():
        raise ValueError("Round 74 target assembly manifest file differs")
    manifest = Round74TargetAssemblyManifest.from_dict(
        _strict_json_object(
            selected_path,
            label="target assembly manifest",
        )
    )
    audit_round74_target_assembly_manifest(
        manifest,
        source_artifact_root=source_artifact_root,
    )
    return manifest


__all__ = [
    "ROUND74_TARGET_ASSEMBLY_MANIFEST_SCHEMA_VERSION",
    "ROUND74_TARGET_SOURCE_LABELS",
    "Round74TargetAssemblyManifest",
    "Round74TargetSourceArtifactBinding",
    "audit_round74_target_assembly_manifest",
    "load_and_audit_round74_target_assembly_manifest",
]
