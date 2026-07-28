"""Offline construction of one fully source-bound Round 74 target manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Mapping

from .impact_absorption_event_targets import (
    Round74EventTargetSpec,
)
from .impact_absorption_execution_scenario import (
    ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT_SHA256,
    load_round74_execution_aggregate_source,
    load_round74_public_execution_scenario_artifact,
)
from .impact_absorption_target_assembly import (
    Round74SourceTargetAssembly,
)
from .round74_public_target_sources import (
    ROUND74_PUBLIC_TARGET_ENVIRONMENT,
    ROUND74_TESTNET_EXECUTION_ENVIRONMENT,
    audit_round74_cohort_capture_source_payload,
    load_round74_canonical_source_artifact,
    parse_round74_commission_source,
    parse_round74_exchange_info_source,
    parse_round74_funding_source,
)
from .round74_target_assembly_manifest import (
    ROUND74_TARGET_SOURCE_LABELS,
    Round74TargetAssemblyManifest,
    Round74TargetSourceArtifactBinding,
    audit_round74_target_assembly_manifest,
)


_MAXIMUM_RELATIVE_PATH_CHARACTERS = 512


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
        raise ValueError("Round 74 target operator output is not canonical") from exc


def _resolve_source(
    root: Path,
    relative_path: str,
    *,
    label: str,
) -> tuple[str, Path]:
    selected = str(relative_path)
    relative = PurePosixPath(selected)
    if (
        not selected
        or len(selected) > _MAXIMUM_RELATIVE_PATH_CHARACTERS
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.suffix != ".json"
        or "\\" in selected
    ):
        raise ValueError(f"Round 74 {label} relative path differs")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Round 74 {label} source file differs")
    resolved = path.resolve()
    if root not in resolved.parents:
        raise ValueError(f"Round 74 {label} source escapes root")
    return relative.as_posix(), resolved


def _binding(
    *,
    label: str,
    relative_path: str,
    artifact_sha256: str,
    artifact_file_sha256: str,
    environment: str,
    evidence_sha256_by_kind: tuple[tuple[str, str], ...],
) -> Round74TargetSourceArtifactBinding:
    selected = Round74TargetSourceArtifactBinding(
        label=label,
        relative_path=relative_path,
        artifact_file_sha256=artifact_file_sha256,
        artifact_sha256=artifact_sha256,
        environment=environment,
        evidence_sha256_by_kind=evidence_sha256_by_kind,
        run_id_bound=label
        in {
            "cohort_capture",
            "funding",
            "execution_scenario",
        },
    )
    selected.validate()
    return selected


def build_round74_public_target_manifest(
    *,
    source_artifact_root: str | Path,
    source_relative_paths: Mapping[str, str],
) -> Round74TargetAssemblyManifest:
    """Reopen six exact artifacts and derive one no-authority target."""

    root_path = Path(source_artifact_root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise ValueError("Round 74 target source root differs")
    root = root_path.resolve()
    if set(source_relative_paths) != set(ROUND74_TARGET_SOURCE_LABELS) or len(
        set(source_relative_paths.values())
    ) != len(ROUND74_TARGET_SOURCE_LABELS):
        raise ValueError("Round 74 target source path panel differs")
    resolved: dict[str, tuple[str, Path]] = {
        label: _resolve_source(
            root,
            source_relative_paths[label],
            label=label,
        )
        for label in ROUND74_TARGET_SOURCE_LABELS
    }

    cohort_artifact = load_round74_canonical_source_artifact(
        resolved["cohort_capture"][1],
        label="cohort capture",
    )
    raw_binding = cohort_artifact.payload.get("cohort_binding")
    if not isinstance(raw_binding, Mapping):
        raise ValueError("Round 74 cohort source binding differs")
    run_id = str(cohort_artifact.payload.get("run_id", ""))
    cohort_binding_sha256 = str(
        cohort_artifact.payload.get("cohort_binding_sha256", "")
    )
    binding = audit_round74_cohort_capture_source_payload(
        cohort_artifact.payload,
        run_id=run_id,
        cohort_binding_sha256=cohort_binding_sha256,
    )

    exchange_artifact = load_round74_canonical_source_artifact(
        resolved["exchange_info"][1],
        label="exchange info",
    )
    exchange = parse_round74_exchange_info_source(exchange_artifact.payload)
    commission_artifact = load_round74_canonical_source_artifact(
        resolved["commission"][1],
        label="commission",
    )
    commission = parse_round74_commission_source(commission_artifact.payload)
    funding_artifact = load_round74_canonical_source_artifact(
        resolved["funding"][1],
        label="funding",
    )
    funding = parse_round74_funding_source(
        funding_artifact.payload,
        run_id=run_id,
    )
    aggregate = load_round74_execution_aggregate_source(
        resolved["execution_calibration"][1]
    )
    aggregate_artifact = load_round74_canonical_source_artifact(
        resolved["execution_calibration"][1],
        label="execution calibration",
    )
    scenario = load_round74_public_execution_scenario_artifact(
        resolved["execution_scenario"][1]
    )
    scenario_artifact = load_round74_canonical_source_artifact(
        resolved["execution_scenario"][1],
        label="execution scenario",
    )

    if (
        scenario.bundle.run_id != run_id
        or scenario.bundle.cohort_binding_sha256 != binding.binding_sha256
        or scenario.bundle.capture_report_sha256 != binding.report_sha256
        or scenario.bundle.testnet_aggregate_artifact_sha256
        != aggregate.artifact_sha256
        or scenario.bundle.testnet_aggregate_artifact_file_sha256
        != aggregate.artifact_file_sha256
        or scenario.bundle.testnet_latency_evidence_sha256
        != aggregate.bundle.entry_exit_latency_evidence.evidence_sha256
        or scenario.bundle.testnet_slippage_evidence_sha256
        != aggregate.bundle.residual_slippage_evidence.evidence_sha256
        or aggregate_artifact.artifact_sha256 != aggregate.artifact_sha256
        or aggregate_artifact.artifact_file_sha256 != aggregate.artifact_file_sha256
        or scenario_artifact.artifact_sha256 != scenario.artifact_sha256
        or scenario_artifact.artifact_file_sha256 != scenario.artifact_file_sha256
    ):
        raise ValueError("Round 74 target execution source chain differs")

    rules = exchange.mapping()
    spec = Round74EventTargetSpec.create(
        reference_quote_notional=scenario.bundle.reference_quote_notional,
        decision_to_entry_latency_ns_by_symbol=(
            scenario.bundle.entry_latency_mapping()
        ),
        decision_to_exit_latency_ns_by_symbol=(scenario.bundle.exit_latency_mapping()),
        taker_fee_bps_by_symbol=commission.mapping(),
        funding_boundary_intervals_monotonic_ns=(funding.boundary_mapping()),
        funding_schedule_coverage_monotonic_ns=(funding.coverage_mapping()),
        additional_slippage_bps_per_side_by_symbol=(scenario.bundle.slippage_mapping()),
        quantity_rules_evidence=exchange.evidence,
        commission_evidence=commission.evidence,
        entry_exit_latency_evidence=(scenario.bundle.entry_exit_latency_evidence),
        slippage_evidence=scenario.bundle.residual_slippage_evidence,
        funding_schedule_evidence=funding.evidence,
    )
    assembly = Round74SourceTargetAssembly(
        spec=spec,
        quantity_rules_by_symbol=tuple(rules.items()),
    )
    sources = (
        _binding(
            label="cohort_capture",
            relative_path=resolved["cohort_capture"][0],
            artifact_sha256=cohort_artifact.artifact_sha256,
            artifact_file_sha256=cohort_artifact.artifact_file_sha256,
            environment=ROUND74_PUBLIC_TARGET_ENVIRONMENT,
            evidence_sha256_by_kind=(),
        ),
        _binding(
            label="exchange_info",
            relative_path=resolved["exchange_info"][0],
            artifact_sha256=exchange_artifact.artifact_sha256,
            artifact_file_sha256=exchange_artifact.artifact_file_sha256,
            environment=ROUND74_PUBLIC_TARGET_ENVIRONMENT,
            evidence_sha256_by_kind=(
                ("quantity_rules", exchange.evidence.evidence_sha256),
            ),
        ),
        _binding(
            label="commission",
            relative_path=resolved["commission"][0],
            artifact_sha256=commission_artifact.artifact_sha256,
            artifact_file_sha256=commission_artifact.artifact_file_sha256,
            environment=ROUND74_PUBLIC_TARGET_ENVIRONMENT,
            evidence_sha256_by_kind=(
                ("commission", commission.evidence.evidence_sha256),
            ),
        ),
        _binding(
            label="funding",
            relative_path=resolved["funding"][0],
            artifact_sha256=funding_artifact.artifact_sha256,
            artifact_file_sha256=funding_artifact.artifact_file_sha256,
            environment=ROUND74_PUBLIC_TARGET_ENVIRONMENT,
            evidence_sha256_by_kind=(
                ("funding_schedule", funding.evidence.evidence_sha256),
            ),
        ),
        _binding(
            label="execution_calibration",
            relative_path=resolved["execution_calibration"][0],
            artifact_sha256=aggregate.artifact_sha256,
            artifact_file_sha256=aggregate.artifact_file_sha256,
            environment=ROUND74_TESTNET_EXECUTION_ENVIRONMENT,
            evidence_sha256_by_kind=(
                (
                    "entry_exit_latency",
                    aggregate.bundle.entry_exit_latency_evidence.evidence_sha256,
                ),
                (
                    "residual_slippage",
                    aggregate.bundle.residual_slippage_evidence.evidence_sha256,
                ),
            ),
        ),
        _binding(
            label="execution_scenario",
            relative_path=resolved["execution_scenario"][0],
            artifact_sha256=scenario.artifact_sha256,
            artifact_file_sha256=scenario.artifact_file_sha256,
            environment=ROUND74_PUBLIC_TARGET_ENVIRONMENT,
            evidence_sha256_by_kind=(
                (
                    "entry_exit_latency",
                    scenario.bundle.entry_exit_latency_evidence.evidence_sha256,
                ),
                (
                    "residual_slippage",
                    scenario.bundle.residual_slippage_evidence.evidence_sha256,
                ),
            ),
        ),
    )
    manifest = Round74TargetAssemblyManifest(
        run_id=run_id,
        cohort_binding_sha256=binding.binding_sha256,
        scenario_contract_sha256=(ROUND74_PUBLIC_EXECUTION_SCENARIO_CONTRACT_SHA256),
        assembly=assembly,
        source_artifacts=sources,
    )
    manifest.validate()
    audit_round74_target_assembly_manifest(
        manifest,
        source_artifact_root=root,
    )
    return manifest


def write_round74_public_target_manifest(
    *,
    manifest: Round74TargetAssemblyManifest,
    output_directory: str | Path,
) -> Path:
    """Write immutable canonical bytes, allowing exact idempotent replay."""

    manifest.validate()
    selected_output = Path(output_directory)
    if selected_output.is_symlink():
        raise ValueError("Round 74 target output directory differs")
    selected_output.mkdir(parents=True, exist_ok=True)
    output = selected_output.resolve()
    encoded = (_canonical_json(manifest.as_dict()) + "\n").encode("ascii")
    target = output / f"{manifest.run_id}.json"
    if target.exists():
        if target.is_symlink() or target.read_bytes() != encoded:
            raise FileExistsError("Round 74 target manifest already differs")
        return target
    temporary = target.with_suffix(".json.tmp")
    if temporary.exists():
        raise FileExistsError("Round 74 target manifest temporary file exists")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
    temporary.replace(target)
    if (
        hashlib.sha256(target.read_bytes()).hexdigest()
        != hashlib.sha256(encoded).hexdigest()
    ):
        raise OSError("Round 74 target manifest write verification failed")
    return target


__all__ = [
    "build_round74_public_target_manifest",
    "write_round74_public_target_manifest",
]
