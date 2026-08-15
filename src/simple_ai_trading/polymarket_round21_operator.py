"""Executable, evidence-bound development assembly for Polymarket Round 21."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_resolution import load_official_resolutions
from .polymarket_round21_core_features import (
    POLYMARKET_ROUND21_FEATURE_SCHEMA,
    Round21CoreFeatureSnapshot,
)
from .polymarket_round21_corpus_store import (
    load_round21_core_development_publication,
)
from .polymarket_round21_dataset import (
    Round21CausalFeatureRow,
    Round21OfficialOutcome,
    Round21PartitionPolicy,
    build_round21_development_panel,
)
from .polymarket_round21_model import (
    Round21DevelopmentPanel,
    fit_round21_development,
    round21_development_dataset_identity,
)
from .polymarket_round21_ablation import (
    evaluate_round21_probability_basis_ablation,
    load_round21_probability_basis_ablation_design,
    validate_round21_probability_basis_ablation_result,
)
from .polymarket_round21_sidecar_replay import (
    Round21SidecarReplay,
    replay_round21_optional_binance_features,
)
from .polymarket_round21_sidecar_terminal import (
    validate_round21_sidecar_terminal_manifest,
)
from .polymarket_round21_terminal import (
    validate_round21_terminal_transport_manifest,
)


_DEVELOPMENT_ROLES = ("train", "tune_calibration", "tune_selection")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class Round21CoreDevelopmentAssembly:
    """Audited core baseline inputs; optional predictor comparison remains separate."""

    publication_manifest_sha256: str
    terminal_transport_manifest_sha256: str
    partition_policy: Round21PartitionPolicy
    train: Round21DevelopmentPanel
    tune_calibration: Round21DevelopmentPanel
    tune_selection: Round21DevelopmentPanel
    outcome_count: int
    population_layer: str = "core"
    profitability_claim: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    def validated(self) -> Round21CoreDevelopmentAssembly:
        policy = self.partition_policy.validated()
        train = self.train.validate()
        calibration = self.tune_calibration.validate()
        selection = self.tune_selection.validate()
        if (
            self.population_layer != "core"
            or train.role != "train"
            or calibration.role != "tune_calibration"
            or selection.role != "tune_selection"
            or self.outcome_count
            != len(
                set(train.condition_ids)
                | set(calibration.condition_ids)
                | set(selection.condition_ids)
            )
            or _SHA256.fullmatch(self.publication_manifest_sha256) is None
            or _SHA256.fullmatch(self.terminal_transport_manifest_sha256) is None
            or any(
                value
                for value in (
                    self.profitability_claim,
                    self.paper_trading_authority,
                    self.live_trading_authority,
                )
            )
        ):
            raise ValueError("Round 21 core development assembly differs")
        return Round21CoreDevelopmentAssembly(
            publication_manifest_sha256=self.publication_manifest_sha256,
            terminal_transport_manifest_sha256=(
                self.terminal_transport_manifest_sha256
            ),
            partition_policy=policy,
            train=train,
            tune_calibration=calibration,
            tune_selection=selection,
            outcome_count=self.outcome_count,
        )


@dataclass(frozen=True, slots=True)
class Round21MatchedDevelopmentAssembly:
    """Exact core rows plus optional public Binance features on matched decisions."""

    publication_manifest_sha256: str
    terminal_transport_manifest_sha256: str
    sidecar_terminal_manifest_sha256: str
    sidecar_receipt_chain_sha256: str
    partition_policy: Round21PartitionPolicy
    train: Round21DevelopmentPanel
    tune_calibration: Round21DevelopmentPanel
    tune_selection: Round21DevelopmentPanel
    outcome_count: int
    sidecar_raw_message_count: int
    sidecar_stream_gap_count: int
    population_layer: str = "matched_core_and_optional_binance"
    profitability_claim: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    def validated(self) -> Round21MatchedDevelopmentAssembly:
        policy = self.partition_policy.validated()
        train = self.train.validate()
        calibration = self.tune_calibration.validate()
        selection = self.tune_selection.validate()
        if (
            self.population_layer != "matched_core_and_optional_binance"
            or train.role != "train"
            or calibration.role != "tune_calibration"
            or selection.role != "tune_selection"
            or self.outcome_count
            != len(
                set(train.condition_ids)
                | set(calibration.condition_ids)
                | set(selection.condition_ids)
            )
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.publication_manifest_sha256,
                    self.terminal_transport_manifest_sha256,
                    self.sidecar_terminal_manifest_sha256,
                    self.sidecar_receipt_chain_sha256,
                )
            )
            or self.sidecar_raw_message_count <= 0
            or self.sidecar_stream_gap_count < 0
            or not any(train.spot_available)
            or not any(calibration.spot_available)
            or not any(selection.spot_available)
            or any(
                value
                for value in (
                    self.profitability_claim,
                    self.paper_trading_authority,
                    self.live_trading_authority,
                )
            )
        ):
            raise ValueError("Round 21 matched development assembly differs")
        return Round21MatchedDevelopmentAssembly(
            publication_manifest_sha256=self.publication_manifest_sha256,
            terminal_transport_manifest_sha256=(
                self.terminal_transport_manifest_sha256
            ),
            sidecar_terminal_manifest_sha256=(self.sidecar_terminal_manifest_sha256),
            sidecar_receipt_chain_sha256=self.sidecar_receipt_chain_sha256,
            partition_policy=policy,
            train=train,
            tune_calibration=calibration,
            tune_selection=selection,
            outcome_count=self.outcome_count,
            sidecar_raw_message_count=self.sidecar_raw_message_count,
            sidecar_stream_gap_count=self.sidecar_stream_gap_count,
        )


def build_round21_core_causal_rows(
    snapshots: Sequence[Round21CoreFeatureSnapshot],
) -> tuple[Round21CausalFeatureRow, ...]:
    """Convert exact core snapshots without inventing optional observations."""

    selected = tuple(snapshots)
    if not selected:
        raise ValueError("Round 21 core snapshot population is empty")
    rows: list[Round21CausalFeatureRow] = []
    keys: set[tuple[str, int]] = set()
    for snapshot in selected:
        if (
            not isinstance(snapshot, Round21CoreFeatureSnapshot)
            or not snapshot.available
            or snapshot.trading_authority
        ):
            raise ValueError("Round 21 core snapshot population differs")
        key = (snapshot.condition_id, snapshot.decision_time_ms)
        if key in keys:
            raise ValueError("Round 21 core snapshot population contains duplicates")
        keys.add(key)
        rows.append(
            Round21CausalFeatureRow.create(
                condition_id=snapshot.condition_id,
                event_start_ms=snapshot.event_start_ms,
                decision_time_ms=snapshot.decision_time_ms,
                structural_probability=snapshot.structural_probability,
                market_prior_probability=snapshot.market_prior_probability,
                core_values=snapshot.values,
                spot_values=(0.0,) * len(POLYMARKET_ROUND21_FEATURE_SCHEMA.spot_names),
                usdm_values=(0.0,) * len(POLYMARKET_ROUND21_FEATURE_SCHEMA.usdm_names),
                spot_available=False,
                usdm_available=False,
                feature_schema=POLYMARKET_ROUND21_FEATURE_SCHEMA,
                core_source_chain_sha256=snapshot.source_chain_sha256,
                spot_source_chain_sha256=_EMPTY_SHA256,
                usdm_source_chain_sha256=_EMPTY_SHA256,
                core_maximum_receipt_ms=snapshot.maximum_receipt_ms,
            )
        )
    return tuple(
        sorted(rows, key=lambda item: (item.event_start_ms, item.decision_time_ms))
    )


def apply_round21_optional_binance_features(
    core_rows: Sequence[Round21CausalFeatureRow],
    replay: Round21SidecarReplay,
) -> tuple[Round21CausalFeatureRow, ...]:
    """Join one target-blind optional snapshot to each exact core decision."""

    selected_rows = tuple(core_rows)
    selected_replay = replay.validated()
    if (
        not selected_rows
        or len(selected_rows) != len(selected_replay.features)
        or tuple(row.decision_time_ms for row in selected_rows)
        != selected_replay.decision_times_ms
        or any(row.spot_available or row.usdm_available for row in selected_rows)
    ):
        raise ValueError("Round 21 optional Binance join population differs")
    output: list[Round21CausalFeatureRow] = []
    for row, optional in zip(
        selected_rows,
        selected_replay.features,
        strict=True,
    ):
        if optional.trading_authority:
            raise ValueError("Round 21 optional Binance join authority differs")
        output.append(
            Round21CausalFeatureRow.create(
                condition_id=row.condition_id,
                event_start_ms=row.event_start_ms,
                decision_time_ms=row.decision_time_ms,
                structural_probability=row.structural_probability,
                market_prior_probability=row.market_prior_probability,
                core_values=row.core_values,
                spot_values=optional.spot_values,
                usdm_values=optional.usdm_values,
                spot_available=optional.spot_available,
                usdm_available=optional.usdm_available,
                feature_schema=POLYMARKET_ROUND21_FEATURE_SCHEMA,
                core_source_chain_sha256=row.core_source_chain_sha256,
                spot_source_chain_sha256=optional.spot_source_chain_sha256,
                usdm_source_chain_sha256=optional.usdm_source_chain_sha256,
                core_maximum_receipt_ms=row.core_maximum_receipt_ms,
                spot_maximum_receipt_ms=optional.spot_maximum_receipt_ms,
                usdm_maximum_receipt_ms=optional.usdm_maximum_receipt_ms,
            )
        )
    return tuple(output)


def load_round21_official_outcomes(
    *,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    condition_event_starts: Mapping[str, int],
) -> tuple[Round21OfficialOutcome, ...]:
    """Load exact CLOB/Gamma consensus outcomes for admitted conditions only."""

    transport = validate_round21_terminal_transport_manifest(
        terminal_transport_manifest
    )
    expected = {str(key): int(value) for key, value in condition_event_starts.items()}
    if not expected:
        raise ValueError("Round 21 official outcome population is empty")
    run_ids = tuple(
        str(segment["run_id"])
        for segment in transport["segments"]
        if segment["eligible_for_condition_rebuild"]
    )
    if not run_ids:
        raise ValueError("Round 21 terminal transport has no eligible run")

    output: dict[str, Round21OfficialOutcome] = {}
    with PolymarketEvidenceStore(
        Path(source_database),
        read_only=True,
        memory_limit="1GB",
        threads=2,
    ) as store:
        for run_id in run_ids:
            for resolution in load_official_resolutions(store, run_id=run_id):
                condition_id = str(resolution.condition_id)
                if condition_id not in expected:
                    continue
                if (
                    condition_id in output
                    or resolution.asset != "BTC"
                    or resolution.winning_outcome not in {"Up", "Down"}
                ):
                    raise ValueError("Round 21 official outcome population differs")
                output[condition_id] = Round21OfficialOutcome.create(
                    condition_id=condition_id,
                    event_start_ms=expected[condition_id],
                    resolved_up=resolution.winning_outcome == "Up",
                    observed_at_ms=resolution.observed_wall_ms,
                    source="polymarket_clob_gamma_consensus",
                    source_payload_sha256=resolution.evidence_sha256,
                )
    if set(output) != set(expected):
        raise ValueError("Round 21 official outcome population is incomplete")
    return tuple(
        output[condition_id]
        for condition_id in sorted(
            output,
            key=lambda value: (expected[value], value),
        )
    )


def _build_role_panels(
    *,
    rows: Sequence[Round21CausalFeatureRow],
    outcomes: Sequence[Round21OfficialOutcome],
    policy: Round21PartitionPolicy,
) -> dict[str, Round21DevelopmentPanel]:
    panels: dict[str, Round21DevelopmentPanel] = {}
    for role in _DEVELOPMENT_ROLES:
        role_rows = tuple(
            row
            for row in rows
            if policy.role_for_event_start(row.event_start_ms) == role
        )
        role_conditions = {row.condition_id for row in role_rows}
        role_outcomes = tuple(
            outcome for outcome in outcomes if outcome.condition_id in role_conditions
        )
        panels[role] = build_round21_development_panel(
            role=role,
            feature_schema=POLYMARKET_ROUND21_FEATURE_SCHEMA,
            partition_policy=policy,
            feature_rows=role_rows,
            outcomes=role_outcomes,
        )
    return panels


def _load_round21_core_development_evidence(
    *,
    publication_directory: str | Path,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
) -> tuple[
    Mapping[str, object],
    Round21PartitionPolicy,
    tuple[Round21CausalFeatureRow, ...],
    tuple[Round21OfficialOutcome, ...],
]:
    transport = validate_round21_terminal_transport_manifest(
        terminal_transport_manifest
    )
    publication, snapshots = load_round21_core_development_publication(
        publication_directory
    )
    if (
        publication["terminal_transport_manifest_sha256"]
        != transport["manifest_sha256"]
    ):
        raise ValueError("Round 21 publication and terminal transport differ")
    policy = Round21PartitionPolicy.create(
        campaign_start_ms=int(
            publication["development_partition"]["campaign_start_ms"]
        ),
        campaign_end_ms=int(publication["development_partition"]["campaign_end_ms"]),
    )
    if (
        publication["development_partition"]["partition_policy_sha256"]
        != policy.policy_sha256
    ):
        raise ValueError("Round 21 development partition policy differs")
    rows = build_round21_core_causal_rows(snapshots)
    identities: dict[str, int] = {}
    for row in rows:
        previous = identities.setdefault(row.condition_id, row.event_start_ms)
        if previous != row.event_start_ms:
            raise ValueError("Round 21 core condition identity differs")
    outcomes = load_round21_official_outcomes(
        source_database=source_database,
        terminal_transport_manifest=transport,
        condition_event_starts=identities,
    )
    return publication, policy, rows, outcomes


def assemble_round21_core_development(
    *,
    publication_directory: str | Path,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
) -> Round21CoreDevelopmentAssembly:
    """Build the three frozen development panels from verified stored evidence."""

    publication, policy, rows, outcomes = _load_round21_core_development_evidence(
        publication_directory=publication_directory,
        source_database=source_database,
        terminal_transport_manifest=terminal_transport_manifest,
    )
    panels = _build_role_panels(rows=rows, outcomes=outcomes, policy=policy)
    return Round21CoreDevelopmentAssembly(
        publication_manifest_sha256=str(publication["manifest_sha256"]),
        terminal_transport_manifest_sha256=str(
            publication["terminal_transport_manifest_sha256"]
        ),
        partition_policy=policy,
        train=panels["train"],
        tune_calibration=panels["tune_calibration"],
        tune_selection=panels["tune_selection"],
        outcome_count=len(outcomes),
    ).validated()


def assemble_round21_matched_development(
    *,
    publication_directory: str | Path,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    sidecar_database: str | Path,
    sidecar_terminal_manifest: Mapping[str, object],
) -> Round21MatchedDevelopmentAssembly:
    """Assemble matched core and optional predictor panels in one sidecar scan."""

    publication, policy, core_rows, outcomes = _load_round21_core_development_evidence(
        publication_directory=publication_directory,
        source_database=source_database,
        terminal_transport_manifest=terminal_transport_manifest,
    )
    sidecar_terminal = validate_round21_sidecar_terminal_manifest(
        sidecar_terminal_manifest
    )
    if (
        int(sidecar_terminal["campaign_start_ms"]) != policy.campaign_start_ms
        or int(sidecar_terminal["campaign_end_ms"]) != policy.campaign_end_ms
    ):
        raise ValueError("Round 21 core and sidecar campaign boundaries differ")
    replay = replay_round21_optional_binance_features(
        source_database=sidecar_database,
        terminal_manifest=sidecar_terminal,
        decision_times_ms=tuple(row.decision_time_ms for row in core_rows),
    )
    rows = apply_round21_optional_binance_features(core_rows, replay)
    panels = _build_role_panels(rows=rows, outcomes=outcomes, policy=policy)
    return Round21MatchedDevelopmentAssembly(
        publication_manifest_sha256=str(publication["manifest_sha256"]),
        terminal_transport_manifest_sha256=str(
            publication["terminal_transport_manifest_sha256"]
        ),
        sidecar_terminal_manifest_sha256=replay.terminal_manifest_sha256,
        sidecar_receipt_chain_sha256=replay.receipt_chain_sha256,
        partition_policy=policy,
        train=panels["train"],
        tune_calibration=panels["tune_calibration"],
        tune_selection=panels["tune_selection"],
        outcome_count=len(outcomes),
        sidecar_raw_message_count=replay.raw_message_count,
        sidecar_stream_gap_count=replay.stream_gap_count,
    ).validated()


def fit_round21_core_baseline(
    *,
    publication_directory: str | Path,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    basis_ablation_result: Mapping[str, object],
    compute_backend: str = "auto",
) -> dict[str, object]:
    """Fit the frozen core baseline; this is not the optional-layer comparison."""

    assembly = assemble_round21_core_development(
        publication_directory=publication_directory,
        source_database=source_database,
        terminal_transport_manifest=terminal_transport_manifest,
    )
    accepted_basis = _require_accepted_round21_probability_basis(
        basis_ablation_result,
        assembly=assembly,
        require_exact_dataset_identity=True,
    )
    return fit_round21_development(
        train=assembly.train,
        tune_calibration=assembly.tune_calibration,
        tune_selection=assembly.tune_selection,
        basis_ablation_result=accepted_basis,
        compute_backend=compute_backend,
        feature_layers=("core",),
    )


def evaluate_round21_core_probability_basis(
    *,
    repository: str | Path,
    publication_directory: str | Path,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Run the preregistered six-fit basis gate on audited development rows."""

    load_round21_probability_basis_ablation_design(repository)
    assembly = assemble_round21_core_development(
        publication_directory=publication_directory,
        source_database=source_database,
        terminal_transport_manifest=terminal_transport_manifest,
    )
    return evaluate_round21_probability_basis_ablation(
        train=assembly.train,
        tune_calibration=assembly.tune_calibration,
        tune_selection=assembly.tune_selection,
        publication_manifest_sha256=assembly.publication_manifest_sha256,
        terminal_transport_manifest_sha256=(
            assembly.terminal_transport_manifest_sha256
        ),
    )


def _require_accepted_round21_probability_basis(
    value: Mapping[str, object],
    *,
    assembly: Round21CoreDevelopmentAssembly | Round21MatchedDevelopmentAssembly,
    require_exact_dataset_identity: bool,
) -> dict[str, object]:
    result = validate_round21_probability_basis_ablation_result(value)
    source = result["source_evidence"]
    datasets = result["dataset_and_partition"]
    if (
        result["basis_accepted"] is not True
        or source["publication_manifest_sha256"] != assembly.publication_manifest_sha256
        or source["terminal_transport_manifest_sha256"]
        != assembly.terminal_transport_manifest_sha256
    ):
        raise ValueError("Round 21 probability-basis gate is not accepted")
    shared_identity_keys = {
        "role",
        "row_count",
        "condition_count",
        "first_event_start_ms",
        "last_event_start_ms",
        "target_manifest_sha256",
        "dataset_design_sha256",
    }
    for role, panel in (
        ("train", assembly.train),
        ("tune_calibration", assembly.tune_calibration),
        ("tune_selection", assembly.tune_selection),
    ):
        expected = round21_development_dataset_identity(panel)
        observed = datasets[role]
        matches = (
            observed == expected
            if require_exact_dataset_identity
            else all(observed[key] == expected[key] for key in shared_identity_keys)
        )
        if not matches:
            raise ValueError("Round 21 probability-basis dataset identity differs")
    return result


def fit_round21_matched_optional_candidate(
    *,
    publication_directory: str | Path,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    sidecar_database: str | Path,
    sidecar_terminal_manifest: Mapping[str, object],
    basis_ablation_result: Mapping[str, object],
    compute_backend: str = "auto",
) -> dict[str, object]:
    """Fit core and optional layers on the frozen matched development population."""

    assembly = assemble_round21_matched_development(
        publication_directory=publication_directory,
        source_database=source_database,
        terminal_transport_manifest=terminal_transport_manifest,
        sidecar_database=sidecar_database,
        sidecar_terminal_manifest=sidecar_terminal_manifest,
    )
    accepted_basis = _require_accepted_round21_probability_basis(
        basis_ablation_result,
        assembly=assembly,
        require_exact_dataset_identity=False,
    )
    return fit_round21_development(
        train=assembly.train,
        tune_calibration=assembly.tune_calibration,
        tune_selection=assembly.tune_selection,
        basis_ablation_result=accepted_basis,
        compute_backend=compute_backend,
    )


__all__ = [
    "Round21CoreDevelopmentAssembly",
    "Round21MatchedDevelopmentAssembly",
    "apply_round21_optional_binance_features",
    "assemble_round21_core_development",
    "assemble_round21_matched_development",
    "build_round21_core_causal_rows",
    "evaluate_round21_core_probability_basis",
    "fit_round21_core_baseline",
    "fit_round21_matched_optional_candidate",
    "load_round21_official_outcomes",
]
