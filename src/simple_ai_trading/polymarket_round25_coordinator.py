"""Fail-closed, resumable post-capture coordinator for Round 25."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Iterator

from .polymarket_round25_dataset import (
    Round25DevelopmentDataset,
    Round25OfficialResolution,
    Round25ResolutionAuthority,
    build_round25_development_dataset,
    require_round25_dataset_minimum,
)
from .polymarket_round25_evaluation import (
    Round25SelectionAccessStore,
    evaluate_round25_predictive_candidates,
    load_round25_predictive_result,
    write_round25_predictive_result,
)
from .polymarket_round25_economic import (
    load_round25_economic_result,
    replay_round25_development_economics,
    write_round25_economic_result,
)
from .polymarket_round25_joint_features import Round25JointFeatureSnapshot
from .polymarket_round25_joint_store import (
    audit_round25_joint_store,
    load_round25_joint_endpoint_inputs,
    materialize_round25_joint_feature_store,
)
from .polymarket_round25_model_ledger import (
    POLYMARKET_ROUND25_MODEL_IMPLEMENTATION_PATHS,
    fit_round25_model_ledger_coordinated,
    load_round25_model_ledger,
    round25_model_implementation_sha256,
    write_round25_model_ledger,
)
from .polymarket_round25_prediction import (
    freeze_round25_prepared_prediction,
    load_round25_prepared_prediction,
    prepare_round25_target_free_prediction,
    write_round25_prepared_prediction,
)
from .polymarket_round25_resolution_store import (
    Round25ResolutionPublicClient,
    audit_round25_resolution_store,
    collect_round25_resolutions_once,
    finalize_round25_resolution_store,
    initialize_round25_resolution_collection,
    load_round25_fit_resolution_inputs,
    load_round25_selection_resolution_inputs,
)
from .polymarket_round25_tcn import Round25TCNFitProgress
from .polymarket_round25_tcn_store_source import (
    create_round25_store_tcn_fit_sources,
)
from .polymarket_round25_terminal import (
    validate_round25_terminal_transport_manifest,
)


POLYMARKET_ROUND25_COORDINATOR_CONTRACT_SHA256 = (
    "dff318629a03c273ff3f437eca187dca9dc65ae6ad4398b0e78195ef847e30fa"
)
POLYMARKET_ROUND25_COORDINATOR_STATE_SCHEMA_VERSION = (
    "polymarket-round25-post-capture-coordinator-state-v2"
)
POLYMARKET_ROUND25_COORDINATOR_PHASES = (
    "feature_materialized",
    "resolution_collection_pending",
    "resolution_finalized",
    "model_fitted",
    "selection_prediction_frozen",
    "predictive_evaluation_complete",
    "economic_evaluation_complete",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 25 coordinator JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 25 coordinator JSON contains {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class Round25CoordinatorPaths:
    repository: Path
    source_database: Path
    feature_database: Path
    resolution_database: Path
    model_ledger: Path
    prepared_prediction: Path
    selection_access_store: Path
    predictive_result: Path
    economic_result: Path
    state: Path
    lock: Path

    def validated(self) -> Round25CoordinatorPaths:
        paths = (
            self.source_database,
            self.feature_database,
            self.resolution_database,
            self.model_ledger,
            self.prepared_prediction,
            self.selection_access_store,
            self.predictive_result,
            self.economic_result,
            self.state,
            self.lock,
        )
        resolved = tuple(path.resolve(strict=False) for path in paths)
        if (
            not self.repository.is_dir()
            or len(set(resolved)) != len(resolved)
            or any(path.is_symlink() for path in paths)
        ):
            raise ValueError("Round 25 coordinator paths differ")
        return self


def validate_round25_coordinator_state(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("state_sha256", "")).strip().lower()
    expected = {
        "contract_sha256",
        "development_economic_gate_passed",
        "economic_result_sha256",
        "feature_store_manifest_sha256",
        "live_trading_authority",
        "model_data_eligible",
        "model_ledger_sha256",
        "paper_trading_authority",
        "phase",
        "predictive_gate_passed",
        "predictive_result_sha256",
        "prepared_prediction_sha256",
        "profitability_claim",
        "resolution_pending_count",
        "resolution_store_manifest_sha256",
        "schema_version",
        "selection_access_status",
        "updated_at_ms",
    }
    nullable_hashes = (
        "feature_store_manifest_sha256",
        "resolution_store_manifest_sha256",
        "model_ledger_sha256",
        "prepared_prediction_sha256",
        "predictive_result_sha256",
        "economic_result_sha256",
    )
    if (
        set(payload) != expected
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND25_COORDINATOR_STATE_SCHEMA_VERSION
        or payload.get("contract_sha256")
        != POLYMARKET_ROUND25_COORDINATOR_CONTRACT_SHA256
        or payload.get("phase") not in POLYMARKET_ROUND25_COORDINATOR_PHASES
        or type(payload.get("updated_at_ms")) is not int
        or payload["updated_at_ms"] <= 0
        or any(
            payload.get(field) is not None
            and _SHA256.fullmatch(str(payload[field])) is None
            for field in nullable_hashes
        )
        or type(payload.get("resolution_pending_count")) is not int
        or payload["resolution_pending_count"] < 0
        or payload.get("selection_access_status")
        not in {None, "prediction_panel_frozen", "target_access_consumed"}
        or type(payload.get("predictive_gate_passed")) is not bool
        or type(payload.get("development_economic_gate_passed")) is not bool
        or any(
            payload.get(field) is not False
            for field in (
                "model_data_eligible",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
    ):
        raise ValueError("Round 25 coordinator state differs")
    phase_index = POLYMARKET_ROUND25_COORDINATOR_PHASES.index(str(payload["phase"]))
    required_by_phase = (
        "feature_store_manifest_sha256",
        "resolution_store_manifest_sha256",
        "model_ledger_sha256",
        "prepared_prediction_sha256",
        "predictive_result_sha256",
        "economic_result_sha256",
    )
    required_count = max(1, phase_index)
    if (
        any(payload[field] is None for field in required_by_phase[:required_count])
        or phase_index == 1 and payload["resolution_pending_count"] <= 0
        or phase_index >= 2 and payload["resolution_pending_count"] != 0
        or phase_index >= 4 and payload["selection_access_status"] is None
        or phase_index < 4 and payload["selection_access_status"] is not None
        or phase_index < 5 and payload["predictive_gate_passed"]
        or phase_index < 6 and payload["development_economic_gate_passed"]
        or phase_index >= 5
        and payload["selection_access_status"] != "target_access_consumed"
        or phase_index == 6 and not payload["predictive_gate_passed"]
    ):
        raise ValueError("Round 25 coordinator phase evidence differs")
    return {**payload, "state_sha256": claimed}


def load_round25_coordinator_state(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 64 * 1024:
        raise ValueError("Round 25 coordinator state file differs")
    try:
        decoded = json.loads(
            source.read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 25 coordinator state is unreadable") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("Round 25 coordinator state is not an object")
    return validate_round25_coordinator_state(decoded)


def _write_state(path: Path, body: Mapping[str, object]) -> dict[str, object]:
    state = validate_round25_coordinator_state(
        {**body, "state_sha256": _canonical_sha256(body)}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError("Round 25 coordinator state temporary path differs")
    payload = (_canonical_json(state) + "\n").encode("ascii")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return state


def _state_body(
    *,
    phase: str,
    updated_at_ms: int,
    feature_store_manifest_sha256: str | None,
    resolution_store_manifest_sha256: str | None = None,
    model_ledger_sha256: str | None = None,
    prepared_prediction_sha256: str | None = None,
    selection_access_status: str | None = None,
    predictive_result_sha256: str | None = None,
    economic_result_sha256: str | None = None,
    resolution_pending_count: int = 0,
    predictive_gate_passed: bool = False,
    development_economic_gate_passed: bool = False,
) -> dict[str, object]:
    return {
        "contract_sha256": POLYMARKET_ROUND25_COORDINATOR_CONTRACT_SHA256,
        "development_economic_gate_passed": development_economic_gate_passed,
        "economic_result_sha256": economic_result_sha256,
        "feature_store_manifest_sha256": feature_store_manifest_sha256,
        "live_trading_authority": False,
        "model_data_eligible": False,
        "model_ledger_sha256": model_ledger_sha256,
        "paper_trading_authority": False,
        "phase": phase,
        "predictive_gate_passed": predictive_gate_passed,
        "predictive_result_sha256": predictive_result_sha256,
        "prepared_prediction_sha256": prepared_prediction_sha256,
        "profitability_claim": False,
        "resolution_pending_count": resolution_pending_count,
        "resolution_store_manifest_sha256": resolution_store_manifest_sha256,
        "schema_version": POLYMARKET_ROUND25_COORDINATOR_STATE_SCHEMA_VERSION,
        "selection_access_status": selection_access_status,
        "updated_at_ms": updated_at_ms,
    }


@contextmanager
def _coordinator_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError("Round 25 coordinator is already running") from exc
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _emit(
    progress: ProgressCallback | None,
    stage: str,
    details: Mapping[str, object],
) -> None:
    if progress is not None:
        progress(stage, details)


def _verified_source_identity(
    repository: Path,
    *,
    source_commit_oid: str,
) -> tuple[tuple[str, str], ...]:
    if _COMMIT.fullmatch(source_commit_oid) is None:
        raise ValueError("Round 25 fitted source commit differs")
    implementation = round25_model_implementation_sha256(repository)
    if (repository / ".git").exists():
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip().lower()
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *POLYMARKET_ROUND25_MODEL_IMPLEMENTATION_PATHS,
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        if head != source_commit_oid or status:
            raise ValueError("Round 25 fitted implementation is not commit-clean")
    return implementation


def _role_dataset(
    *,
    role: str,
    snapshots: Sequence[Round25JointFeatureSnapshot],
    resolutions: Sequence[Round25OfficialResolution],
    authority: Round25ResolutionAuthority,
) -> Round25DevelopmentDataset:
    condition_ids = {snapshot.condition_id for snapshot in snapshots}
    selected_resolutions = tuple(
        resolution
        for resolution in resolutions
        if resolution.condition_id in condition_ids
    )
    return require_round25_dataset_minimum(build_round25_development_dataset(
        role=role,
        snapshots=snapshots,
        resolutions=selected_resolutions,
        resolution_authority=authority,
    ))


def _selection_claim_sha256(
    *,
    feature_manifest_sha256: str,
    resolution_manifest_sha256: str,
    ledger_sha256: str,
    prepared_sha256: str,
) -> str:
    return _canonical_sha256({
        "contract_sha256": POLYMARKET_ROUND25_COORDINATOR_CONTRACT_SHA256,
        "feature_store_manifest_sha256": feature_manifest_sha256,
        "model_ledger_sha256": ledger_sha256,
        "prepared_prediction_sha256": prepared_sha256,
        "resolution_store_manifest_sha256": resolution_manifest_sha256,
        "selection_target_access_one_use": True,
    })


def advance_round25_post_capture(
    *,
    paths: Round25CoordinatorPaths,
    terminal_transport_manifest: Mapping[str, object],
    source_commit_oid: str,
    resolution_client: Round25ResolutionPublicClient | None = None,
    maximum_resolution_conditions: int = 128,
    lightgbm_backend: str = "auto",
    tcn_backend: str = "auto",
    observed_at_ms: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Advance through bounded collection or the complete leakage-safe model path."""

    selected_paths = paths.validated()
    transport = validate_round25_terminal_transport_manifest(
        terminal_transport_manifest
    )
    now_ms = int(observed_at_ms or time.time_ns() // 1_000_000)
    with _coordinator_lock(selected_paths.lock):
        _emit(progress, "coordinator_started", {"observed_at_ms": now_ms})
        if selected_paths.feature_database.exists():
            feature_manifest = audit_round25_joint_store(
                selected_paths.feature_database
            )
        else:
            feature_manifest, _receipt_audit = materialize_round25_joint_feature_store(
                source_database=selected_paths.source_database,
                destination_database=selected_paths.feature_database,
                terminal_transport_manifest=transport,
                observed_at_ms=now_ms,
            )
        _write_state(
            selected_paths.state,
            _state_body(
                phase="feature_materialized",
                updated_at_ms=now_ms,
                feature_store_manifest_sha256=feature_manifest["manifest_sha256"],
            ),
        )
        _emit(
            progress,
            "feature_materialized",
            {"manifest_sha256": feature_manifest["manifest_sha256"]},
        )

        if selected_paths.resolution_database.exists():
            resolution_manifest = audit_round25_resolution_store(
                selected_paths.resolution_database
            )
        else:
            collection, _claim = initialize_round25_resolution_collection(
                feature_database=selected_paths.feature_database,
                destination_database=selected_paths.resolution_database,
                created_at_ms=now_ms,
            )
            report = collect_round25_resolutions_once(
                collection_database=collection,
                client=resolution_client or Round25ResolutionPublicClient(),
                maximum_conditions=maximum_resolution_conditions,
                progress=progress,
            )
            if not report["finalization_ready"]:
                state = _write_state(
                    selected_paths.state,
                    _state_body(
                        phase="resolution_collection_pending",
                        updated_at_ms=now_ms,
                        feature_store_manifest_sha256=(
                            feature_manifest["manifest_sha256"]
                        ),
                        resolution_pending_count=int(
                            report["pending_condition_count"]
                        ),
                    ),
                )
                _emit(progress, "resolution_collection_pending", report)
                return state
            resolution_manifest = finalize_round25_resolution_store(
                feature_database=selected_paths.feature_database,
                destination_database=selected_paths.resolution_database,
                created_at_ms=now_ms,
            )
        _write_state(
            selected_paths.state,
            _state_body(
                phase="resolution_finalized",
                updated_at_ms=now_ms,
                feature_store_manifest_sha256=feature_manifest["manifest_sha256"],
                resolution_store_manifest_sha256=(
                    resolution_manifest["manifest_sha256"]
                ),
            ),
        )
        _emit(
            progress,
            "resolution_finalized",
            {"manifest_sha256": resolution_manifest["manifest_sha256"]},
        )

        endpoint_manifest, endpoints = load_round25_joint_endpoint_inputs(
            selected_paths.feature_database
        )
        if endpoint_manifest["manifest_sha256"] != feature_manifest["manifest_sha256"]:
            raise ValueError("Round 25 coordinator endpoint source differs")
        if selected_paths.model_ledger.exists():
            ledger = load_round25_model_ledger(selected_paths.model_ledger)
            implementation = _verified_source_identity(
                selected_paths.repository,
                source_commit_oid=source_commit_oid,
            )
            if (
                ledger.source_commit_oid != source_commit_oid
                or ledger.implementation_sha256 != implementation
            ):
                raise ValueError("Round 25 existing model ledger source differs")
        else:
            fit_manifest, authority, fit_resolutions = (
                load_round25_fit_resolution_inputs(
                    selected_paths.resolution_database
                )
            )
            if fit_manifest["manifest_sha256"] != resolution_manifest["manifest_sha256"]:
                raise ValueError("Round 25 coordinator fit target source differs")
            train = _role_dataset(
                role="train",
                snapshots=endpoints["train"],
                resolutions=fit_resolutions["train"],
                authority=authority,
            )
            calibration = _role_dataset(
                role="calibration",
                snapshots=endpoints["calibration"],
                resolutions=fit_resolutions["calibration"],
                authority=authority,
            )
            implementation = _verified_source_identity(
                selected_paths.repository,
                source_commit_oid=source_commit_oid,
            )

            def source_factory(logistic: object) -> tuple[object, object]:
                return create_round25_store_tcn_fit_sources(
                    feature_database=selected_paths.feature_database,
                    feature_store_manifest_sha256=feature_manifest[
                        "manifest_sha256"
                    ],
                    train=train,
                    calibration=calibration,
                    logistic=logistic,
                    progress=progress,
                )

            def tcn_progress(value: Round25TCNFitProgress) -> None:
                _emit(progress, "tcn_fit", asdict(value))

            ledger = fit_round25_model_ledger_coordinated(
                source_commit_oid=source_commit_oid,
                implementation_sha256=implementation,
                train=train,
                calibration=calibration,
                tcn_source_factory=source_factory,
                lightgbm_backend=lightgbm_backend,
                tcn_backend=tcn_backend,
                progress_callback=tcn_progress,
            )
            write_round25_model_ledger(selected_paths.model_ledger, ledger)
        _write_state(
            selected_paths.state,
            _state_body(
                phase="model_fitted",
                updated_at_ms=now_ms,
                feature_store_manifest_sha256=feature_manifest["manifest_sha256"],
                resolution_store_manifest_sha256=(
                    resolution_manifest["manifest_sha256"]
                ),
                model_ledger_sha256=ledger.ledger_sha256,
            ),
        )
        _emit(progress, "model_fitted", {"ledger_sha256": ledger.ledger_sha256})

        if selected_paths.prepared_prediction.exists():
            prepared = load_round25_prepared_prediction(
                selected_paths.prepared_prediction
            )
            if prepared.model_ledger_sha256 != ledger.ledger_sha256:
                raise ValueError("Round 25 prepared prediction ledger differs")
        else:
            prepared = prepare_round25_target_free_prediction(
                ledger=ledger,
                snapshots=endpoints["selection"],
                source_receipt_audit_sha256=feature_manifest[
                    "terminal_receipt_audit_sha256"
                ],
                tcn_backend=tcn_backend,
            )
            write_round25_prepared_prediction(
                selected_paths.prepared_prediction,
                prepared,
            )
        access_store = Round25SelectionAccessStore(
            selected_paths.selection_access_store
        )
        try:
            access_status, access_claim = access_store.validate_prediction_binding(
                panel=prepared.panel
            )
        except RuntimeError:
            access_claim = _selection_claim_sha256(
                feature_manifest_sha256=feature_manifest["manifest_sha256"],
                resolution_manifest_sha256=resolution_manifest["manifest_sha256"],
                ledger_sha256=ledger.ledger_sha256,
                prepared_sha256=prepared.prepared_sha256,
            )
            freeze_round25_prepared_prediction(
                store=access_store,
                prepared=prepared,
                one_use_claim_sha256=access_claim,
            )
            access_status = "prediction_panel_frozen"
        _write_state(
            selected_paths.state,
            _state_body(
                phase="selection_prediction_frozen",
                updated_at_ms=now_ms,
                feature_store_manifest_sha256=feature_manifest["manifest_sha256"],
                resolution_store_manifest_sha256=(
                    resolution_manifest["manifest_sha256"]
                ),
                model_ledger_sha256=ledger.ledger_sha256,
                prepared_prediction_sha256=prepared.prepared_sha256,
                selection_access_status=access_status,
            ),
        )
        _emit(
            progress,
            "selection_prediction_bound",
            {"status": access_status, "claim_sha256": access_claim},
        )

        selection_manifest, selection_authority, selection_resolutions, observed_claim = (
            load_round25_selection_resolution_inputs(
                selected_paths.resolution_database,
                panel=prepared.panel,
                access_store=access_store,
                allow_consumed_recovery=access_status == "target_access_consumed",
            )
        )
        if (
            selection_manifest["manifest_sha256"]
            != resolution_manifest["manifest_sha256"]
            or observed_claim != access_claim
        ):
            raise ValueError("Round 25 coordinator selection target source differs")
        selection = _role_dataset(
            role="selection",
            snapshots=endpoints["selection"],
            resolutions=selection_resolutions,
            authority=selection_authority,
        )
        if access_status == "prediction_panel_frozen":
            receipt = access_store.consume_target_access(
                panel=prepared.panel,
                selection=selection,
            )
        else:
            receipt = access_store.load_consumed_receipt(
                panel=prepared.panel,
                selection=selection,
            )
        result = evaluate_round25_predictive_candidates(
            panel=prepared.panel,
            selection=selection,
            target_access_receipt=receipt,
            target_access_store=access_store,
        )
        if selected_paths.predictive_result.exists():
            existing = load_round25_predictive_result(
                selected_paths.predictive_result
            )
            if existing.result_sha256 != result.result_sha256:
                raise ValueError("Round 25 existing predictive result differs")
            result = existing
        else:
            write_round25_predictive_result(
                selected_paths.predictive_result,
                result,
            )
        state = _write_state(
            selected_paths.state,
            _state_body(
                phase="predictive_evaluation_complete",
                updated_at_ms=now_ms,
                feature_store_manifest_sha256=feature_manifest["manifest_sha256"],
                resolution_store_manifest_sha256=(
                    resolution_manifest["manifest_sha256"]
                ),
                model_ledger_sha256=ledger.ledger_sha256,
                prepared_prediction_sha256=prepared.prepared_sha256,
                selection_access_status="target_access_consumed",
                predictive_result_sha256=result.result_sha256,
                predictive_gate_passed=result.predictive_gate_passed,
            ),
        )
        _emit(
            progress,
            "predictive_evaluation_complete",
            {
                "predictive_gate_passed": result.predictive_gate_passed,
                "result_sha256": result.result_sha256,
            },
        )
        if not result.predictive_gate_passed:
            return state

        if selected_paths.economic_result.exists():
            economic = load_round25_economic_result(
                selected_paths.economic_result
            )
            if (
                economic.feature_store_manifest_sha256
                != feature_manifest["manifest_sha256"]
                or economic.terminal_transport_manifest_sha256
                != transport["manifest_sha256"]
                or economic.terminal_receipt_audit_sha256
                != feature_manifest["terminal_receipt_audit_sha256"]
                or economic.resolution_store_manifest_sha256
                != resolution_manifest["manifest_sha256"]
                or economic.resolution_authority_sha256
                != selection_authority.authority_sha256
                or economic.prepared_prediction_sha256
                != prepared.prepared_sha256
                or economic.predictive_result_sha256 != result.result_sha256
                or economic.nominated_candidate_id
                != result.nominated_candidate_id
            ):
                raise ValueError("Round 25 existing economic result differs")
        else:
            economic = replay_round25_development_economics(
                source_database=selected_paths.source_database,
                terminal_transport_manifest=transport,
                feature_database=selected_paths.feature_database,
                resolution_store_manifest=resolution_manifest,
                resolution_authority=selection_authority,
                selection_resolutions=selection_resolutions,
                prepared_prediction=prepared,
                predictive_result=result,
                progress=progress,
            )
            write_round25_economic_result(
                selected_paths.economic_result,
                economic,
            )
        state = _write_state(
            selected_paths.state,
            _state_body(
                phase="economic_evaluation_complete",
                updated_at_ms=now_ms,
                feature_store_manifest_sha256=feature_manifest["manifest_sha256"],
                resolution_store_manifest_sha256=(
                    resolution_manifest["manifest_sha256"]
                ),
                model_ledger_sha256=ledger.ledger_sha256,
                prepared_prediction_sha256=prepared.prepared_sha256,
                selection_access_status="target_access_consumed",
                predictive_result_sha256=result.result_sha256,
                economic_result_sha256=economic.result_sha256,
                predictive_gate_passed=True,
                development_economic_gate_passed=(
                    economic.development_economic_gate_passed
                ),
            ),
        )
        _emit(
            progress,
            "economic_evaluation_complete",
            {
                "development_economic_gate_passed": (
                    economic.development_economic_gate_passed
                ),
                "result_sha256": economic.result_sha256,
            },
        )
        return state


__all__ = [
    "POLYMARKET_ROUND25_COORDINATOR_CONTRACT_SHA256",
    "POLYMARKET_ROUND25_COORDINATOR_PHASES",
    "POLYMARKET_ROUND25_COORDINATOR_STATE_SCHEMA_VERSION",
    "Round25CoordinatorPaths",
    "advance_round25_post_capture",
    "load_round25_coordinator_state",
    "validate_round25_coordinator_state",
]
