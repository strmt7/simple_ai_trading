"""Operational, fail-closed handoff from Round 25 capture to model research."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Iterator

from .compute import SUPPORTED_COMPUTE_BACKENDS
from .polymarket_round25_active_campaign import (
    POLYMARKET_ROUND25_ACTIVE_STATE_SCHEMA_VERSION,
    load_round25_active_campaign_plan,
)
from .polymarket_round25_coordinator import (
    Round25CoordinatorPaths,
    advance_round25_post_capture,
)
from .polymarket_round25_terminal import (
    build_round25_terminal_transport_manifest,
    load_round25_terminal_transport_manifest,
    write_round25_terminal_transport_manifest,
)


ProgressCallback = Callable[[str, Mapping[str, object]], None]
_TERMINAL_STATUSES = frozenset(
    ("campaign_window_ended", "source_regime_changed", "campaign_failed")
)
_AUTHORITY_FIELDS = (
    "model_data_eligible",
    "profitability_claim",
    "paper_trading_authority",
    "live_trading_authority",
)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 25 runner JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 25 runner JSON contains {value}")


def _read_state(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file() or not 2 <= path.stat().st_size <= 64 * 1024:
        raise ValueError("Round 25 campaign state is unavailable")
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 25 campaign state is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("Round 25 campaign state is invalid")
    return value


def _emit(
    progress: ProgressCallback | None,
    event: str,
    values: Mapping[str, object],
) -> None:
    if progress is not None:
        progress(event, values)


def _source_commit(repository: Path, explicit: str | None) -> str:
    if explicit:
        selected = explicit.strip().lower()
        if _COMMIT.fullmatch(selected) is None:
            raise ValueError("Round 25 model source commit differs")
        return selected
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
    )
    if completed.returncode:
        raise ValueError("Round 25 model source commit is unavailable")
    selected = completed.stdout.strip().lower()
    if _COMMIT.fullmatch(selected) is None:
        raise ValueError("Round 25 model source commit differs")
    return selected


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Round 25 runner lock path differs")
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
        raise RuntimeError("Round 25 post-capture runner is already active") from exc
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


@dataclass(frozen=True, slots=True)
class Round25PostCaptureRunnerConfig:
    repository: Path
    source_database: Path
    plan_path: Path
    state_root: Path
    output_root: Path
    source_commit_oid: str | None = None
    maximum_resolution_conditions: int = 128
    lightgbm_backend: str = "auto"
    tcn_backend: str = "auto"

    def validated(self) -> Round25PostCaptureRunnerConfig:
        repository = self.repository.resolve()
        source = self.source_database.resolve(strict=False)
        plan = self.plan_path.resolve()
        state = self.state_root.resolve()
        output = self.output_root.resolve(strict=False)
        if (
            not repository.is_dir()
            or not plan.is_file()
            or not state.is_dir()
            or any(path.is_symlink() for path in (self.plan_path, self.state_root))
            or output in {repository, source, plan, state}
            or state in output.parents
            or output in state.parents
            or output in source.parents
            or source == plan
            or state == plan
            or not 1 <= self.maximum_resolution_conditions <= 512
            or not isinstance(self.lightgbm_backend, str)
            or self.lightgbm_backend.strip().lower() not in SUPPORTED_COMPUTE_BACKENDS
            or not isinstance(self.tcn_backend, str)
            or self.tcn_backend.strip().lower() not in SUPPORTED_COMPUTE_BACKENDS
            or self.source_commit_oid is not None
            and _COMMIT.fullmatch(self.source_commit_oid.strip().lower()) is None
        ):
            raise ValueError("Round 25 post-capture runner configuration differs")
        return Round25PostCaptureRunnerConfig(
            repository=repository,
            source_database=source,
            plan_path=plan,
            state_root=state,
            output_root=output,
            source_commit_oid=self.source_commit_oid,
            maximum_resolution_conditions=self.maximum_resolution_conditions,
            lightgbm_backend=self.lightgbm_backend.strip().lower(),
            tcn_backend=self.tcn_backend.strip().lower(),
        )


def round25_post_capture_paths(
    config: Round25PostCaptureRunnerConfig,
) -> tuple[Path, Round25CoordinatorPaths, Path]:
    selected = config.validated()
    output = selected.output_root
    terminal = output / "terminal-transport-v2.json"
    launcher_lock = output / "post-capture-runner.lock"
    coordinator = Round25CoordinatorPaths(
        repository=selected.repository,
        source_database=selected.source_database,
        feature_database=output / "joint-feature-store-v2.duckdb",
        resolution_database=output / "official-resolution-store-v1.duckdb",
        model_ledger=output / "model-ledger-v1.json",
        prepared_prediction=output / "selection-prediction-panel-v1.json",
        selection_access_store=output / "selection-access-v1.duckdb",
        predictive_result=output / "predictive-result-v1.json",
        economic_result=output / "economic-result-v2.json",
        state=output / "coordinator-state-v2.json",
        lock=output / "coordinator.lock",
    ).validated()
    return terminal, coordinator, launcher_lock


def run_round25_post_capture(
    config: Round25PostCaptureRunnerConfig,
    *,
    observed_at_ms: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Advance one bounded, resumable post-capture pass without trading authority."""

    selected = config.validated()
    plan = load_round25_active_campaign_plan(selected.plan_path)
    observed = time.time_ns() // 1_000_000 if observed_at_ms is None else int(observed_at_ms)
    if observed < plan.scheduled_end_ms:
        status = {
            "status": "waiting_for_terminal_capture",
            "observed_at_ms": observed,
            "campaign_end_ms": plan.scheduled_end_ms,
            "remaining_seconds": (plan.scheduled_end_ms - observed) / 1_000.0,
            "source_database_opened": False,
            "orders_submitted": 0,
            **{field: False for field in _AUTHORITY_FIELDS},
        }
        _emit(progress, "waiting_for_terminal_capture", status)
        return status

    campaign_state = _read_state(selected.state_root / "campaign-state.json")
    if campaign_state is None or campaign_state.get("status") not in _TERMINAL_STATUSES:
        if campaign_state is not None and (
            campaign_state.get("schema_version")
            != POLYMARKET_ROUND25_ACTIVE_STATE_SCHEMA_VERSION
            or any(campaign_state.get(field) is not False for field in _AUTHORITY_FIELDS)
        ):
            raise ValueError("Round 25 nonterminal campaign state differs")
        status = {
            "status": "waiting_for_terminal_state",
            "observed_at_ms": observed,
            "source_database_opened": False,
            "orders_submitted": 0,
            **{field: False for field in _AUTHORITY_FIELDS},
        }
        _emit(progress, "waiting_for_terminal_state", status)
        return status

    if (
        campaign_state.get("schema_version")
        != POLYMARKET_ROUND25_ACTIVE_STATE_SCHEMA_VERSION
        or any(campaign_state.get(field) is not False for field in _AUTHORITY_FIELDS)
    ):
        raise ValueError("Round 25 terminal campaign state differs")

    if selected.source_database.is_symlink() or not selected.source_database.is_file():
        raise ValueError("Round 25 terminal source database is unavailable")
    terminal_path, coordinator_paths, launcher_lock = round25_post_capture_paths(selected)
    with _exclusive_lock(launcher_lock):
        if terminal_path.exists():
            terminal = load_round25_terminal_transport_manifest(terminal_path)
            if terminal["source_plan_sha256"] != plan.plan_sha256:
                raise ValueError("Round 25 existing terminal plan differs")
        else:
            terminal = build_round25_terminal_transport_manifest(
                selected.repository,
                plan_path=selected.plan_path,
                state_root=selected.state_root,
                observed_at_ms=observed,
            )
            terminal_path.parent.mkdir(parents=True, exist_ok=True)
            write_round25_terminal_transport_manifest(terminal_path, terminal)
            terminal = load_round25_terminal_transport_manifest(terminal_path)
        _emit(
            progress,
            "terminal_transport_ready",
            {
                "manifest_sha256": terminal["manifest_sha256"],
                "eligible_run_count": len(terminal["eligible_run_ids"]),
            },
        )
        state = advance_round25_post_capture(
            paths=coordinator_paths,
            terminal_transport_manifest=terminal,
            source_commit_oid=_source_commit(
                selected.repository,
                selected.source_commit_oid,
            ),
            maximum_resolution_conditions=selected.maximum_resolution_conditions,
            lightgbm_backend=selected.lightgbm_backend,
            tcn_backend=selected.tcn_backend,
            observed_at_ms=observed,
            progress=progress,
        )
    result = {
        "status": "post_capture_pass_complete",
        "coordinator_phase": state["phase"],
        "coordinator_state_sha256": state["state_sha256"],
        "terminal_transport_manifest_sha256": terminal["manifest_sha256"],
        "orders_submitted": 0,
        **{field: False for field in _AUTHORITY_FIELDS},
    }
    _emit(progress, "post_capture_pass_complete", result)
    return result


__all__ = [
    "Round25PostCaptureRunnerConfig",
    "round25_post_capture_paths",
    "run_round25_post_capture",
]
