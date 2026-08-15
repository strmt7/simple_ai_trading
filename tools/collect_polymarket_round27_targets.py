#!/usr/bin/env python3
"""Collect role-gated official targets for the Round 27 campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_replay import PolymarketEvidenceReplay
from simple_ai_trading.polymarket_round25_resolution_store import (
    Round25ResolutionPublicClient,
)
from simple_ai_trading.polymarket_round27_feature_store import Round27FeatureStore
from simple_ai_trading.polymarket_round27_features import Round27FeatureRow
from simple_ai_trading.polymarket_round27_campaign_admission import (
    load_round27_campaign_admission,
)
from simple_ai_trading.polymarket_round27_model import Round27RoleInterval
from simple_ai_trading.polymarket_round27_model_contract import (
    load_round27_model_contract,
)
from simple_ai_trading.polymarket_round27_target_store import Round27TargetStore


_ROLES = ("train", "calibration", "selection", "sealed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--target-store", type=Path, required=True)
    parser.add_argument("--campaign-admission", type=Path, required=True)
    parser.add_argument("--slot-id", required=True)
    parser.add_argument("--role", choices=_ROLES, required=True)
    parser.add_argument("--selection-claim", type=Path)
    parser.add_argument("--selection-economic-claim", type=Path)
    parser.add_argument("--selection-economic-report", type=Path)
    parser.add_argument("--open-only", action="store_true")
    return parser


def _progress(phase: str, detail: Mapping[str, object]) -> None:
    print(
        json.dumps(
            {
                "at_ms": time.time_ns() // 1_000_000,
                "phase": phase,
                **detail,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 27 target artifact contains duplicate keys")
        output[key] = value
    return output


def _mapping(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        value = json.loads(
            path.resolve(strict=True).read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Round 27 target artifact is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Round 27 target artifact must be an object")
    return value


def _role_rows(
    rows: Sequence[Round27FeatureRow],
    *,
    slot_id: str,
    role: str,
    partitions: Sequence[Mapping[str, object]],
) -> tuple[Round27FeatureRow, ...]:
    intervals = tuple(Round27RoleInterval.from_mapping(item) for item in partitions)
    selected: list[Round27FeatureRow] = []
    for raw in rows:
        row = raw.validated()
        matches = [
            item
            for item in intervals
            if item.slot_id == slot_id
            and item.start_ms <= row.event_start_ms < item.end_ms
        ]
        if len(matches) != 1:
            raise ValueError("Round 27 target feature role is ambiguous")
        if matches[0].role == role:
            selected.append(row)
    if not selected:
        raise ValueError("Round 27 target role has no feature rows")
    return tuple(selected)


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository = arguments.repository.resolve()
    source_database = _resolve(repository, arguments.source_database).resolve(
        strict=True
    )
    feature_database = _resolve(repository, arguments.feature_store).resolve(
        strict=True
    )
    target_database = _resolve(repository, arguments.target_store).resolve()
    if (
        source_database.is_symlink()
        or feature_database.is_symlink()
        or Path(f"{source_database}.wal").exists()
        or Path(f"{feature_database}.wal").exists()
    ):
        raise ValueError("Round 27 target collection requires terminal databases")
    contract = load_round27_model_contract(repository)
    partitions = contract.get("partitions")
    if not isinstance(partitions, list):
        raise ValueError("Round 27 target contract partitions differ")
    with Round27FeatureStore(feature_database, read_only=True) as feature_store:
        feature_audit = feature_store.audit()
        slot_rows = feature_store.load_rows(slot_id=arguments.slot_id)
    campaign_admission = load_round27_campaign_admission(
        _resolve(repository, arguments.campaign_admission),
        contract=contract,
        feature_store_audit_sha256=str(feature_audit["audit_sha256"]),
    )
    role_rows = _role_rows(
        slot_rows,
        slot_id=arguments.slot_id,
        role=arguments.role,
        partitions=[item for item in partitions if isinstance(item, Mapping)],
    )
    run_ids = {row.run_id for row in role_rows}
    condition_ids = sorted({row.condition_id for row in role_rows})
    if len(run_ids) != 1:
        raise ValueError("Round 27 target role spans source runs")
    run_id = next(iter(run_ids))
    with PolymarketEvidenceStore(
        source_database,
        read_only=True,
        memory_limit="512MB",
        threads=1,
    ) as source:
        run = source.connect().execute(
            """
            SELECT status, report_sha256 FROM polymarket_recorder_run
            WHERE run_id=?
            """,
            [run_id],
        ).fetchone()
        markets = PolymarketEvidenceReplay.load_markets(
            source,
            run_id=run_id,
            condition_ids=condition_ids,
        )
        snapshots = source.connect().execute(
            """
            SELECT condition_id, snapshot_sha256
            FROM polymarket_market_snapshot
            WHERE run_id=? AND condition_id IN (SELECT unnest(?::VARCHAR[]))
            ORDER BY event_start_ms, condition_id
            """,
            [run_id, condition_ids],
        ).fetchall()
    snapshots_by_id = {str(row[0]): str(row[1]) for row in snapshots}
    if (
        run is None
        or run[0] not in {"complete", "degraded"}
        or not isinstance(run[1], str)
        or len(run[1]) != 64
        or len(markets) != len(condition_ids)
        or {market.condition_id for market in markets} != set(condition_ids)
        or set(snapshots_by_id) != set(condition_ids)
    ):
        raise ValueError("Round 27 target source lineage differs")
    selection_claim = _mapping(arguments.selection_claim)
    selection_economic_claim = _mapping(arguments.selection_economic_claim)
    selection_economic_report = _mapping(arguments.selection_economic_report)
    with Round27TargetStore(target_database) as target_store:
        opened = target_store.open_role(
            role=arguments.role,
            slot_id=arguments.slot_id,
            run_id=run_id,
            contract=contract,
            feature_store_audit_sha256=str(feature_audit["audit_sha256"]),
            campaign_admission=campaign_admission,
            role_intervals=[
                item for item in partitions if isinstance(item, Mapping)
            ],
            feature_rows=role_rows,
            markets=tuple(
                (market, snapshots_by_id[market.condition_id]) for market in markets
            ),
            opened_at_ms=time.time_ns() // 1_000_000,
            selection_claim=selection_claim,
            selection_economic_claim=selection_economic_claim,
            selection_economic_report=selection_economic_report,
        )
        _progress(
            "target-access-open",
            {
                "role": arguments.role,
                "slot_id": arguments.slot_id,
                "run_id": run_id,
                "condition_count": len(condition_ids),
                "newly_opened": opened,
                "feature_store_audit_sha256": feature_audit["audit_sha256"],
                "campaign_admission_sha256": campaign_admission[
                    "admission_sha256"
                ],
                "orders_submitted": False,
                "trading_authority": False,
            },
        )
        if arguments.open_only:
            return 0
        result = target_store.collect_once(
            role=arguments.role,
            client=Round25ResolutionPublicClient(),
        )
        _progress("target-collection", result)
        if result["pending_condition_count"]:
            return 2
        finalized = target_store.finalize_role(arguments.role)
        _progress("target-role-finalized", finalized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
