"""Central immutable cutoff for reusable historical research data."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
import hashlib
import json
from pathlib import Path
import re


RESEARCH_DATA_SNAPSHOT_SCHEMA_VERSION = "research-data-snapshot-contract-v1"
RESEARCH_DATA_SNAPSHOT_ID = "historical-through-2026-08-13-utc-v1"
RESEARCH_DATA_CUTOFF_UTC = datetime(2026, 8, 14, tzinfo=UTC)
RESEARCH_DATA_CUTOFF_MS = 1_786_665_600_000
RESEARCH_DATA_CUTOFF_NS = 1_786_665_600_000_000_000
RESEARCH_DATA_SNAPSHOT_CONTRACT_RELATIVE_PATH = Path(
    "docs/model-research/research-data-snapshot-contract-v1.json"
)
RESEARCH_DATA_SNAPSHOT_CONTRACT_SHA256 = (
    "c9493771fa966c4935b182bff8e29c2cb3cde05e25e96e430426388429c374ee"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("research data snapshot has duplicate keys")
        output[key] = value
    return output


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


def validate_research_data_snapshot_contract(
    value: Mapping[str, object],
) -> dict[str, object]:
    contract = dict(value)
    claimed = str(contract.pop("contract_sha256", "")).lower()
    cutoff = contract.get("cutoff")
    governance = contract.get("governance")
    prospective = contract.get("prospective_evidence")
    registered = contract.get("registered_prospective_experiments")
    scope = contract.get("scope")
    authority = contract.get("authority")
    if (
        claimed != RESEARCH_DATA_SNAPSHOT_CONTRACT_SHA256
        or _SHA256.fullmatch(claimed) is None
        or claimed != _canonical_sha256(contract)
        or contract.get("schema_version")
        != RESEARCH_DATA_SNAPSHOT_SCHEMA_VERSION
        or contract.get("snapshot_id") != RESEARCH_DATA_SNAPSHOT_ID
        or cutoff
        != {
            "epoch_milliseconds": RESEARCH_DATA_CUTOFF_MS,
            "epoch_nanoseconds": RESEARCH_DATA_CUTOFF_NS,
            "event_time_rule": "event_timestamp_utc_strictly_less_than_cutoff",
            "exclusive_utc": "2026-08-14T00:00:00Z",
            "last_included_utc_date": "2026-08-13",
        }
        or governance
        != {
            "automatic_latest_extension_allowed": False,
            "cutoff_change_in_place_allowed": False,
            "dynamic_current_date_used_by_research_ingestion": False,
            "future_snapshot_requires_new_contract_and_source_audit": True,
            "historical_snapshot_is_append_only": False,
            "historical_snapshot_is_immutable": True,
            "missing_data_may_be_fabricated_or_forward_filled": False,
            "new_columns_or_tables_may_reuse_same_cutoff": True,
            "partial_or_missing_source_coverage_must_be_reported": True,
            "snapshot_cutoff_implies_complete_coverage_through_cutoff": False,
        }
        or prospective
        != {
            "automatic_merge_into_historical_training_allowed": False,
            "may_have_event_time_at_or_after_cutoff": True,
            "must_bind_a_frozen_campaign_or_experiment_identifier": True,
            "must_remain_partitioned_from_reusable_historical_training": True,
            "reusable_historical_training_eligible": False,
        }
        or registered
        != [
            {
                "contract_sha256": (
                    "3f484154d69baed632e617f2de41b149385299a97b47e5e9184c694c43c89392"
                ),
                "cutoff_filter_applied_to_frozen_campaign": False,
                "experiment_id": "polymarket-round27-stage1",
                "frozen_internal_fit_and_evaluation_permitted": True,
                "historical_snapshot_automatically_extended": False,
                "reusable_historical_training_eligible": False,
            },
            {
                "contract_sha256": (
                    "8239488145f0ffe331cf9823e5517120dda0d12eb5f366cf00c5e106318d4668"
                ),
                "cutoff_filter_applied_to_frozen_campaign": False,
                "experiment_id": "polymarket-round28-binance-bbo",
                "frozen_internal_fit_and_evaluation_permitted": True,
                "historical_snapshot_automatically_extended": False,
                "reusable_historical_training_eligible": False,
            },
            {
                "contract_sha256": (
                    "eca592acb4c3f37c6d043d37664614d35994d6e9b3ebea2e801351c287a49bbf"
                ),
                "cutoff_filter_applied_to_frozen_campaign": False,
                "experiment_id": "action-value-round75-continuous-capture",
                "frozen_internal_fit_and_evaluation_permitted": True,
                "historical_snapshot_automatically_extended": False,
                "reusable_historical_training_eligible": False,
            },
        ]
        or scope
        != {
            "assets": ["BTC", "ETH", "SOL"],
            "historical_research_data_repo_wide": True,
            "markets": [
                "Binance spot",
                "Binance USD-M futures",
                "Polymarket",
            ],
            "timestamp_basis": "event_time_utc",
        }
        or not isinstance(authority, Mapping)
        or set(authority)
        != {
            "credentials_used",
            "edge_claim",
            "live_trading_authority",
            "orders_submitted",
            "paper_trading_authority",
            "profitability_claim",
        }
        or any(item is not False for item in authority.values())
    ):
        raise ValueError("research data snapshot contract differs")
    return {**contract, "contract_sha256": claimed}


def load_research_data_snapshot_contract(
    repository: str | Path,
) -> dict[str, object]:
    path = Path(repository).resolve(strict=True) / (
        RESEARCH_DATA_SNAPSHOT_CONTRACT_RELATIVE_PATH
    )
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {raw}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("research data snapshot contract is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("research data snapshot contract root differs")
    return validate_research_data_snapshot_contract(value)


def require_historical_event_time_ms(value: int, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= RESEARCH_DATA_CUTOFF_MS
    ):
        raise ValueError(f"{name} is outside the immutable historical snapshot")
    return value


def require_historical_event_time_ns(value: int, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= RESEARCH_DATA_CUTOFF_NS
    ):
        raise ValueError(f"{name} is outside the immutable historical snapshot")
    return value


def require_historical_utc_date(value: str, *, name: str) -> str:
    try:
        selected = date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{name} is not a canonical UTC date") from exc
    if selected.isoformat() != value or selected >= RESEARCH_DATA_CUTOFF_UTC.date():
        raise ValueError(f"{name} is outside the immutable historical snapshot")
    return value


def validate_prospective_partition(
    *,
    experiment_id: str,
    reusable_historical_training_eligible: bool,
) -> str:
    selected = str(experiment_id or "").strip()
    if (
        not selected
        or len(selected) > 160
        or reusable_historical_training_eligible is not False
    ):
        raise ValueError("prospective evidence partition differs")
    return selected


__all__ = [
    "RESEARCH_DATA_CUTOFF_MS",
    "RESEARCH_DATA_CUTOFF_NS",
    "RESEARCH_DATA_CUTOFF_UTC",
    "RESEARCH_DATA_SNAPSHOT_CONTRACT_RELATIVE_PATH",
    "RESEARCH_DATA_SNAPSHOT_CONTRACT_SHA256",
    "RESEARCH_DATA_SNAPSHOT_ID",
    "RESEARCH_DATA_SNAPSHOT_SCHEMA_VERSION",
    "load_research_data_snapshot_contract",
    "require_historical_event_time_ms",
    "require_historical_event_time_ns",
    "require_historical_utc_date",
    "validate_prospective_partition",
    "validate_research_data_snapshot_contract",
]
