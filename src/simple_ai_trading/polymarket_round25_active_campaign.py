"""Read-only contract for the active Round 25 exact-TWAP capture."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .polymarket_recorder import POLYMARKET_MAXIMUM_CAPTURE_DURATION_SECONDS


POLYMARKET_ROUND25_ACTIVE_DESIGN_SHA256 = (
    "c083a7fb4a3ecb0f080ce2bf2c2bb3021000b2ab85562caa6a9a69067c6e673d"
)
POLYMARKET_ROUND25_ACTIVE_PLAN_SHA256 = (
    "a0b5525697c3c1e1b175bd0f0ac724fdb62845638d2040e9964221031d3e7b20"
)
POLYMARKET_ROUND25_ACTIVE_SOURCE_QUALIFICATION_SHA256 = (
    "d06f76ba7305b55e254209839360af277f68947f1ff5fbe986df5a9dd374334d"
)
POLYMARKET_ROUND25_ACTIVE_PLAN_SCHEMA_VERSION = (
    "polymarket-round25-twap-core-campaign-plan-v2"
)
POLYMARKET_ROUND25_ACTIVE_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round25-twap-core-segment-manifest-v2"
)
POLYMARKET_ROUND25_ACTIVE_STATE_SCHEMA_VERSION = (
    "polymarket-round25-twap-core-campaign-state-v2"
)
POLYMARKET_ROUND25_ACTIVE_RESULT_SCHEMA_VERSION = (
    "polymarket-round25-twap-core-segment-result-v2"
)
POLYMARKET_ROUND25_RESOLUTION_SOURCE = (
    "https://data.chain.link/streams/btc-usd-twap-30s-streams"
)
POLYMARKET_ROUND25_ACTIVE_TWAP_TOPIC = "crypto_prices_twap_thirty"
POLYMARKET_ROUND25_START_MS = 1_786_406_400_000
POLYMARKET_ROUND25_END_MS = 1_788_046_800_000
POLYMARKET_ROUND25_DATABASE_CAP_BYTES = 200 * 1024**3
POLYMARKET_ROUND25_MINIMUM_FREE_BYTES = 512 * 1024**3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_REQUIRED_FILES = frozenset(
    (
        "docs/model-research/polymarket/round-025-twap-core-capture-design-v2.json",
        "docs/model-research/polymarket/round-025-twap-wire-source-qualification-v2-2026-08-10.json",
        "src/simple_ai_trading/polymarket.py",
        "src/simple_ai_trading/polymarket_recorder.py",
        "src/simple_ai_trading/polymarket_round25_campaign.py",
        "tests/test_polymarket.py",
        "tests/test_polymarket_round25_campaign.py",
        "tools/qualify_polymarket_round25_twap_wire_v2.py",
        "tools/run_polymarket_round25_campaign.py",
    )
)
_FALSE_FIELDS = (
    "binance_captured",
    "credentials_used",
    "outcomes_consulted",
    "model_scores_consulted",
    "model_data_eligible",
    "profitability_claim",
    "paper_trading_authority",
    "live_trading_authority",
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Round 25 active campaign JSON contains duplicate keys")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 25 active campaign JSON contains {value}")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or not 2 <= path.stat().st_size <= 2**20:
        raise ValueError(f"Round 25 active {label} is unavailable")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Round 25 active {label} is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Round 25 active {label} is invalid")
    return value


@dataclass(frozen=True, slots=True)
class PolymarketRound25ActiveCampaignPlan:
    created_at_ms: int
    repository_commit_oid: str
    repository_tree_oid: str
    repository_file_sha256: Mapping[str, str]
    source_qualification_sha256: str
    plan_sha256: str

    @property
    def scheduled_start_ms(self) -> int:
        return POLYMARKET_ROUND25_START_MS

    @property
    def scheduled_end_ms(self) -> int:
        return POLYMARKET_ROUND25_END_MS


def validate_round25_active_campaign_plan(
    value: Mapping[str, object],
) -> PolymarketRound25ActiveCampaignPlan:
    payload = dict(value)
    claimed = str(payload.pop("plan_sha256", "")).strip().lower()
    files = payload.get("repository_file_sha256")
    expected_keys = {
        "schema_version",
        "created_at_ms",
        "scheduled_start_ms",
        "scheduled_end_ms",
        "design_sha256",
        "source_qualification_sha256",
        "resolution_source",
        "required_assets",
        "required_streams",
        "required_clob_lanes",
        "required_rtds_topics",
        "database_cap_bytes",
        "minimum_free_bytes",
        "repository_commit_oid",
        "repository_tree_oid",
        "repository_file_sha256",
        *_FALSE_FIELDS,
    }
    if (
        set(payload) != expected_keys
        or claimed != POLYMARKET_ROUND25_ACTIVE_PLAN_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND25_ACTIVE_PLAN_SCHEMA_VERSION
        or type(payload.get("created_at_ms")) is not int
        or not 0 < int(payload["created_at_ms"]) < POLYMARKET_ROUND25_START_MS
        or payload.get("scheduled_start_ms") != POLYMARKET_ROUND25_START_MS
        or payload.get("scheduled_end_ms") != POLYMARKET_ROUND25_END_MS
        or payload.get("design_sha256") != POLYMARKET_ROUND25_ACTIVE_DESIGN_SHA256
        or payload.get("source_qualification_sha256")
        != POLYMARKET_ROUND25_ACTIVE_SOURCE_QUALIFICATION_SHA256
        or payload.get("resolution_source") != POLYMARKET_ROUND25_RESOLUTION_SOURCE
        or payload.get("required_assets") != ["BTC"]
        or payload.get("required_streams") != ["clob_market", "polymarket_rtds"]
        or payload.get("required_clob_lanes") != ["clob"]
        or payload.get("required_rtds_topics")
        != [POLYMARKET_ROUND25_ACTIVE_TWAP_TOPIC]
        or payload.get("database_cap_bytes") != POLYMARKET_ROUND25_DATABASE_CAP_BYTES
        or payload.get("minimum_free_bytes") != POLYMARKET_ROUND25_MINIMUM_FREE_BYTES
        or _GIT_OID.fullmatch(str(payload.get("repository_commit_oid") or "")) is None
        or _GIT_OID.fullmatch(str(payload.get("repository_tree_oid") or "")) is None
        or not isinstance(files, Mapping)
        or set(files) != _REQUIRED_FILES
        or any(_SHA256.fullmatch(str(item)) is None for item in files.values())
        or any(payload.get(field) is not False for field in _FALSE_FIELDS)
    ):
        raise ValueError("Round 25 active campaign plan differs")
    return PolymarketRound25ActiveCampaignPlan(
        created_at_ms=int(payload["created_at_ms"]),
        repository_commit_oid=str(payload["repository_commit_oid"]),
        repository_tree_oid=str(payload["repository_tree_oid"]),
        repository_file_sha256=dict(files),
        source_qualification_sha256=str(payload["source_qualification_sha256"]),
        plan_sha256=claimed,
    )


def load_round25_active_campaign_plan(
    path: str | Path,
) -> PolymarketRound25ActiveCampaignPlan:
    return validate_round25_active_campaign_plan(
        _read_object(Path(path), label="campaign plan")
    )


def build_round25_active_segment_manifest(
    plan: PolymarketRound25ActiveCampaignPlan,
    *,
    run_id: str,
    created_at_ms: int,
    capture_duration_seconds: int,
    segment_index: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND25_ACTIVE_MANIFEST_SCHEMA_VERSION,
        "run_id": str(run_id),
        "created_at_ms": int(created_at_ms),
        "capture_duration_seconds": int(capture_duration_seconds),
        "segment_index": int(segment_index),
        "plan_sha256": plan.plan_sha256,
        "design_sha256": POLYMARKET_ROUND25_ACTIVE_DESIGN_SHA256,
        "source_qualification_sha256": plan.source_qualification_sha256,
        "resolution_source": POLYMARKET_ROUND25_RESOLUTION_SOURCE,
        "scheduled_campaign_start_ms": POLYMARKET_ROUND25_START_MS,
        "scheduled_campaign_end_ms": POLYMARKET_ROUND25_END_MS,
        "required_assets": ["BTC"],
        "required_streams": ["clob_market", "polymarket_rtds"],
        "required_clob_lanes": ["clob"],
        "required_rtds_topics": [POLYMARKET_ROUND25_ACTIVE_TWAP_TOPIC],
        "repository_commit_oid": plan.repository_commit_oid,
        "repository_tree_oid": plan.repository_tree_oid,
        **{field: False for field in _FALSE_FIELDS},
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return validate_round25_active_segment_manifest(payload, plan)


def validate_round25_active_segment_manifest(
    value: Mapping[str, object],
    plan: PolymarketRound25ActiveCampaignPlan,
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("manifest_sha256", "")).strip().lower()
    created = payload.get("created_at_ms")
    duration = payload.get("capture_duration_seconds")
    expected_keys = {
        "schema_version",
        "run_id",
        "created_at_ms",
        "capture_duration_seconds",
        "segment_index",
        "plan_sha256",
        "design_sha256",
        "source_qualification_sha256",
        "resolution_source",
        "scheduled_campaign_start_ms",
        "scheduled_campaign_end_ms",
        "required_assets",
        "required_streams",
        "required_clob_lanes",
        "required_rtds_topics",
        "repository_commit_oid",
        "repository_tree_oid",
        *_FALSE_FIELDS,
    }
    if (
        set(payload) != expected_keys
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND25_ACTIVE_MANIFEST_SCHEMA_VERSION
        or _RUN_ID.fullmatch(str(payload.get("run_id") or "")) is None
        or type(created) is not int
        or not POLYMARKET_ROUND25_START_MS <= int(created) < POLYMARKET_ROUND25_END_MS
        or type(duration) is not int
        or not 5 <= int(duration) <= POLYMARKET_MAXIMUM_CAPTURE_DURATION_SECONDS
        or int(created) + int(duration) * 1_000 > POLYMARKET_ROUND25_END_MS + 1_000
        or type(payload.get("segment_index")) is not int
        or int(payload["segment_index"]) < 0
        or payload.get("plan_sha256") != plan.plan_sha256
        or payload.get("design_sha256") != POLYMARKET_ROUND25_ACTIVE_DESIGN_SHA256
        or payload.get("source_qualification_sha256")
        != plan.source_qualification_sha256
        or payload.get("resolution_source") != POLYMARKET_ROUND25_RESOLUTION_SOURCE
        or payload.get("scheduled_campaign_start_ms") != POLYMARKET_ROUND25_START_MS
        or payload.get("scheduled_campaign_end_ms") != POLYMARKET_ROUND25_END_MS
        or payload.get("required_assets") != ["BTC"]
        or payload.get("required_streams") != ["clob_market", "polymarket_rtds"]
        or payload.get("required_clob_lanes") != ["clob"]
        or payload.get("required_rtds_topics")
        != [POLYMARKET_ROUND25_ACTIVE_TWAP_TOPIC]
        or payload.get("repository_commit_oid") != plan.repository_commit_oid
        or payload.get("repository_tree_oid") != plan.repository_tree_oid
        or any(payload.get(field) is not False for field in _FALSE_FIELDS)
    ):
        raise ValueError("Round 25 active segment manifest differs")
    return {**payload, "manifest_sha256": claimed}


__all__ = [
    "POLYMARKET_ROUND25_ACTIVE_DESIGN_SHA256",
    "POLYMARKET_ROUND25_ACTIVE_MANIFEST_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_ACTIVE_PLAN_SHA256",
    "POLYMARKET_ROUND25_ACTIVE_PLAN_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_ACTIVE_RESULT_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_ACTIVE_SOURCE_QUALIFICATION_SHA256",
    "POLYMARKET_ROUND25_ACTIVE_STATE_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_ACTIVE_TWAP_TOPIC",
    "POLYMARKET_ROUND25_END_MS",
    "POLYMARKET_ROUND25_RESOLUTION_SOURCE",
    "POLYMARKET_ROUND25_START_MS",
    "PolymarketRound25ActiveCampaignPlan",
    "build_round25_active_segment_manifest",
    "load_round25_active_campaign_plan",
    "validate_round25_active_campaign_plan",
    "validate_round25_active_segment_manifest",
]
