"""Terminal-only, single-pass Round 17 campaign development inputs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_replay import PolymarketEvidenceReplay
from .polymarket_round14_campaign import (
    POLYMARKET_ROUND14_CAMPAIGN_SLOT_RESULT_SCHEMA_VERSION,
    POLYMARKET_ROUND14_CAMPAIGN_STATE_SCHEMA_VERSION,
    PolymarketRound14CampaignPlan,
    load_round14_campaign_plan,
)
from .polymarket_round14_dataset import (
    PolymarketRound14AdmissionSpec,
    load_round14_admission_spec,
    materialize_round14_label_free_run_context,
)
from .polymarket_round17_cohort import (
    POLYMARKET_ROUND17_CAMPAIGN_PLAN_SHA256,
    POLYMARKET_ROUND17_CAPTURE_CONTRACT_SHA256,
    Round17CohortCondition,
    Round17CohortPlan,
    build_round17_cohort_condition,
    load_round17_cohort_plan,
)
from .polymarket_round17_dataset import (
    PolymarketRound17ConditionDataset,
    materialize_round17_condition_rows,
)


POLYMARKET_ROUND17_CAMPAIGN_OPERATOR_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-campaign-operator-v1"
)
_DEVELOPMENT_ROLES = frozenset(
    {
        "train",
        "tune_calibration",
        "tune_selection",
        "tune_uncertainty",
        "tune_economic",
    }
)
_TERMINAL_SLOT_STATUSES = frozenset(
    {"complete", "degraded", "failed", "missed", "resource_blocked"}
)
_MATERIALIZABLE_SLOT_STATUSES = frozenset({"complete", "degraded"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9:._-]{1,160}$")
_MAXIMUM_STATE_BYTES = 1024 * 1024


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 17 campaign state contains duplicate JSON keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 17 campaign state contains {value}")


def _read_hashed_json(path: Path, *, label: str) -> dict[str, object]:
    if (
        path.is_symlink()
        or not path.is_file()
        or not 2 <= path.stat().st_size <= _MAXIMUM_STATE_BYTES
    ):
        raise ValueError(f"{label} is unavailable")
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} is not an object")
    payload = dict(raw)
    claimed = str(payload.pop("artifact_sha256", "")).strip().lower()
    if _SHA256.fullmatch(claimed) is None or claimed != _canonical_sha256(payload):
        raise ValueError(f"{label} hash differs")
    return {**payload, "artifact_sha256": claimed}


@dataclass(frozen=True, slots=True)
class Round17CampaignOperatorConfig:
    campaign_plan_path: Path
    cohort_plan_path: Path
    admission_spec_path: Path
    database_path: Path
    state_root: Path
    memory_limit: str = "2GB"
    database_threads: int = 2

    def validate(self) -> Round17CampaignOperatorConfig:
        files = (
            self.campaign_plan_path,
            self.cohort_plan_path,
            self.admission_spec_path,
        )
        resolved_files = tuple(item.resolve() for item in files)
        database = self.database_path.resolve()
        state = self.state_root.resolve()
        if (
            any(item.is_symlink() or not item.is_file() for item in files)
            or len(set(resolved_files)) != len(resolved_files)
            or database in resolved_files
            or state in resolved_files
            or database == state
            or database in state.parents
            or state in database.parents
            or not isinstance(self.memory_limit, str)
            or not self.memory_limit
            or isinstance(self.database_threads, bool)
            or not 1 <= int(self.database_threads) <= 8
        ):
            raise ValueError("Round 17 campaign operator configuration is invalid")
        return self


@dataclass(frozen=True, slots=True)
class Round17CampaignSlotSource:
    slot_index: int
    scheduled_start_ms: int
    status: str
    run_id: str
    slot_result_sha256: str


@dataclass(frozen=True, slots=True)
class Round17CampaignReadiness:
    campaign_plan: PolymarketRound14CampaignPlan
    cohort_plan: Round17CohortPlan
    admission_spec: PolymarketRound14AdmissionSpec
    slot_sources: tuple[Round17CampaignSlotSource, ...]
    terminal_slot_count: int
    active_slot_indexes: tuple[int, ...]
    database_bytes: int
    wal_bytes: int
    gates: Mapping[str, bool]
    readiness_sha256: str

    @property
    def ready(self) -> bool:
        return all(self.gates.values())

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND17_CAMPAIGN_OPERATOR_SCHEMA_VERSION,
            "campaign_plan_sha256": self.campaign_plan.plan_sha256,
            "cohort_plan_sha256": self.cohort_plan.plan_sha256,
            "admission_spec_sha256": self.admission_spec.spec_sha256,
            "terminal_slot_count": self.terminal_slot_count,
            "active_slot_indexes": list(self.active_slot_indexes),
            "slot_sources": [
                {
                    "slot_index": item.slot_index,
                    "scheduled_start_ms": item.scheduled_start_ms,
                    "status": item.status,
                    "run_id": item.run_id,
                    "slot_result_sha256": item.slot_result_sha256,
                }
                for item in self.slot_sources
            ],
            "database_bytes": self.database_bytes,
            "wal_bytes": self.wal_bytes,
            "gates": dict(self.gates),
            "database_opened": False,
            "test_features_accessed": False,
            "test_targets_accessed": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "ready": self.ready,
            "readiness_sha256": self.readiness_sha256,
        }

    def validated(self) -> Round17CampaignReadiness:
        if (
            _SHA256.fullmatch(self.readiness_sha256) is None
            or self.readiness_sha256 != _canonical_sha256(self.identity_payload())
            or self.terminal_slot_count < 0
            or self.database_bytes < 0
            or self.wal_bytes < 0
            or tuple(sorted(set(self.active_slot_indexes))) != self.active_slot_indexes
            or any(type(value) is not bool for value in self.gates.values())
        ):
            raise ValueError("Round 17 campaign readiness integrity differs")
        return self


def _validate_campaign_identity(
    campaign: PolymarketRound14CampaignPlan,
    cohort: Round17CohortPlan,
    spec: PolymarketRound14AdmissionSpec,
) -> None:
    if (
        campaign.plan_sha256 != POLYMARKET_ROUND17_CAMPAIGN_PLAN_SHA256
        or campaign.contract_sha256 != POLYMARKET_ROUND17_CAPTURE_CONTRACT_SHA256
        or campaign.scheduled_start_ms != cohort.roles[0].start_ms
        or campaign.scheduled_end_ms != cohort.roles[-1].end_ms_exclusive
        or campaign.total_slots != 1_440
        or spec.campaign_plan_sha256 != campaign.plan_sha256
        or spec.contract_sha256 != campaign.contract_sha256
    ):
        raise ValueError("Round 17 campaign operator parent identity differs")


def _slot_result(
    *,
    path: Path,
    campaign: PolymarketRound14CampaignPlan,
    expected_index: int,
) -> tuple[dict[str, object], Round17CampaignSlotSource | None]:
    payload = _read_hashed_json(path, label="Round 14 campaign slot result")
    details = payload.get("details")
    status = str(payload.get("status") or "")
    scheduled_start_ms = campaign.scheduled_slot_ms(expected_index)
    if (
        payload.get("schema_version")
        != POLYMARKET_ROUND14_CAMPAIGN_SLOT_RESULT_SCHEMA_VERSION
        or payload.get("plan_sha256") != campaign.plan_sha256
        or payload.get("slot_index") != expected_index
        or payload.get("scheduled_start_ms") != scheduled_start_ms
        or status not in _TERMINAL_SLOT_STATUSES
        or not isinstance(details, Mapping)
        or payload.get("model_data_eligible") is not False
        or any(
            payload.get(name) is not False
            for name in (
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
        or payload.get("condition_level_admission_required")
        is not (status in _MATERIALIZABLE_SLOT_STATUSES)
    ):
        raise ValueError("Round 14 campaign slot result integrity differs")
    if status not in _MATERIALIZABLE_SLOT_STATUSES:
        return payload, None
    run_id = str(details.get("run_id") or "")
    if (
        _RUN_ID.fullmatch(run_id) is None
        or _SHA256.fullmatch(str(details.get("report_sha256") or "")) is None
        or details.get("errors") != []
        or details.get("integrity_errors") != []
    ):
        raise ValueError("Round 14 materializable slot evidence differs")
    return payload, Round17CampaignSlotSource(
        slot_index=expected_index,
        scheduled_start_ms=scheduled_start_ms,
        status=status,
        run_id=run_id,
        slot_result_sha256=str(payload["artifact_sha256"]),
    )


def inspect_round17_campaign_readiness(
    config: Round17CampaignOperatorConfig,
) -> Round17CampaignReadiness:
    """Inspect immutable JSON/filesystem gates without opening the database."""

    selected = config.validate()
    campaign = load_round14_campaign_plan(selected.campaign_plan_path)
    cohort = load_round17_cohort_plan(selected.cohort_plan_path)
    spec = load_round14_admission_spec(selected.admission_spec_path)
    _validate_campaign_identity(campaign, cohort, spec)
    state = _read_hashed_json(
        selected.state_root / "campaign-state.json",
        label="Round 14 campaign state",
    )
    active = state.get("active_slot_indexes")
    terminal_slot_count = state.get("terminal_slot_count")
    next_slot_index = state.get("next_slot_index")
    if (
        state.get("schema_version") != POLYMARKET_ROUND14_CAMPAIGN_STATE_SCHEMA_VERSION
        or state.get("plan_sha256") != campaign.plan_sha256
        or not isinstance(active, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in active
        )
        or active != sorted(set(active))
        or any(not 0 <= value < campaign.total_slots for value in active)
        or isinstance(terminal_slot_count, bool)
        or not isinstance(terminal_slot_count, int)
        or not 0 <= terminal_slot_count <= campaign.total_slots
        or isinstance(next_slot_index, bool)
        or not isinstance(next_slot_index, int)
        or not 0 <= next_slot_index <= campaign.total_slots
        or any(
            state.get(name) is not False
            for name in (
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
    ):
        raise ValueError("Round 14 campaign state integrity differs")
    slot_root = selected.state_root / "slots"
    slot_paths = tuple(sorted(slot_root.glob("slot-*.json")))
    expected_paths = tuple(
        slot_root / f"slot-{index:04d}.json" for index in range(len(slot_paths))
    )
    if slot_paths != expected_paths:
        raise ValueError("Round 14 campaign slot files are not contiguous")
    sources: list[Round17CampaignSlotSource] = []
    for index, path in enumerate(slot_paths):
        _payload, source = _slot_result(
            path=path,
            campaign=campaign,
            expected_index=index,
        )
        if source is not None:
            sources.append(source)
    database_bytes = (
        selected.database_path.stat().st_size
        if selected.database_path.is_file() and not selected.database_path.is_symlink()
        else 0
    )
    wal_path = Path(f"{selected.database_path}.wal")
    wal_bytes = wal_path.stat().st_size if wal_path.is_file() else 0
    gates = {
        "all_slots_terminal": terminal_slot_count == campaign.total_slots,
        "slot_file_count_complete": len(slot_paths) == campaign.total_slots,
        "state_slot_count_matches_files": terminal_slot_count == len(slot_paths),
        "next_slot_is_terminal": next_slot_index == campaign.total_slots,
        "no_active_slot": not active,
        "database_exists": database_bytes > 0,
        "database_size_matches_state": (state.get("database_bytes") == database_bytes),
        "wal_absent": wal_bytes == 0,
    }
    provisional = Round17CampaignReadiness(
        campaign_plan=campaign,
        cohort_plan=cohort,
        admission_spec=spec,
        slot_sources=tuple(sources),
        terminal_slot_count=terminal_slot_count,
        active_slot_indexes=tuple(active),
        database_bytes=database_bytes,
        wal_bytes=wal_bytes,
        gates=gates,
        readiness_sha256="0" * 64,
    )
    return Round17CampaignReadiness(
        campaign_plan=provisional.campaign_plan,
        cohort_plan=provisional.cohort_plan,
        admission_spec=provisional.admission_spec,
        slot_sources=provisional.slot_sources,
        terminal_slot_count=provisional.terminal_slot_count,
        active_slot_indexes=provisional.active_slot_indexes,
        database_bytes=provisional.database_bytes,
        wal_bytes=provisional.wal_bytes,
        gates=provisional.gates,
        readiness_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


@dataclass(frozen=True, slots=True)
class Round17DevelopmentConditionMaterialization:
    source: Round17CampaignSlotSource
    dataset: PolymarketRound17ConditionDataset
    cohort_condition: Round17CohortCondition


def _development_role(
    cohort: Round17CohortPlan,
    *,
    event_start_ms: int,
    event_end_ms: int,
    source_slot_index: int,
) -> str | None:
    campaign_start_ms = cohort.roles[0].start_ms
    campaign_end_ms = cohort.roles[-1].end_ms_exclusive
    if (
        event_start_ms < campaign_start_ms
        or event_start_ms >= campaign_end_ms
        or (event_start_ms - campaign_start_ms) // 1_800_000 != source_slot_index
    ):
        return None
    role = cohort.role_for_condition(
        event_start_ms=event_start_ms,
        event_end_ms=event_end_ms,
        source_slot_index=source_slot_index,
    ).name
    return role if role in _DEVELOPMENT_ROLES else None


def iter_round17_campaign_development_conditions(
    config: Round17CampaignOperatorConfig,
) -> Iterator[Round17DevelopmentConditionMaterialization]:
    """Read each terminal development run once; never inspect reserved test slots."""

    readiness = inspect_round17_campaign_readiness(config)
    if not readiness.ready:
        failed = ",".join(
            name for name, passed in readiness.gates.items() if not passed
        )
        raise RuntimeError(f"Round 17 campaign is not terminal: {failed}")
    sources = tuple(
        source
        for source in readiness.slot_sources
        if any(
            role.first_slot <= source.slot_index <= role.last_slot
            and role.name in _DEVELOPMENT_ROLES
            for role in readiness.cohort_plan.roles
        )
    )
    with PolymarketEvidenceStore(
        config.database_path,
        memory_limit=config.memory_limit,
        threads=config.database_threads,
        read_only=True,
    ) as store:
        for source in sources:
            markets = PolymarketEvidenceReplay.load_markets(
                store,
                run_id=source.run_id,
            )
            scoped_markets = tuple(
                market
                for market in markets
                if _development_role(
                    readiness.cohort_plan,
                    event_start_ms=market.event_start_ms,
                    event_end_ms=market.end_ms,
                    source_slot_index=source.slot_index,
                )
                is not None
            )
            if not scoped_markets:
                continue
            context = materialize_round14_label_free_run_context(
                store,
                run_id=source.run_id,
                spec=readiness.admission_spec,
                diagnostic_only=False,
                condition_ids=tuple(market.condition_id for market in scoped_markets),
            )
            admission_by_condition = {
                item.condition_id: item for item in context.dataset.admissions
            }
            rows_by_condition = {
                condition_id: tuple(
                    row
                    for row in context.dataset.rows
                    if row.condition_id == condition_id
                )
                for condition_id in admission_by_condition
            }
            market_by_condition = {
                market.condition_id: market for market in context.markets
            }
            for condition_id in sorted(
                market_by_condition,
                key=lambda value: (
                    market_by_condition[value].event_start_ms,
                    value,
                ),
            ):
                admission = admission_by_condition[condition_id]
                if not admission.core_eligible:
                    continue
                market = market_by_condition[condition_id]
                books = tuple(
                    book
                    for book in context.replay.books
                    if book.market.condition_id == condition_id
                )
                dataset = materialize_round17_condition_rows(
                    market=market,
                    admission=admission,
                    base_rows=rows_by_condition[condition_id],
                    events=context.feature_events,
                    books=books,
                )
                cohort_condition = build_round17_cohort_condition(
                    readiness.cohort_plan,
                    dataset,
                    source_slot_index=source.slot_index,
                )
                yield Round17DevelopmentConditionMaterialization(
                    source=source,
                    dataset=dataset,
                    cohort_condition=cohort_condition,
                )


__all__ = [
    "POLYMARKET_ROUND17_CAMPAIGN_OPERATOR_SCHEMA_VERSION",
    "Round17CampaignOperatorConfig",
    "Round17CampaignReadiness",
    "Round17CampaignSlotSource",
    "Round17DevelopmentConditionMaterialization",
    "inspect_round17_campaign_readiness",
    "iter_round17_campaign_development_conditions",
]
