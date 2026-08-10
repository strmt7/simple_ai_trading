from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import duckdb
import pytest

from simple_ai_trading import polymarket_round25_resolution_store as resolution_store
from simple_ai_trading.polymarket_round25_campaign import (
    POLYMARKET_ROUND25_DESIGN_SHA256,
    POLYMARKET_ROUND25_RESOLUTION_SOURCE,
)
from simple_ai_trading.polymarket_round25_dataset import (
    POLYMARKET_ROUND25_CALIBRATION_END_MS,
    POLYMARKET_ROUND25_CAMPAIGN_START_MS,
    POLYMARKET_ROUND25_CAPTURE_PLAN_SHA256,
    POLYMARKET_ROUND25_SELECTION_END_MS,
    POLYMARKET_ROUND25_TRAIN_END_MS,
)
from simple_ai_trading.polymarket_round25_evaluation import (
    POLYMARKET_ROUND25_CANDIDATE_IDS,
    Round25SelectionAccessStore,
    create_round25_prediction_panel,
)
from simple_ai_trading.polymarket_round25_joint_materialization import (
    Round25JointReceiptCondition,
    reject_round25_joint_condition,
)
from simple_ai_trading.polymarket_round25_joint_store import (
    Round25JointStoreWriter,
    audit_round25_joint_store,
)
from simple_ai_trading.polymarket_round25_resolution_store import (
    POLYMARKET_ROUND25_RESOLUTION_CONTRACT_SHA256,
    Round25OfficialPublicPayload,
    Round25ResolutionPublicClient,
    audit_round25_resolution_collection,
    audit_round25_resolution_store,
    collect_round25_resolutions_once,
    finalize_round25_resolution_store,
    initialize_round25_resolution_collection,
    load_round25_fit_resolution_inputs,
    load_round25_resolution_access_claim,
    load_round25_resolution_source_conditions,
    load_round25_selection_resolution_inputs,
)
from simple_ai_trading.polymarket_round25_terminal import (
    POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256,
    POLYMARKET_ROUND25_TERMINAL_RECEIPT_AUDIT_SCHEMA_VERSION,
    POLYMARKET_ROUND25_TERMINAL_TRANSPORT_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-official-resolution-collection-contract-v1.json"
)
RUN_ID = "3" * 32
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
OPENED_AT_MS = POLYMARKET_ROUND25_SELECTION_END_MS + 10_000


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


def _transport() -> dict[str, object]:
    duration_ms = (
        POLYMARKET_ROUND25_SELECTION_END_MS
        - POLYMARKET_ROUND25_CAMPAIGN_START_MS
    )
    segment = {
        "condition_count": 3,
        "duration_seconds": duration_ms / 1_000.0,
        "eligible_for_condition_rebuild": True,
        "ended_at_ms": POLYMARKET_ROUND25_SELECTION_END_MS,
        "errors": [],
        "exclusion_reasons": [],
        "integrity_errors": [],
        "observed_at_ms": POLYMARKET_ROUND25_SELECTION_END_MS,
        "raw_message_count": 2,
        "report_sha256": "9" * 64,
        "resolution_source": POLYMARKET_ROUND25_RESOLUTION_SOURCE,
        "run_id": RUN_ID,
        "segment_index": 0,
        "source_manifest_sha256": "8" * 64,
        "source_result_sha256": "7" * 64,
        "started_at_ms": POLYMARKET_ROUND25_CAMPAIGN_START_MS,
        "status": "complete",
        "stream_counts": {"clob_market": 1, "polymarket_rtds": 1},
        "stream_gap_count": 0,
    }
    interval = {
        "duration_ms": duration_ms,
        "end_ms": POLYMARKET_ROUND25_SELECTION_END_MS,
        "segment_index": 0,
        "start_ms": POLYMARKET_ROUND25_CAMPAIGN_START_MS,
    }
    body: dict[str, object] = {
        "all_scheduled_transport_interval_covered": True,
        "campaign_end_ms": POLYMARKET_ROUND25_SELECTION_END_MS,
        "campaign_start_ms": POLYMARKET_ROUND25_CAMPAIGN_START_MS,
        "campaign_state_artifact_sha256": "6" * 64,
        "campaign_status": "campaign_window_ended",
        "condition_admission_pending": True,
        "created_at_ms": POLYMARKET_ROUND25_SELECTION_END_MS,
        "eligible_run_ids": [RUN_ID],
        "known_ineligible_or_unobserved_intervals": [],
        "live_trading_authority": False,
        "model_data_eligible": False,
        "model_scores_consulted": False,
        "outcomes_consulted": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
        "provisional_eligible_transport_intervals": [interval],
        "resolution_source": POLYMARKET_ROUND25_RESOLUTION_SOURCE,
        "schema_version": POLYMARKET_ROUND25_TERMINAL_TRANSPORT_SCHEMA_VERSION,
        "segments": [segment],
        "source_capture_design_sha256": POLYMARKET_ROUND25_DESIGN_SHA256,
        "source_plan_sha256": POLYMARKET_ROUND25_CAPTURE_PLAN_SHA256,
        "source_qualification_sha256": "5" * 64,
        "terminal_design_sha256": POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256,
    }
    return {**body, "manifest_sha256": _canonical_sha256(body)}


def _receipt_audit(transport: dict[str, object]) -> dict[str, object]:
    eligible_run = {
        "first_gap_opened_at_ms": None,
        "first_receipt_wall_ms": POLYMARKET_ROUND25_CAMPAIGN_START_MS,
        "gap_chain_sha256": EMPTY_SHA256,
        "gap_count": 0,
        "last_gap_opened_at_ms": None,
        "last_receipt_wall_ms": POLYMARKET_ROUND25_CAMPAIGN_START_MS + 1,
        "preregistration_manifest_sha256": "8" * 64,
        "receipt_chain_sha256": "a" * 64,
        "receipt_count": 2,
        "report_sha256": "9" * 64,
        "run_id": RUN_ID,
        "segment_index": 0,
        "status": "complete",
        "stream_counts": {"clob_market": 1, "polymarket_rtds": 1},
    }
    body: dict[str, object] = {
        "condition_admission_pending": True,
        "created_at_ms": POLYMARKET_ROUND25_SELECTION_END_MS,
        "database_run_count": 1,
        "eligible_runs": [eligible_run],
        "ineligible_runs": [],
        "live_trading_authority": False,
        "model_data_eligible": False,
        "model_scores_consulted": False,
        "outcomes_consulted": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
        "receipt_replay_complete": True,
        "schema_version": POLYMARKET_ROUND25_TERMINAL_RECEIPT_AUDIT_SCHEMA_VERSION,
        "terminal_design_sha256": POLYMARKET_ROUND25_TERMINAL_DESIGN_SHA256,
        "terminal_transport_manifest_sha256": transport["manifest_sha256"],
    }
    return {**body, "audit_sha256": _canonical_sha256(body)}


def _conditions() -> tuple[Round25JointReceiptCondition, ...]:
    starts = (
        POLYMARKET_ROUND25_CAMPAIGN_START_MS + 300_000,
        POLYMARKET_ROUND25_TRAIN_END_MS + 300_000,
        POLYMARKET_ROUND25_CALIBRATION_END_MS + 300_000,
    )
    roles = ("train", "calibration", "selection")
    output = []
    for index, (start, role) in enumerate(zip(starts, roles, strict=True), start=1):
        output.append(
            Round25JointReceiptCondition(
                run_id=RUN_ID,
                segment_index=0,
                snapshot_sha256=str(index) * 64,
                snapshot_observed_wall_ms=start - 10_000,
                market_id=str(1000 + index),
                condition_id="0x" + str(index) * 64,
                slug=f"btc-updown-5m-{start // 1_000}",
                event_start_ms=start,
                event_end_ms=start + 300_000,
                up_token_id=str(index) * 40,
                down_token_id=str(index + 3) * 40,
                resolution_source=POLYMARKET_ROUND25_RESOLUTION_SOURCE,
                role=role,
            ).validated()
        )
    return tuple(output)


def _feature_store(path: Path) -> tuple[Round25JointReceiptCondition, ...]:
    transport = _transport()
    conditions = _conditions()
    writer = Round25JointStoreWriter(
        path,
        terminal_transport_manifest=transport,
    )
    for condition in conditions:
        writer.add(
            reject_round25_joint_condition(
                condition=condition,
                source_record_count=0,
                rejection_reasons=("fixture_target_free_rejection",),
            )
        )
    writer.finalize(
        terminal_receipt_audit=_receipt_audit(transport),
        source_counts={
            "admitted_condition_count": 3,
            "calibration_condition_count": 1,
            "purged_condition_count": 0,
            "selection_condition_count": 1,
            "source_snapshot_count": 3,
            "train_condition_count": 1,
        },
    )
    audit_round25_joint_store(path)
    return conditions


def _payload(value: Mapping[str, object], observed: int) -> Round25OfficialPublicPayload:
    canonical = _canonical_json(value)
    return Round25OfficialPublicPayload(
        value=dict(value),
        canonical_json=canonical,
        sha256=hashlib.sha256(canonical.encode("ascii")).hexdigest(),
        observed_wall_ms=observed,
        observed_monotonic_ns=observed * 1_000_000,
    )


def _gamma(condition: Round25JointReceiptCondition, *, closed: bool = True) -> dict[str, object]:
    return {
        "acceptingOrders": not closed,
        "closed": closed,
        "clobTokenIds": [condition.up_token_id, condition.down_token_id],
        "conditionId": condition.condition_id,
        "id": condition.market_id,
        "outcomePrices": ["1", "0"] if closed else ["0.52", "0.48"],
        "outcomes": ["Up", "Down"],
        "resolutionSource": condition.resolution_source,
        "slug": condition.slug,
    }


def _clob(condition: Round25JointReceiptCondition, *, winner: str = "Up") -> dict[str, object]:
    return {
        "accepting_orders": False,
        "closed": True,
        "condition_id": condition.condition_id,
        "market_slug": condition.slug,
        "tokens": [
            {
                "outcome": "Up",
                "price": "1" if winner == "Up" else "0",
                "token_id": condition.up_token_id,
                "winner": winner == "Up",
            },
            {
                "outcome": "Down",
                "price": "1" if winner == "Down" else "0",
                "token_id": condition.down_token_id,
                "winner": winner == "Down",
            },
        ],
    }


class _Client:
    def __init__(
        self,
        conditions: tuple[Round25JointReceiptCondition, ...],
        *,
        pending: bool = False,
        disagree: bool = False,
    ) -> None:
        self.by_market = {item.market_id: item for item in conditions}
        self.by_condition = {item.condition_id: item for item in conditions}
        self.pending = pending
        self.disagree = disagree

    def gamma_market(self, market_id: str) -> Round25OfficialPublicPayload:
        condition = self.by_market[market_id]
        return _payload(
            _gamma(condition, closed=not self.pending),
            OPENED_AT_MS,
        )

    def clob_market(self, condition_id: str) -> Round25OfficialPublicPayload:
        condition = self.by_condition[condition_id]
        return _payload(
            _clob(condition, winner="Down" if self.disagree else "Up"),
            OPENED_AT_MS + 1,
        )


def _selection_panel(condition: Round25JointReceiptCondition) -> object:
    count = 16
    prior = tuple(0.5 for _ in range(count))
    probabilities = {
        candidate_id: (
            prior
            if candidate_id == "market-prior-v1"
            else tuple(0.5 + index * 0.001 for _ in range(count))
        )
        for index, candidate_id in enumerate(POLYMARKET_ROUND25_CANDIDATE_IDS)
    }
    sources = {
        candidate_id: hashlib.sha256(candidate_id.encode("ascii")).hexdigest()
        for candidate_id in POLYMARKET_ROUND25_CANDIDATE_IDS
    }
    return create_round25_prediction_panel(
        row_condition_ids=(condition.condition_id,) * count,
        event_start_ms=(condition.event_start_ms,) * count,
        decision_time_ms=tuple(
            condition.event_start_ms + index * 10_000 for index in range(count)
        ),
        feature_source_chain_sha256=tuple(
            hashlib.sha256(f"selection:{index}".encode("ascii")).hexdigest()
            for index in range(count)
        ),
        market_prior_probability=prior,
        candidate_probabilities=probabilities,
        candidate_source_artifact_sha256=sources,
    )


def _open(tmp_path: Path) -> tuple[Path, Path, tuple[Round25JointReceiptCondition, ...]]:
    feature = tmp_path / "round25-features.duckdb"
    conditions = _feature_store(feature)
    destination = tmp_path / "round25-resolutions.duckdb"
    collection, _ = initialize_round25_resolution_collection(
        feature_database=feature,
        destination_database=destination,
        created_at_ms=OPENED_AT_MS,
    )
    return feature, collection, conditions


def test_resolution_contract_is_self_hashed_and_target_blind() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    claimed = payload.pop("contract_sha256")

    assert claimed == _canonical_sha256(payload)
    assert claimed == POLYMARKET_ROUND25_RESOLUTION_CONTRACT_SHA256
    assert payload["resolution_semantics"]["binance_signal_allowed_for_resolution"] is False
    assert payload["authority"]["live_trading_authority"] is False


def test_target_opening_persists_claim_and_exact_source_population(tmp_path: Path) -> None:
    feature, collection, conditions = _open(tmp_path)

    claim = load_round25_resolution_access_claim(collection)
    stored = load_round25_resolution_source_conditions(collection)

    assert feature.exists()
    assert collection.exists()
    assert claim["target_access_opened"] is True
    assert claim["condition_count"] == 3
    assert stored == conditions


def test_target_database_must_be_distinct_from_feature_store(tmp_path: Path) -> None:
    feature = tmp_path / "round25-features.duckdb"
    _feature_store(feature)

    with pytest.raises(ValueError, match="database boundary differs"):
        initialize_round25_resolution_collection(
            feature_database=feature,
            destination_database=feature,
            created_at_ms=OPENED_AT_MS,
        )


def test_unresolved_official_sources_remain_pending_without_a_label(tmp_path: Path) -> None:
    _, collection, conditions = _open(tmp_path)

    report = collect_round25_resolutions_once(
        collection_database=collection,
        client=_Client(conditions, pending=True),
    )
    audit = audit_round25_resolution_collection(collection)

    assert report["newly_resolved_condition_count"] == 0
    assert report["pending_condition_count"] == 3
    assert audit["resolved_condition_count"] == 0
    assert audit["evidence_chain_sha256"] == EMPTY_SHA256


def test_dual_source_winner_disagreement_aborts_the_batch(tmp_path: Path) -> None:
    _, collection, conditions = _open(tmp_path)

    with pytest.raises(ValueError, match="disagree"):
        collect_round25_resolutions_once(
            collection_database=collection,
            client=_Client(conditions, disagree=True),
        )

    assert audit_round25_resolution_collection(collection)["resolved_condition_count"] == 0


def test_resolved_payloads_round_trip_through_deep_audit(tmp_path: Path) -> None:
    _, collection, conditions = _open(tmp_path)

    report = collect_round25_resolutions_once(
        collection_database=collection,
        client=_Client(conditions),
    )
    audit = audit_round25_resolution_collection(collection)

    assert report["finalization_ready"] is True
    assert report["newly_resolved_condition_count"] == 3
    assert audit["pending_condition_count"] == 0
    assert audit["resolved_role_counts"] == {
        "train": 1,
        "calibration": 1,
        "selection": 1,
    }


def test_compressed_target_tampering_is_detected(tmp_path: Path) -> None:
    _, collection, conditions = _open(tmp_path)
    collect_round25_resolutions_once(
        collection_database=collection,
        client=_Client(conditions),
    )
    with duckdb.connect(str(collection)) as connection:
        blob = bytes(
            connection.execute(
                "SELECT gamma_payload FROM round25_resolution_evidence LIMIT 1"
            ).fetchone()[0]
        )
        changed = bytes([blob[0] ^ 1]) + blob[1:]
        connection.execute(
            """
            UPDATE round25_resolution_evidence SET gamma_payload = ?
            WHERE condition_id = (SELECT condition_id FROM round25_resolution_evidence LIMIT 1)
            """,
            [changed],
        )

    with pytest.raises(ValueError, match="Gamma resolution payload envelope differs"):
        audit_round25_resolution_collection(collection)


def test_incomplete_collection_cannot_create_resolution_authority(tmp_path: Path) -> None:
    feature, _, _ = _open(tmp_path)

    with pytest.raises(ValueError, match="finalization gate is closed"):
        finalize_round25_resolution_store(
            feature_database=feature,
            destination_database=tmp_path / "round25-resolutions.duckdb",
            created_at_ms=OPENED_AT_MS,
        )


def test_complete_collection_atomically_publishes_final_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature, collection, conditions = _open(tmp_path)
    destination = tmp_path / "round25-resolutions.duckdb"
    for role in ("train", "calibration", "selection"):
        monkeypatch.setitem(
            resolution_store.POLYMARKET_ROUND25_MINIMUM_CONDITIONS,
            role,
            1,
        )
    collect_round25_resolutions_once(
        collection_database=collection,
        client=_Client(conditions),
    )

    manifest = finalize_round25_resolution_store(
        feature_database=feature,
        destination_database=destination,
        created_at_ms=OPENED_AT_MS,
    )

    assert destination.exists()
    assert not collection.exists()
    assert not Path(f"{destination}.wal").exists()
    assert manifest == audit_round25_resolution_store(destination)
    assert manifest["resolution_count"] == 3
    assert manifest["model_data_eligible"] is False
    assert manifest["live_trading_authority"] is False
    fit_manifest, authority, fit_roles = load_round25_fit_resolution_inputs(
        destination
    )
    assert fit_manifest == manifest
    assert authority.authority_sha256 == manifest["authority_sha256"]
    assert len(fit_roles["train"]) == 1
    assert len(fit_roles["calibration"]) == 1
    panel = _selection_panel(conditions[2])
    access_store = Round25SelectionAccessStore(
        tmp_path / "round25-selection-access.sqlite3"
    )
    access_store.freeze_prediction_panel(
        panel=panel,
        one_use_claim_sha256="b" * 64,
    )
    selection_manifest, selection_authority, selection, claim_sha256 = (
        load_round25_selection_resolution_inputs(
            destination,
            panel=panel,
            access_store=access_store,
        )
    )
    assert selection_manifest == manifest
    assert selection_authority == authority
    assert len(selection) == 1
    assert claim_sha256 == "b" * 64


class _Cookies(list[object]):
    def clear(self) -> None:
        super().clear()


class _AuthenticatedSession:
    headers = {"Authorization": "not-a-real-secret"}
    cookies = _Cookies()

    def get(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("authenticated session must fail before network access")


def test_public_client_rejects_authority_headers_before_request() -> None:
    client = Round25ResolutionPublicClient(session=_AuthenticatedSession())

    with pytest.raises(ValueError, match="authority headers"):
        client.gamma_market("123")
