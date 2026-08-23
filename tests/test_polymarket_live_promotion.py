from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from polymarket_live_support import (
    POLYMARKET_LIVE_PROMOTION_GATES,
    build_polymarket_live_promotion_fixture,
)
from simple_ai_trading.polymarket_live import PolymarketLiveBlocked
from simple_ai_trading.polymarket_live_promotion import (
    PolymarketLivePromotion,
    PolymarketPromotionEvidence,
    VerifiedPolymarketLivePromotion,
    load_polymarket_live_promotion,
    validate_polymarket_live_promotion,
)


NOW_MS = 1_800_000_000_000


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(
    root: Path,
    *,
    live: bool = True,
    market_variant: str = "fiveminute",
) -> dict[str, object]:
    return build_polymarket_live_promotion_fixture(
        root,
        now_ms=NOW_MS,
        live=live,
        market_variant=market_variant,
    )


def test_live_promotion_is_hash_and_evidence_bound(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    path = tmp_path / "promotion.json"
    path.write_text(_canonical(payload), encoding="ascii")

    verified = load_polymarket_live_promotion(
        path,
        evidence_root=tmp_path,
        observed_at_ms=NOW_MS,
    )
    promotion = verified.promotion

    assert promotion.live_authority is True
    assert promotion.market_variant == "fiveminute"
    assert promotion.model_artifact.sha256 == _file_sha(tmp_path / "model.json")
    assert all(promotion.gates.values())
    assert verified.model_artifact_path == (tmp_path / "model.json").resolve()
    assert verified.cross_regime_evaluation.aggregate.passed is True
    assert verified.cross_regime_evaluation.risk_profile == "conservative"


def test_unverified_promotion_cannot_construct_verified_capability() -> None:
    evidence = PolymarketPromotionEvidence(path="model.json", sha256="c" * 64)
    promotion = PolymarketLivePromotion(
        promotion_id="a" * 64,
        promotion_sha256="d" * 64,
        created_at_ms=NOW_MS - 1_000,
        expires_at_ms=NOW_MS + 1_000,
        source_commit="b" * 40,
        bot_id="simple-ai-trading-polymarket-btc",
        market_variant="fiveminute",
        risk_profile="conservative",
        model_artifact=evidence,
        evaluation_report=evidence,
        implementation_manifest=evidence,
        gates=POLYMARKET_LIVE_PROMOTION_GATES,
        minimum_expected_edge_quote_per_share="0.02",
        maximum_prediction_age_ms=1_000,
        minimum_remaining_seconds=30,
        paper_authority=True,
        live_authority=True,
    )

    with pytest.raises(ValueError, match="evidence was not verified"):
        VerifiedPolymarketLivePromotion(
            promotion=promotion,
            model_artifact_path=Path("model.json"),
            evaluation_report_path=Path("model.json"),
            implementation_manifest_path=Path("model.json"),
            implementation=object(),  # type: ignore[arg-type]
            cross_regime_evaluation=object(),  # type: ignore[arg-type]
            _capability=object(),
        )


def test_live_promotion_rejects_payload_or_evidence_tampering(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    payload["bot_id"] = "tampered-bot"
    with pytest.raises(ValueError, match="promotion hash differs"):
        validate_polymarket_live_promotion(payload)

    payload = _payload(tmp_path)
    path = tmp_path / "promotion.json"
    path.write_text(_canonical(payload), encoding="ascii")
    (tmp_path / "model.json").write_text('{"model":"tampered"}\n', encoding="ascii")
    with pytest.raises(ValueError, match="evidence hash differs"):
        load_polymarket_live_promotion(
            path,
            evidence_root=tmp_path,
            observed_at_ms=NOW_MS,
        )


def test_live_promotion_binds_the_exact_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "promotion.json"
    path.write_text(_canonical(_payload(tmp_path)), encoding="ascii")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    verified = load_polymarket_live_promotion(
        path,
        evidence_root=tmp_path,
        observed_at_ms=NOW_MS,
        expected_file_sha256=expected,
    )
    assert verified.promotion.live_authority is True

    with pytest.raises(ValueError, match="promotion file hash differs"):
        load_polymarket_live_promotion(
            path,
            evidence_root=tmp_path,
            observed_at_ms=NOW_MS,
            expected_file_sha256="0" * 64,
        )


def test_live_promotion_rejects_missing_gate_or_live_authority(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    gates = dict(payload["gates"])
    gates.pop("after_cost_edge")
    body = {**payload, "gates": gates}
    body.pop("promotion_sha256")
    body["promotion_sha256"] = _sha(body)
    with pytest.raises(ValueError, match="gate set"):
        validate_polymarket_live_promotion(body)

    payload = _payload(tmp_path, live=False)
    path = tmp_path / "promotion.json"
    path.write_text(_canonical(payload), encoding="ascii")
    with pytest.raises(PolymarketLiveBlocked, match="no live authority"):
        load_polymarket_live_promotion(
            path,
            evidence_root=tmp_path,
            observed_at_ms=NOW_MS,
        )


def test_live_promotion_rejects_expired_or_escaping_evidence(
    tmp_path: Path,
) -> None:
    payload = _payload(tmp_path)
    path = tmp_path / "promotion.json"
    path.write_text(_canonical(payload), encoding="ascii")
    with pytest.raises(PolymarketLiveBlocked, match="expired"):
        load_polymarket_live_promotion(
            path,
            evidence_root=tmp_path,
            observed_at_ms=NOW_MS + 86_400_000,
        )

    payload = _payload(tmp_path)
    payload["model_artifact"] = {
        "path": "../model.json",
        "sha256": payload["model_artifact"]["sha256"],
    }
    body = dict(payload)
    body.pop("promotion_sha256")
    body["promotion_sha256"] = _sha(body)
    with pytest.raises(ValueError, match="safe repository-relative"):
        validate_polymarket_live_promotion(body)


def test_live_promotion_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "promotion.json"
    path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="ascii")

    with pytest.raises(ValueError, match="duplicate keys"):
        load_polymarket_live_promotion(path, evidence_root=tmp_path)


def test_live_promotion_rejects_hash_valid_dummy_evaluation(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text('{"evaluation":"passed"}', encoding="ascii")
    payload["evaluation_report"] = {
        "path": evaluation_path.name,
        "sha256": _file_sha(evaluation_path),
    }
    body = dict(payload)
    body.pop("promotion_sha256")
    body["promotion_sha256"] = _sha(body)
    promotion_path = tmp_path / "promotion.json"
    promotion_path.write_text(_canonical(body), encoding="ascii")

    with pytest.raises(ValueError, match="evaluation schema is invalid"):
        load_polymarket_live_promotion(
            promotion_path,
            evidence_root=tmp_path,
            observed_at_ms=NOW_MS,
        )


def test_live_promotion_accepts_only_five_or_fifteen_minute_scope(
    tmp_path: Path,
) -> None:
    promotion = validate_polymarket_live_promotion(
        _payload(tmp_path, market_variant="fifteenminute")
    )
    assert promotion.market_variant == "fifteenminute"

    payload = _payload(tmp_path, market_variant="hourly")
    with pytest.raises(ValueError, match="scope is invalid"):
        validate_polymarket_live_promotion(payload)
