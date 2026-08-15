from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_live import PolymarketLiveBlocked
from simple_ai_trading.polymarket_live_activation import (
    POLYMARKET_LIVE_ACTIVATION_SCHEMA_VERSION,
    PolymarketLiveActivation,
    VerifiedPolymarketLiveActivation,
    load_polymarket_live_activation,
    write_polymarket_live_activation,
)


def _sources(tmp_path: Path) -> dict[str, Path]:
    evidence = tmp_path / "evidence"
    evidence.mkdir(parents=True)
    promotion = evidence / "promotion.json"
    lifecycle = evidence / "lifecycle.json"
    round16 = evidence / "round16.json"
    promotion.write_text('{"promotion":true}\n', encoding="utf-8")
    lifecycle.write_text('{"qualification":true}\n', encoding="utf-8")
    round16.write_text('{"contract":true}\n', encoding="utf-8")
    return {
        "evidence": evidence,
        "promotion": promotion,
        "lifecycle": lifecycle,
        "round16": round16,
    }


def _write_five_minute(tmp_path: Path, **overrides: object):
    sources = _sources(tmp_path)
    values: dict[str, object] = {
        "output_path": tmp_path / "portable" / "activation.json",
        "market_variant": "fiveminute",
        "risk_level": "conservative",
        "risk_capital_quote": Decimal("250.00"),
        "requested_quantity": Decimal("5"),
        "promotion_path": sources["promotion"],
        "evidence_root": sources["evidence"],
        "lifecycle_qualification_path": sources["lifecycle"],
        "round16_contract_path": None,
        "pretest_envelope_sha256": "",
        "evaluation_envelope_sha256": "",
        "created_at_ms": 1_800_000_000_000,
    }
    values.update(overrides)
    return write_polymarket_live_activation(**values), sources


def test_portable_activation_round_trip_contains_no_credentials(tmp_path: Path) -> None:
    verified, sources = _write_five_minute(tmp_path)

    assert verified.activation.market_variant == "fiveminute"
    assert verified.activation.risk_level == "conservative"
    assert verified.activation.risk_capital_quote == Decimal("250.00")
    assert verified.activation.requested_quantity == Decimal("5")
    assert verified.promotion_path == sources["promotion"].resolve()
    assert verified.evidence_root == sources["evidence"].resolve()
    assert verified.lifecycle_qualification_path == sources["lifecycle"].resolve()
    assert verified.round16_contract_path is None
    payload = json.loads(verified.path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == POLYMARKET_LIVE_ACTIVATION_SCHEMA_VERSION
    assert not Path(payload["promotion"]["path"]).is_absolute()
    assert not Path(payload["lifecycle_qualification"]["path"]).is_absolute()
    serialized = verified.path.read_text(encoding="utf-8").lower()
    for token in ("private_key", "api_key", "api_secret", "passphrase"):
        assert token not in serialized


def test_activation_rejects_file_drift_and_bundle_tampering(tmp_path: Path) -> None:
    verified, sources = _write_five_minute(tmp_path)
    sources["promotion"].write_text('{"promotion":false}\n', encoding="utf-8")
    with pytest.raises(PolymarketLiveBlocked, match="promotion file drifted"):
        load_polymarket_live_activation(verified.path)

    verified, _ = _write_five_minute(
        tmp_path / "second",
    )
    payload = json.loads(verified.path.read_text(encoding="utf-8"))
    payload["risk_level"] = "aggressive"
    verified.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash differs"):
        load_polymarket_live_activation(verified.path)


def test_activation_rejects_symlinked_authority_sources(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    link = tmp_path / "promotion-link.json"
    try:
        link.symlink_to(sources["promotion"])
    except OSError as exc:  # pragma: no cover - host policy controls symlink creation
        pytest.skip(f"host cannot create test symlink: {exc}")

    with pytest.raises(ValueError, match="promotion cannot be a symlink"):
        write_polymarket_live_activation(
            tmp_path / "activation.json",
            market_variant="fiveminute",
            risk_level="conservative",
            risk_capital_quote=Decimal("250"),
            requested_quantity=Decimal("5"),
            promotion_path=link,
            evidence_root=sources["evidence"],
            lifecycle_qualification_path=sources["lifecycle"],
            round16_contract_path=None,
            pretest_envelope_sha256="",
            evaluation_envelope_sha256="",
        )


def test_activation_rejects_duplicate_and_unknown_keys(tmp_path: Path) -> None:
    verified, _ = _write_five_minute(tmp_path)
    original = verified.path.read_text(encoding="utf-8").strip()
    duplicate = original[:-1] + ',"risk_level":"regular"}'
    verified.path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate keys"):
        load_polymarket_live_activation(verified.path)

    verified, _ = _write_five_minute(tmp_path / "unknown")
    payload = json.loads(verified.path.read_text(encoding="utf-8"))
    payload["credentials"] = "forbidden"
    verified.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema is invalid"):
        load_polymarket_live_activation(verified.path)


def test_fifteen_minute_activation_requires_exact_horizon_pins(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    output = tmp_path / "activation.json"
    verified = write_polymarket_live_activation(
        output,
        market_variant="fifteenminute",
        risk_level="regular",
        risk_capital_quote=Decimal("500"),
        requested_quantity=Decimal("7.5"),
        promotion_path=sources["promotion"],
        evidence_root=sources["evidence"],
        lifecycle_qualification_path=sources["lifecycle"],
        round16_contract_path=sources["round16"],
        pretest_envelope_sha256="a" * 64,
        evaluation_envelope_sha256="b" * 64,
        created_at_ms=1_800_000_000_000,
    )

    assert verified.activation.market_variant == "fifteenminute"
    assert verified.round16_contract_path == sources["round16"].resolve()
    assert verified.activation.pretest_envelope_sha256 == "a" * 64
    assert verified.activation.evaluation_envelope_sha256 == "b" * 64

    with pytest.raises(ValueError, match="missing required pins"):
        write_polymarket_live_activation(
            tmp_path / "invalid.json",
            market_variant="fifteenminute",
            risk_level="regular",
            risk_capital_quote=Decimal("500"),
            requested_quantity=Decimal("7.5"),
            promotion_path=sources["promotion"],
            evidence_root=sources["evidence"],
            lifecycle_qualification_path=sources["lifecycle"],
            round16_contract_path=None,
            pretest_envelope_sha256="",
            evaluation_envelope_sha256="",
            created_at_ms=1_800_000_000_000,
        )


def test_activation_overwrite_and_capability_are_fail_closed(tmp_path: Path) -> None:
    verified, sources = _write_five_minute(tmp_path)
    with pytest.raises(FileExistsError, match="already exists"):
        write_polymarket_live_activation(
            verified.path,
            market_variant="fiveminute",
            risk_level="conservative",
            risk_capital_quote=Decimal("250"),
            requested_quantity=Decimal("5"),
            promotion_path=sources["promotion"],
            evidence_root=sources["evidence"],
            lifecycle_qualification_path=sources["lifecycle"],
            round16_contract_path=None,
            pretest_envelope_sha256="",
            evaluation_envelope_sha256="",
        )

    with pytest.raises(TypeError, match="must come from its loader"):
        VerifiedPolymarketLiveActivation(
            activation=verified.activation,
            path=verified.path,
            file_sha256=verified.file_sha256,
            promotion_path=verified.promotion_path,
            evidence_root=verified.evidence_root,
            lifecycle_qualification_path=verified.lifecycle_qualification_path,
            round16_contract_path=None,
            _capability=object(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("risk_level", "reckless"),
        ("risk_capital_quote", Decimal("0")),
        ("requested_quantity", Decimal("NaN")),
        ("requested_quantity", Decimal("1000001")),
    ],
)
def test_activation_rejects_invalid_risk_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        _write_five_minute(tmp_path, **{field: value})


def test_activation_dataclass_does_not_accept_unhashed_body() -> None:
    with pytest.raises(ValueError, match="file references are invalid"):
        PolymarketLiveActivation(  # type: ignore[arg-type]
            activation_sha256="0" * 64,
            created_at_ms=1,
            market_variant="fiveminute",
            risk_level="conservative",
            risk_capital_quote=Decimal("1"),
            requested_quantity=Decimal("1"),
            promotion=None,
            evidence_root=".",
            lifecycle_qualification=None,
            round16_contract=None,
            pretest_envelope_sha256="",
            evaluation_envelope_sha256="",
        )
