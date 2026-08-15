"""Portable, non-secret activation contract for autonomous Polymarket live use."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import time
from typing import Mapping

from .polymarket_live import PolymarketLiveBlocked


POLYMARKET_LIVE_ACTIVATION_SCHEMA_VERSION = "polymarket-live-activation-v1"
DEFAULT_POLYMARKET_LIVE_ACTIVATION = Path("data/polymarket/live-activation.json")
_MAXIMUM_ACTIVATION_BYTES = 64 * 1024
_MAXIMUM_REFERENCE_LENGTH = 1024
_RISK_LEVELS = frozenset({"conservative", "regular", "aggressive"})
_MARKET_VARIANTS = frozenset({"fiveminute", "fifteenminute"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERIFIED_CAPABILITY = object()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Polymarket activation JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Polymarket activation JSON contains {value}")


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} schema is invalid")


def _sha(value: object, *, name: str, allow_empty: bool = False) -> str:
    normalized = str(value or "").strip().lower()
    if allow_empty and not normalized:
        return ""
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return normalized


def _positive_decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite decimal") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{name} must be a positive finite decimal")
    return parsed


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _relative_reference(value: object, *, name: str) -> str:
    text = str(value or "").strip()
    path = PurePosixPath(text)
    if (
        not text
        or len(text) > _MAXIMUM_REFERENCE_LENGTH
        or "\\" in text
        or "\x00" in text
        or path.is_absolute()
        or any(":" in part for part in path.parts)
    ):
        raise ValueError(f"{name} must be a portable relative path")
    return path.as_posix()


def _resolve_reference(base: Path, reference: str, *, name: str) -> Path:
    relative = PurePosixPath(_relative_reference(reference, name=name))
    candidate = base.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise ValueError(f"{name} cannot be a symlink")
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} does not resolve to an existing path") from exc


def _portable_reference(path: Path, *, base: Path, name: str) -> str:
    resolved = path.resolve(strict=True)
    relative = os.path.relpath(resolved, base.resolve()).replace(os.sep, "/")
    return _relative_reference(relative, name=name)


def _source_path(value: str | Path, *, name: str) -> Path:
    selected = Path(value).expanduser()
    if selected.is_symlink():
        raise ValueError(f"{name} cannot be a symlink")
    return selected.resolve(strict=True)


@dataclass(frozen=True, slots=True)
class PolymarketActivationFile:
    path: str
    file_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            _relative_reference(self.path, name="activation file path"),
        )
        object.__setattr__(
            self,
            "file_sha256",
            _sha(self.file_sha256, name="activation file hash"),
        )


@dataclass(frozen=True, slots=True)
class PolymarketLiveActivation:
    activation_sha256: str
    created_at_ms: int
    market_variant: str
    risk_level: str
    risk_capital_quote: Decimal
    requested_quantity: Decimal
    promotion: PolymarketActivationFile
    evidence_root: str
    lifecycle_qualification: PolymarketActivationFile
    round16_contract: PolymarketActivationFile | None
    pretest_envelope_sha256: str
    evaluation_envelope_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "activation_sha256",
            _sha(self.activation_sha256, name="activation hash"),
        )
        created = int(self.created_at_ms)
        if created <= 0:
            raise ValueError("activation creation time is invalid")
        object.__setattr__(self, "created_at_ms", created)
        variant = str(self.market_variant or "").strip().lower()
        if variant not in _MARKET_VARIANTS:
            raise ValueError("activation market variant is invalid")
        object.__setattr__(self, "market_variant", variant)
        risk_level = str(self.risk_level or "").strip().lower()
        if risk_level not in _RISK_LEVELS:
            raise ValueError("activation risk level is invalid")
        object.__setattr__(self, "risk_level", risk_level)
        object.__setattr__(
            self,
            "risk_capital_quote",
            _positive_decimal(self.risk_capital_quote, name="activation risk capital"),
        )
        quantity = _positive_decimal(
            self.requested_quantity,
            name="activation requested quantity",
        )
        if quantity > Decimal("1000000"):
            raise ValueError("activation requested quantity is unreasonably large")
        object.__setattr__(self, "requested_quantity", quantity)
        if not isinstance(self.promotion, PolymarketActivationFile) or not isinstance(
            self.lifecycle_qualification,
            PolymarketActivationFile,
        ):
            raise ValueError("activation file references are invalid")
        if self.round16_contract is not None and not isinstance(
            self.round16_contract,
            PolymarketActivationFile,
        ):
            raise ValueError("activation Round 16 reference is invalid")
        object.__setattr__(
            self,
            "evidence_root",
            _relative_reference(self.evidence_root, name="activation evidence root"),
        )
        pretest_raw = str(self.pretest_envelope_sha256 or "").strip()
        evaluation_raw = str(self.evaluation_envelope_sha256 or "").strip()
        if variant == "fifteenminute" and (
            self.round16_contract is None or not pretest_raw or not evaluation_raw
        ):
            raise ValueError("fifteen-minute activation is missing required pins")
        pretest = _sha(
            pretest_raw,
            name="activation pretest envelope hash",
            allow_empty=variant == "fiveminute",
        )
        evaluation = _sha(
            evaluation_raw,
            name="activation evaluation envelope hash",
            allow_empty=variant == "fiveminute",
        )
        if variant == "fiveminute":
            if self.round16_contract is not None or pretest or evaluation:
                raise ValueError("five-minute activation contains fifteen-minute pins")
        object.__setattr__(self, "pretest_envelope_sha256", pretest)
        object.__setattr__(self, "evaluation_envelope_sha256", evaluation)

    def body(self) -> dict[str, object]:
        return {
            "created_at_ms": self.created_at_ms,
            "evaluation_envelope_sha256": self.evaluation_envelope_sha256,
            "evidence_root": self.evidence_root,
            "lifecycle_qualification": {
                "file_sha256": self.lifecycle_qualification.file_sha256,
                "path": self.lifecycle_qualification.path,
            },
            "market_variant": self.market_variant,
            "pretest_envelope_sha256": self.pretest_envelope_sha256,
            "promotion": {
                "file_sha256": self.promotion.file_sha256,
                "path": self.promotion.path,
            },
            "requested_quantity": _decimal_text(self.requested_quantity),
            "risk_capital_quote": _decimal_text(self.risk_capital_quote),
            "risk_level": self.risk_level,
            "round16_contract": (
                None
                if self.round16_contract is None
                else {
                    "file_sha256": self.round16_contract.file_sha256,
                    "path": self.round16_contract.path,
                }
            ),
            "schema_version": POLYMARKET_LIVE_ACTIVATION_SCHEMA_VERSION,
        }


@dataclass(frozen=True, slots=True)
class VerifiedPolymarketLiveActivation:
    activation: PolymarketLiveActivation
    path: Path
    file_sha256: str
    promotion_path: Path
    evidence_root: Path
    lifecycle_qualification_path: Path
    round16_contract_path: Path | None
    _capability: object

    def __post_init__(self) -> None:
        if self._capability is not _VERIFIED_CAPABILITY:
            raise TypeError("verified Polymarket activation must come from its loader")


def _file_from_payload(value: object, *, name: str) -> PolymarketActivationFile:
    payload = _mapping(value, name=name)
    _exact_keys(payload, {"file_sha256", "path"}, name=name)
    return PolymarketActivationFile(
        path=str(payload["path"]),
        file_sha256=str(payload["file_sha256"]),
    )


def _activation_from_payload(payload: Mapping[str, object]) -> PolymarketLiveActivation:
    expected = {
        "activation_sha256",
        "created_at_ms",
        "evaluation_envelope_sha256",
        "evidence_root",
        "lifecycle_qualification",
        "market_variant",
        "pretest_envelope_sha256",
        "promotion",
        "requested_quantity",
        "risk_capital_quote",
        "risk_level",
        "round16_contract",
        "schema_version",
    }
    _exact_keys(payload, expected, name="Polymarket activation")
    if payload["schema_version"] != POLYMARKET_LIVE_ACTIVATION_SCHEMA_VERSION:
        raise ValueError("Polymarket activation schema version differs")
    round16_raw = payload["round16_contract"]
    activation = PolymarketLiveActivation(
        activation_sha256=str(payload["activation_sha256"]),
        created_at_ms=int(payload["created_at_ms"]),
        market_variant=str(payload["market_variant"]),
        risk_level=str(payload["risk_level"]),
        risk_capital_quote=_positive_decimal(
            payload["risk_capital_quote"],
            name="activation risk capital",
        ),
        requested_quantity=_positive_decimal(
            payload["requested_quantity"],
            name="activation requested quantity",
        ),
        promotion=_file_from_payload(payload["promotion"], name="activation promotion"),
        evidence_root=str(payload["evidence_root"]),
        lifecycle_qualification=_file_from_payload(
            payload["lifecycle_qualification"],
            name="activation lifecycle qualification",
        ),
        round16_contract=(
            None
            if round16_raw is None
            else _file_from_payload(round16_raw, name="activation Round 16 contract")
        ),
        pretest_envelope_sha256=str(payload["pretest_envelope_sha256"]),
        evaluation_envelope_sha256=str(payload["evaluation_envelope_sha256"]),
    )
    if _canonical_sha256(activation.body()) != activation.activation_sha256:
        raise ValueError("Polymarket activation self-hash differs")
    return activation


def load_polymarket_live_activation(
    path: str | Path,
) -> VerifiedPolymarketLiveActivation:
    selected_path = Path(path).expanduser()
    if selected_path.is_symlink():
        raise ValueError("Polymarket activation cannot be a symlink")
    activation_path = selected_path.resolve(strict=True)
    if not activation_path.is_file():
        raise ValueError("Polymarket activation path is not a file")
    with activation_path.open("rb") as handle:
        encoded = handle.read(_MAXIMUM_ACTIVATION_BYTES + 1)
    if not encoded or len(encoded) > _MAXIMUM_ACTIVATION_BYTES:
        raise ValueError("Polymarket activation file size is invalid")
    try:
        raw = encoded.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("Polymarket activation JSON is invalid") from exc
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("Polymarket activation JSON is invalid") from exc
    activation = _activation_from_payload(
        _mapping(payload, name="Polymarket activation")
    )
    base = activation_path.parent
    promotion_path = _resolve_reference(
        base,
        activation.promotion.path,
        name="activation promotion",
    )
    lifecycle_path = _resolve_reference(
        base,
        activation.lifecycle_qualification.path,
        name="activation lifecycle qualification",
    )
    evidence_root = _resolve_reference(
        base,
        activation.evidence_root,
        name="activation evidence root",
    )
    if not promotion_path.is_file() or not lifecycle_path.is_file():
        raise ValueError("Polymarket activation file reference is not a file")
    if not evidence_root.is_dir():
        raise ValueError("Polymarket activation evidence root is not a directory")
    if _sha256_file(promotion_path) != activation.promotion.file_sha256:
        raise PolymarketLiveBlocked("Polymarket activation promotion file drifted")
    if _sha256_file(lifecycle_path) != activation.lifecycle_qualification.file_sha256:
        raise PolymarketLiveBlocked(
            "Polymarket activation lifecycle qualification drifted"
        )
    round16_path: Path | None = None
    if activation.round16_contract is not None:
        round16_path = _resolve_reference(
            base,
            activation.round16_contract.path,
            name="activation Round 16 contract",
        )
        if not round16_path.is_file():
            raise ValueError("Polymarket activation Round 16 reference is not a file")
        if _sha256_file(round16_path) != activation.round16_contract.file_sha256:
            raise PolymarketLiveBlocked("Polymarket activation Round 16 file drifted")
    return VerifiedPolymarketLiveActivation(
        activation=activation,
        path=activation_path,
        file_sha256=hashlib.sha256(encoded).hexdigest(),
        promotion_path=promotion_path,
        evidence_root=evidence_root,
        lifecycle_qualification_path=lifecycle_path,
        round16_contract_path=round16_path,
        _capability=_VERIFIED_CAPABILITY,
    )


def write_polymarket_live_activation(
    output_path: str | Path,
    *,
    market_variant: str,
    risk_level: str,
    risk_capital_quote: Decimal,
    requested_quantity: Decimal,
    promotion_path: str | Path,
    evidence_root: str | Path,
    lifecycle_qualification_path: str | Path,
    round16_contract_path: str | Path | None,
    pretest_envelope_sha256: str,
    evaluation_envelope_sha256: str,
    created_at_ms: int | None = None,
    replace_existing: bool = False,
) -> VerifiedPolymarketLiveActivation:
    selected_destination = Path(output_path).expanduser()
    if selected_destination.is_symlink():
        raise ValueError("Polymarket activation output cannot be a symlink")
    destination = selected_destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace_existing:
        raise FileExistsError("Polymarket activation already exists")
    base = destination.parent
    promotion = _source_path(
        promotion_path,
        name="Polymarket activation promotion",
    )
    lifecycle = _source_path(
        lifecycle_qualification_path,
        name="Polymarket activation lifecycle qualification",
    )
    evidence = _source_path(
        evidence_root,
        name="Polymarket activation evidence root",
    )
    if not promotion.is_file() or not lifecycle.is_file() or not evidence.is_dir():
        raise ValueError("Polymarket activation source paths are invalid")
    round16 = (
        None
        if round16_contract_path is None
        else _source_path(
            round16_contract_path,
            name="Polymarket activation Round 16 contract",
        )
    )
    if round16 is not None and not round16.is_file():
        raise ValueError("Polymarket activation Round 16 source is not a file")
    body: dict[str, object] = {
        "created_at_ms": (
            int(time.time() * 1_000) if created_at_ms is None else int(created_at_ms)
        ),
        "evaluation_envelope_sha256": str(evaluation_envelope_sha256 or ""),
        "evidence_root": _portable_reference(
            evidence,
            base=base,
            name="activation evidence root",
        ),
        "lifecycle_qualification": {
            "file_sha256": _sha256_file(lifecycle),
            "path": _portable_reference(
                lifecycle,
                base=base,
                name="activation lifecycle qualification",
            ),
        },
        "market_variant": str(market_variant),
        "pretest_envelope_sha256": str(pretest_envelope_sha256 or ""),
        "promotion": {
            "file_sha256": _sha256_file(promotion),
            "path": _portable_reference(
                promotion,
                base=base,
                name="activation promotion",
            ),
        },
        "requested_quantity": _decimal_text(
            _positive_decimal(requested_quantity, name="activation requested quantity")
        ),
        "risk_capital_quote": _decimal_text(
            _positive_decimal(risk_capital_quote, name="activation risk capital")
        ),
        "risk_level": str(risk_level),
        "round16_contract": (
            None
            if round16 is None
            else {
                "file_sha256": _sha256_file(round16),
                "path": _portable_reference(
                    round16,
                    base=base,
                    name="activation Round 16 contract",
                ),
            }
        ),
        "schema_version": POLYMARKET_LIVE_ACTIVATION_SCHEMA_VERSION,
    }
    provisional = {"activation_sha256": _canonical_sha256(body), **body}
    _activation_from_payload(provisional)
    encoded = (_canonical_json(provisional) + "\n").encode("ascii")
    temporary = destination.with_name(f".{destination.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() and not replace_existing:
            raise FileExistsError("Polymarket activation already exists")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return load_polymarket_live_activation(destination)


__all__ = [
    "DEFAULT_POLYMARKET_LIVE_ACTIVATION",
    "POLYMARKET_LIVE_ACTIVATION_SCHEMA_VERSION",
    "PolymarketActivationFile",
    "PolymarketLiveActivation",
    "VerifiedPolymarketLiveActivation",
    "load_polymarket_live_activation",
    "write_polymarket_live_activation",
]
