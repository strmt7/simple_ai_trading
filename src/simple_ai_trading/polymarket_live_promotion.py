"""Hash-bound promotion firewall for autonomous Polymarket opening orders."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
from typing import Mapping

from .polymarket_live import PolymarketLiveBlocked


POLYMARKET_LIVE_PROMOTION_SCHEMA_VERSION = "polymarket-live-promotion-v1"
_MAX_PROMOTION_BYTES = 128 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_VERIFIED_PROMOTION_CAPABILITY = object()
_REQUIRED_GATES = frozenset(
    {
        "prospective_untouched_test",
        "source_integrity",
        "causal_feature_replay",
        "proper_scoring_uplift",
        "after_cost_edge",
        "uncertainty_lower_bound",
        "drawdown_limit",
        "latency_stress",
        "displayed_depth_stress",
        "authenticated_order_lifecycle",
        "settlement_and_redemption",
    }
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Polymarket promotion JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Polymarket promotion JSON contains {value}")


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


def _sha(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return normalized


def _decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    return parsed


def _safe_relative_path(value: object, *, name: str) -> str:
    text = str(value or "").strip()
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ValueError(f"{name} must be a safe repository-relative path")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class PolymarketPromotionEvidence:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            _safe_relative_path(self.path, name="promotion evidence path"),
        )
        object.__setattr__(
            self,
            "sha256",
            _sha(self.sha256, name="promotion evidence hash"),
        )


@dataclass(frozen=True, slots=True)
class PolymarketLivePromotion:
    promotion_id: str
    promotion_sha256: str
    created_at_ms: int
    expires_at_ms: int
    source_commit: str
    bot_id: str
    model_artifact: PolymarketPromotionEvidence
    evaluation_report: PolymarketPromotionEvidence
    implementation_manifest: PolymarketPromotionEvidence
    gates: Mapping[str, bool]
    minimum_expected_edge_quote_per_share: Decimal
    maximum_prediction_age_ms: int
    minimum_remaining_seconds: int
    paper_authority: bool
    live_authority: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "promotion_id",
            _sha(self.promotion_id, name="promotion_id"),
        )
        object.__setattr__(
            self,
            "promotion_sha256",
            _sha(self.promotion_sha256, name="promotion_sha256"),
        )
        created = int(self.created_at_ms)
        expires = int(self.expires_at_ms)
        if created <= 0 or expires <= created or expires - created > 31_536_000_000:
            raise ValueError("Polymarket promotion validity interval is invalid")
        object.__setattr__(self, "created_at_ms", created)
        object.__setattr__(self, "expires_at_ms", expires)
        commit = str(self.source_commit or "").strip().lower()
        if _GIT_COMMIT.fullmatch(commit) is None:
            raise ValueError("Polymarket promotion source commit is invalid")
        object.__setattr__(self, "source_commit", commit)
        bot_id = str(self.bot_id or "").strip()
        if not 8 <= len(bot_id) <= 128 or not re.fullmatch(r"[A-Za-z0-9._:-]+", bot_id):
            raise ValueError("Polymarket promotion bot_id is invalid")
        object.__setattr__(self, "bot_id", bot_id)
        gates = dict(self.gates)
        if set(gates) != _REQUIRED_GATES or any(
            type(value) is not bool for value in gates.values()
        ):
            raise ValueError("Polymarket promotion gate set is invalid")
        object.__setattr__(self, "gates", dict(sorted(gates.items())))
        edge = _decimal(
            self.minimum_expected_edge_quote_per_share,
            name="minimum expected edge",
        )
        if edge < Decimal("0.001") or edge > Decimal("0.25"):
            raise ValueError("Polymarket promotion edge threshold is invalid")
        object.__setattr__(
            self,
            "minimum_expected_edge_quote_per_share",
            edge,
        )
        prediction_age = int(self.maximum_prediction_age_ms)
        remaining = int(self.minimum_remaining_seconds)
        if not 100 <= prediction_age <= 5_000:
            raise ValueError("Polymarket promotion prediction age is invalid")
        if not 10 <= remaining <= 240:
            raise ValueError("Polymarket promotion remaining-time gate is invalid")
        object.__setattr__(self, "maximum_prediction_age_ms", prediction_age)
        object.__setattr__(self, "minimum_remaining_seconds", remaining)
        if type(self.paper_authority) is not bool or type(self.live_authority) is not bool:
            raise ValueError("Polymarket promotion authority flags are invalid")
        if self.live_authority and not self.paper_authority:
            raise ValueError("live authority requires paper authority")
        if self.live_authority and not all(gates.values()):
            raise ValueError("live authority requires every promotion gate")

    def assert_live_authority(self, *, observed_at_ms: int | None = None) -> None:
        now = int(time.time() * 1_000) if observed_at_ms is None else int(observed_at_ms)
        if now < self.created_at_ms:
            raise PolymarketLiveBlocked("Polymarket promotion is not yet valid")
        if now >= self.expires_at_ms:
            raise PolymarketLiveBlocked("Polymarket promotion expired")
        if not self.live_authority:
            raise PolymarketLiveBlocked("Polymarket promotion has no live authority")
        failed = tuple(name for name, passed in self.gates.items() if not passed)
        if failed:
            raise PolymarketLiveBlocked(
                f"Polymarket promotion gates failed: {failed}"
            )


@dataclass(frozen=True, slots=True)
class VerifiedPolymarketLivePromotion:
    """Evidence-verified promotion capability returned only by the loader."""

    promotion: PolymarketLivePromotion
    model_artifact_path: Path
    evaluation_report_path: Path
    implementation_manifest_path: Path
    _capability: object

    def __post_init__(self) -> None:
        if self._capability is not _VERIFIED_PROMOTION_CAPABILITY:
            raise ValueError("Polymarket promotion evidence was not verified")

    def assert_live_authority(self, *, observed_at_ms: int | None = None) -> None:
        self.promotion.assert_live_authority(observed_at_ms=observed_at_ms)


def _evidence(value: object, *, name: str) -> PolymarketPromotionEvidence:
    payload = _mapping(value, name=name)
    if set(payload) != {"path", "sha256"}:
        raise ValueError(f"{name} schema is invalid")
    return PolymarketPromotionEvidence(
        path=str(payload["path"]),
        sha256=str(payload["sha256"]),
    )


def validate_polymarket_live_promotion(
    value: Mapping[str, object],
) -> PolymarketLivePromotion:
    payload = dict(value)
    expected_keys = {
        "schema_version",
        "promotion_id",
        "promotion_sha256",
        "created_at_ms",
        "expires_at_ms",
        "source_commit",
        "venue",
        "protocol_version",
        "asset",
        "market_variant",
        "environment",
        "bot_id",
        "model_artifact",
        "evaluation_report",
        "implementation_manifest",
        "gates",
        "policy",
        "authority",
    }
    if set(payload) != expected_keys:
        raise ValueError("Polymarket promotion schema is invalid")
    if (
        payload["schema_version"] != POLYMARKET_LIVE_PROMOTION_SCHEMA_VERSION
        or payload["venue"] != "polymarket"
        or payload["protocol_version"] != 2
        or payload["asset"] != "BTC"
        or payload["market_variant"] != "fiveminute"
        or payload["environment"] != "live"
    ):
        raise ValueError("Polymarket promotion scope is invalid")
    authority = _mapping(payload["authority"], name="promotion authority")
    if set(authority) != {"paper", "live"}:
        raise ValueError("Polymarket promotion authority schema is invalid")
    policy = _mapping(payload["policy"], name="promotion policy")
    if set(policy) != {
        "minimum_expected_edge_quote_per_share",
        "maximum_prediction_age_ms",
        "minimum_remaining_seconds",
    }:
        raise ValueError("Polymarket promotion policy schema is invalid")
    gates = _mapping(payload["gates"], name="promotion gates")
    claimed = _sha(payload["promotion_sha256"], name="promotion_sha256")
    body = dict(payload)
    body.pop("promotion_sha256")
    if _canonical_sha256(body) != claimed:
        raise ValueError("Polymarket promotion hash differs")
    return PolymarketLivePromotion(
        promotion_id=str(payload["promotion_id"]),
        promotion_sha256=claimed,
        created_at_ms=int(payload["created_at_ms"]),
        expires_at_ms=int(payload["expires_at_ms"]),
        source_commit=str(payload["source_commit"]),
        bot_id=str(payload["bot_id"]),
        model_artifact=_evidence(payload["model_artifact"], name="model artifact"),
        evaluation_report=_evidence(
            payload["evaluation_report"],
            name="evaluation report",
        ),
        implementation_manifest=_evidence(
            payload["implementation_manifest"],
            name="implementation manifest",
        ),
        gates={str(key): value for key, value in gates.items()},
        minimum_expected_edge_quote_per_share=_decimal(
            policy["minimum_expected_edge_quote_per_share"],
            name="minimum expected edge",
        ),
        maximum_prediction_age_ms=int(policy["maximum_prediction_age_ms"]),
        minimum_remaining_seconds=int(policy["minimum_remaining_seconds"]),
        paper_authority=authority["paper"],
        live_authority=authority["live"],
    )


def _resolve_evidence(root: Path, evidence: PolymarketPromotionEvidence) -> Path:
    root_resolved = root.resolve(strict=True)
    path = root_resolved.joinpath(*PurePosixPath(evidence.path).parts)
    if path.is_symlink():
        raise ValueError("Polymarket promotion evidence cannot be a symlink")
    resolved = path.resolve(strict=True)
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError("Polymarket promotion evidence escaped its root")
    if not resolved.is_file():
        raise ValueError("Polymarket promotion evidence is not a file")
    if _sha256_file(resolved) != evidence.sha256:
        raise ValueError(
            f"Polymarket promotion evidence hash differs: {evidence.path}"
        )
    return resolved


def load_polymarket_live_promotion(
    path: str | Path,
    *,
    evidence_root: str | Path,
    require_live_authority: bool = True,
    observed_at_ms: int | None = None,
) -> VerifiedPolymarketLivePromotion:
    promotion_path = Path(path)
    if promotion_path.is_symlink():
        raise ValueError("Polymarket promotion cannot be a symlink")
    raw = promotion_path.read_bytes()
    if not raw or len(raw) > _MAX_PROMOTION_BYTES:
        raise ValueError("Polymarket promotion file size is invalid")
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Polymarket promotion is not strict JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Polymarket promotion is not an object")
    promotion = validate_polymarket_live_promotion(payload)
    root = Path(evidence_root)
    model_artifact_path = _resolve_evidence(root, promotion.model_artifact)
    evaluation_report_path = _resolve_evidence(root, promotion.evaluation_report)
    implementation_manifest_path = _resolve_evidence(
        root,
        promotion.implementation_manifest,
    )
    if require_live_authority:
        promotion.assert_live_authority(observed_at_ms=observed_at_ms)
    return VerifiedPolymarketLivePromotion(
        promotion=promotion,
        model_artifact_path=model_artifact_path,
        evaluation_report_path=evaluation_report_path,
        implementation_manifest_path=implementation_manifest_path,
        _capability=_VERIFIED_PROMOTION_CAPABILITY,
    )


__all__ = [
    "POLYMARKET_LIVE_PROMOTION_SCHEMA_VERSION",
    "PolymarketLivePromotion",
    "PolymarketPromotionEvidence",
    "VerifiedPolymarketLivePromotion",
    "load_polymarket_live_promotion",
    "validate_polymarket_live_promotion",
]
