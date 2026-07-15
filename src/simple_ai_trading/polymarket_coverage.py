"""Fail-closed feature and label coverage for Polymarket evidence runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import json
from typing import Mapping

from .assets import SUPPORTED_MAJOR_BASE_ASSETS
from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_replay import PolymarketEvidenceReplay
from .polymarket_resolution import load_official_resolutions


POLYMARKET_COVERAGE_SCHEMA_VERSION = "polymarket-feed-coverage-v2"
DEFAULT_MINIMUM_RESOLVED_MARKETS_PER_ASSET = 30
_ASSETS = tuple(SUPPORTED_MAJOR_BASE_ASSETS)
_COUNT_KEYS = (
    "market_snapshots",
    "clob_token_baselines",
    "direct_binance_book_tickers",
    "direct_binance_trades",
    "rtds_binance_samples",
    "rtds_chainlink_samples",
    "official_resolutions",
)


@dataclass(frozen=True)
class PolymarketFeedCoverage:
    schema_version: str
    run_id: str
    run_status: str
    allow_segmented_gaps: bool
    stream_gap_count: int
    minimum_resolved_markets_per_asset: int
    counts: dict[str, dict[str, int]]
    integrity_errors: tuple[str, ...]
    data_errors: tuple[str, ...]
    shadow_errors: tuple[str, ...]
    training_errors: tuple[str, ...]

    @property
    def shadow_ready(self) -> bool:
        return not self.shadow_errors

    @property
    def training_ready(self) -> bool:
        return not self.training_errors

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["shadow_ready"] = self.shadow_ready
        payload["training_ready"] = self.training_ready
        return payload


def _positive_decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive decimal") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{name} must be a finite positive decimal")
    return parsed


def _timestamp(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _asset(value: object) -> str:
    raw = str(value or "").strip().upper()
    if "/" in raw:
        raw = raw.split("/", 1)[0]
    for suffix in ("USDT", "USD"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            break
    return raw if raw in _ASSETS else ""


def _rtds_sample_count(event: Mapping[str, object]) -> tuple[str, str, int]:
    topic = str(event.get("topic") or "").strip()
    message_type = str(event.get("type") or "").strip().lower()
    payload = event.get("payload")
    if topic not in {"crypto_prices", "crypto_prices_chainlink"}:
        return "", "", 0
    if not isinstance(payload, Mapping):
        raise ValueError("RTDS crypto payload is not an object")
    asset = _asset(payload.get("symbol"))
    if not asset:
        raise ValueError("RTDS crypto symbol is unsupported")
    source = "rtds_binance_samples" if topic == "crypto_prices" else "rtds_chainlink_samples"
    if message_type == "subscribe":
        rows = payload.get("data")
        if not isinstance(rows, list) or not rows:
            raise ValueError("RTDS subscribe history is empty or malformed")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("RTDS subscribe history row is malformed")
            _timestamp(row.get("timestamp"), name="RTDS history timestamp")
            _positive_decimal(row.get("value"), name="RTDS history value")
        return asset, source, len(rows)
    if message_type == "update":
        _timestamp(payload.get("timestamp"), name="RTDS update timestamp")
        _positive_decimal(payload.get("value"), name="RTDS update value")
        return asset, source, 1
    return asset, source, 0


def inspect_polymarket_feed_coverage(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    minimum_resolved_markets_per_asset: int = (
        DEFAULT_MINIMUM_RESOLVED_MARKETS_PER_ASSET
    ),
    allow_segmented_gaps: bool = False,
) -> PolymarketFeedCoverage:
    """Audit whether one immutable run can support shadowing or model fitting."""

    selected = str(run_id or "").strip()
    if not selected:
        raise ValueError("run_id is required for Polymarket feed coverage")
    minimum_resolved = int(minimum_resolved_markets_per_asset)
    if minimum_resolved < 1 or minimum_resolved > 100_000:
        raise ValueError("minimum resolved markets per asset must lie in [1, 100000]")
    connection = store.connect()
    run = connection.execute(
        "SELECT status FROM polymarket_recorder_run WHERE run_id = ?",
        [selected],
    ).fetchone()
    if run is None:
        raise ValueError(f"unknown Polymarket recorder run: {selected}")
    run_status = str(run[0])
    integrity_errors = store.integrity_errors(selected)
    try:
        stream_gap_count = PolymarketEvidenceReplay.validate_stream_gaps(
            store,
            selected,
            allow_segmented_gaps=bool(allow_segmented_gaps),
        )
        gap_validation_error = ""
    except ValueError as exc:
        stream_gap_count = int(
            connection.execute(
                "SELECT count(*) FROM polymarket_stream_gap WHERE run_id = ?",
                [selected],
            ).fetchone()[0]
        )
        gap_validation_error = str(exc)
    counts = {asset: {key: 0 for key in _COUNT_KEYS} for asset in _ASSETS}

    market_rows = connection.execute(
        """
        SELECT asset, condition_id, up_token_id, down_token_id
        FROM polymarket_market_snapshot
        WHERE run_id = ? ORDER BY event_start_ms, asset
        """,
        [selected],
    ).fetchall()
    condition_asset: dict[str, str] = {}
    token_asset: dict[str, str] = {}
    expected_tokens: dict[str, set[str]] = {asset: set() for asset in _ASSETS}
    for raw_asset, condition_id, up_token, down_token in market_rows:
        asset = _asset(raw_asset)
        if not asset:
            continue
        condition = str(condition_id).lower()
        tokens = (str(up_token), str(down_token))
        condition_asset[condition] = asset
        counts[asset]["market_snapshots"] += 1
        for token in tokens:
            token_asset[token] = asset
            expected_tokens[asset].add(token)

    observed_tokens: dict[str, set[str]] = {asset: set() for asset in _ASSETS}
    resolved_conditions: dict[str, set[str]] = {asset: set() for asset in _ASSETS}
    data_errors: list[str] = []
    if gap_validation_error:
        data_errors.append(f"stream_gap_invalid:{gap_validation_error}")
    event_rows = connection.execute(
        """
        SELECT event_id, stream, event_type, symbol, condition_id,
               asset_id, event_json
        FROM polymarket_public_event
        WHERE run_id = ? ORDER BY event_id
        """,
        [selected],
    ).fetchall()
    for event_id, stream, event_type, symbol, condition_id, asset_id, event_json in event_rows:
        try:
            event = json.loads(str(event_json))
            if not isinstance(event, Mapping):
                raise ValueError("normalized event is not an object")
            normalized_stream = str(stream)
            normalized_type = str(event_type).lower()
            if normalized_stream in {"clob_market", "clob_rest_book"}:
                if normalized_type == "book":
                    token = str(event.get("asset_id") or asset_id)
                    asset = token_asset.get(token, "")
                    if asset:
                        observed_tokens[asset].add(token)
                elif normalized_type == "market_resolved":
                    condition = str(event.get("market") or condition_id).lower()
                    asset = condition_asset.get(condition, "")
                    if asset:
                        resolved_conditions[asset].add(condition)
            elif normalized_stream == "binance_spot":
                asset = _asset(symbol)
                payload = event.get("data")
                if not asset or not isinstance(payload, Mapping):
                    raise ValueError("direct Binance event identity is malformed")
                if normalized_type == "bookticker":
                    bid = _positive_decimal(payload.get("b"), name="Binance bid")
                    ask = _positive_decimal(payload.get("a"), name="Binance ask")
                    _positive_decimal(payload.get("B"), name="Binance bid quantity")
                    _positive_decimal(payload.get("A"), name="Binance ask quantity")
                    if bid >= ask:
                        raise ValueError("direct Binance book ticker is crossed")
                    counts[asset]["direct_binance_book_tickers"] += 1
                elif normalized_type == "trade":
                    _timestamp(payload.get("T"), name="Binance trade timestamp")
                    _positive_decimal(payload.get("p"), name="Binance trade price")
                    _positive_decimal(payload.get("q"), name="Binance trade quantity")
                    counts[asset]["direct_binance_trades"] += 1
            elif normalized_stream == "polymarket_rtds":
                asset, key, sample_count = _rtds_sample_count(event)
                if asset and key:
                    counts[asset][key] += sample_count
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            data_errors.append(
                f"feed_event_invalid:{event_id}:{exc.__class__.__name__}:{exc}"
            )

    for asset in _ASSETS:
        counts[asset]["clob_token_baselines"] = len(observed_tokens[asset])
    try:
        for resolution in load_official_resolutions(store, run_id=selected):
            asset = condition_asset.get(resolution.condition_id, "")
            if not asset:
                raise ValueError("official resolution references an unknown condition")
            resolved_conditions[asset].add(resolution.condition_id)
    except ValueError as exc:
        data_errors.append(
            f"official_resolution_invalid:{exc.__class__.__name__}:{exc}"
        )
    for asset in _ASSETS:
        counts[asset]["official_resolutions"] = len(resolved_conditions[asset])

    shadow_errors: list[str] = []
    segmented_status_allowed = bool(allow_segmented_gaps) and run_status == "degraded"
    if run_status != "complete" and not segmented_status_allowed:
        shadow_errors.append(f"run_not_complete:{run_status}")
    if integrity_errors:
        shadow_errors.append("recorder_integrity_failed")
    if data_errors:
        shadow_errors.append("feed_event_validation_failed")
    for asset in _ASSETS:
        values = counts[asset]
        expected_baselines = max(2, len(expected_tokens[asset]))
        requirements = {
            "market_snapshots": 1,
            "clob_token_baselines": expected_baselines,
            "direct_binance_book_tickers": 1,
            "direct_binance_trades": 1,
            "rtds_binance_samples": 1,
            "rtds_chainlink_samples": 1,
        }
        for key, minimum in requirements.items():
            if minimum <= 0 or values[key] < minimum:
                shadow_errors.append(
                    f"insufficient_{key}:{asset}:{values[key]}/{minimum}"
                )

    training_errors = list(shadow_errors)
    for asset in _ASSETS:
        resolutions = counts[asset]["official_resolutions"]
        if resolutions < minimum_resolved:
            training_errors.append(
                f"insufficient_official_resolutions:{asset}:"
                f"{resolutions}/{minimum_resolved}"
            )
    return PolymarketFeedCoverage(
        schema_version=POLYMARKET_COVERAGE_SCHEMA_VERSION,
        run_id=selected,
        run_status=run_status,
        allow_segmented_gaps=bool(allow_segmented_gaps),
        stream_gap_count=stream_gap_count,
        minimum_resolved_markets_per_asset=minimum_resolved,
        counts=counts,
        integrity_errors=tuple(integrity_errors),
        data_errors=tuple(sorted(set(data_errors))),
        shadow_errors=tuple(sorted(set(shadow_errors))),
        training_errors=tuple(sorted(set(training_errors))),
    )


__all__ = [
    "DEFAULT_MINIMUM_RESOLVED_MARKETS_PER_ASSET",
    "POLYMARKET_COVERAGE_SCHEMA_VERSION",
    "PolymarketFeedCoverage",
    "inspect_polymarket_feed_coverage",
]
