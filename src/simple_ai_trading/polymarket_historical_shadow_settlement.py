"""Joint official settlement evidence for a Polymarket shadow opportunity."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


SHADOW_SETTLEMENT_SCHEMA_VERSION = (
    "polymarket-btc-shadow-settlement-evidence-v1"
)


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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} must be a JSON array") from exc
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _source_commit(value: str) -> str:
    commit = str(value or "").strip().lower()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("Polymarket shadow settlement source commit is invalid")
    return commit


def _iso8601_ms(value: object, *, name: str) -> int:
    normalized = str(value or "").strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{name} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return int(parsed.astimezone(timezone.utc).timestamp() * 1_000)


def _verify_opportunity(
    value: Mapping[str, object],
) -> tuple[Mapping[str, object], str]:
    opportunity = dict(value)
    claimed = str(opportunity.pop("artifact_sha256", "")).lower()
    if len(claimed) != 64 or _canonical_sha256(opportunity) != claimed:
        raise ValueError("Polymarket shadow opportunity integrity failed")
    status = opportunity.get("status")
    reason = opportunity.get("reason")
    if (
        opportunity.get("schema_version")
        != "polymarket-btc-shadow-opportunity-v1"
        or status not in {"candidate", "abstain"}
        or not isinstance(reason, str)
        or (status == "candidate" and reason != "")
        or (status == "abstain" and reason == "")
        or opportunity.get("trading_authority") is not False
        or opportunity.get("proposal_authority") is not False
        or opportunity.get("execution_or_profitability_claim") is not False
    ):
        raise ValueError("Polymarket shadow opportunity is not verifiable")
    return value, claimed


def _gamma_winner(
    gamma: Mapping[str, object],
    opportunity: Mapping[str, object],
) -> str:
    if (
        str(gamma.get("id") or "") != str(opportunity["gamma_market_id"])
        or str(gamma.get("conditionId") or "").lower()
        != str(opportunity["condition_id"]).lower()
        or str(gamma.get("slug") or "") != str(opportunity["slug"])
        or gamma.get("closed") is not True
        or gamma.get("acceptingOrders") is not False
        or _iso8601_ms(gamma.get("endDate"), name="Gamma endDate")
        != int(opportunity["event_end_ms"])
    ):
        raise ValueError("Gamma terminal market identity differs")
    outcomes = tuple(
        str(item)
        for item in _sequence(gamma.get("outcomes"), name="Gamma outcomes")
    )
    prices = tuple(
        _decimal(item, name="Gamma outcome price")
        for item in _sequence(
            gamma.get("outcomePrices"),
            name="Gamma outcomePrices",
        )
    )
    if outcomes != ("Up", "Down") or len(prices) != 2:
        raise ValueError("Gamma terminal outcomes differ")
    winners = [
        outcome
        for outcome, price in zip(outcomes, prices, strict=True)
        if price == Decimal("1")
    ]
    if len(winners) != 1 or any(
        price not in {Decimal("0"), Decimal("1")} for price in prices
    ):
        raise ValueError("Gamma market is not terminal")
    return winners[0]


def _clob_winner(
    clob: Mapping[str, object],
    opportunity: Mapping[str, object],
) -> str:
    if (
        str(clob.get("condition_id") or "").lower()
        != str(opportunity["condition_id"]).lower()
        or clob.get("closed") is not True
        or clob.get("accepting_orders") is not False
    ):
        raise ValueError("CLOB terminal market identity differs")
    tokens = _sequence(clob.get("tokens"), name="CLOB tokens")
    if len(tokens) != 2:
        raise ValueError("CLOB terminal token count differs")
    quote_tokens = {
        "Up": str(
            _mapping(opportunity["up_quote"], name="Up quote")["token_id"]
        ),
        "Down": str(
            _mapping(opportunity["down_quote"], name="Down quote")["token_id"]
        ),
    }
    observed: dict[str, tuple[str, Decimal, bool]] = {}
    for raw in tokens:
        token = _mapping(raw, name="CLOB token")
        outcome = str(token.get("outcome") or "")
        if outcome in observed:
            raise ValueError("CLOB terminal outcomes are duplicated")
        observed[outcome] = (
            str(token.get("token_id") or ""),
            _decimal(token.get("price"), name="CLOB token price"),
            token.get("winner") is True,
        )
    if set(observed) != {"Up", "Down"} or any(
        observed[outcome][0] != quote_tokens[outcome]
        for outcome in ("Up", "Down")
    ):
        raise ValueError("CLOB terminal token mapping differs")
    winners = [
        outcome
        for outcome, (_, price, winner) in observed.items()
        if winner and price == Decimal("1")
    ]
    if len(winners) != 1 or any(
        price not in {Decimal("0"), Decimal("1")}
        or winner != (price == Decimal("1"))
        for _, price, winner in observed.values()
    ):
        raise ValueError("CLOB market is not terminal")
    return winners[0]


def validate_shadow_official_resolution(
    opportunity: Mapping[str, object],
    *,
    gamma_market: Mapping[str, object],
    clob_market: Mapping[str, object],
    resolution_observed_at_ms: int,
) -> Mapping[str, object]:
    """Require matching terminal winner and identity from both public APIs."""

    verified, opportunity_sha = _verify_opportunity(opportunity)
    observed_at = int(resolution_observed_at_ms)
    if observed_at < int(verified["event_end_ms"]):
        raise ValueError("Polymarket resolution was observed before market end")
    gamma_winner = _gamma_winner(gamma_market, verified)
    clob_winner = _clob_winner(clob_market, verified)
    if gamma_winner != clob_winner:
        raise ValueError("Gamma and CLOB terminal outcomes disagree")
    return {
        "winner": gamma_winner,
        "opportunity_artifact_sha256": opportunity_sha,
        "gamma_payload_sha256": _canonical_sha256(gamma_market),
        "clob_payload_sha256": _canonical_sha256(clob_market),
        "resolution_observed_at_ms": observed_at,
        "sources_agree": True,
        "trading_authority": False,
        "profitability_claim": False,
    }


def settle_shadow_opportunity(
    opportunity: Mapping[str, object],
    *,
    gamma_market: Mapping[str, object],
    clob_market: Mapping[str, object],
    resolution_observed_at_ms: int,
    source_log_sha256: str,
    source_commit: str,
) -> tuple[Mapping[str, object], str]:
    """Create no-authority shadow evidence from matching terminal sources."""

    verified, opportunity_sha = _verify_opportunity(opportunity)
    if verified.get("status") != "candidate":
        raise ValueError("Polymarket shadow opportunity is not settleable")
    resolution = validate_shadow_official_resolution(
        verified,
        gamma_market=gamma_market,
        clob_market=clob_market,
        resolution_observed_at_ms=resolution_observed_at_ms,
    )
    observed_at = int(resolution["resolution_observed_at_ms"])
    winner = str(resolution["winner"])
    selected = str(verified["selected_outcome"])
    if selected not in {"Up", "Down"}:
        raise ValueError("Polymarket shadow selected outcome differs")
    selected_quote = _mapping(
        verified["up_quote"] if selected == "Up" else verified["down_quote"],
        name="selected quote",
    )
    if selected_quote.get("displayed_fillable") is not True:
        raise ValueError("Polymarket shadow selected quote was not fillable")
    quantity = _decimal(selected_quote.get("quantity"), name="quote quantity")
    filled = _decimal(
        selected_quote.get("filled_quantity"),
        name="quote filled quantity",
    )
    entry_cost = _decimal(
        selected_quote.get("total_cost_quote"),
        name="quote entry cost",
    )
    if quantity <= 0 or filled != quantity or entry_cost <= 0:
        raise ValueError("Polymarket shadow selected quote differs")
    maximum_loss = _decimal(
        verified.get("maximum_loss_quote"),
        name="maximum loss",
    )
    if maximum_loss != entry_cost:
        raise ValueError("Polymarket shadow maximum loss differs")
    probability_up = float(
        _decimal(verified.get("probability_up"), name="probability_up")
    )
    if not 0.0 < probability_up < 1.0:
        raise ValueError("Polymarket shadow probability differs")
    target = 1.0 if winner == "Up" else 0.0
    predicted_winner_probability = (
        probability_up if winner == "Up" else 1.0 - probability_up
    )
    log_loss = -math.log(predicted_winner_probability)
    brier_score = (probability_up - target) ** 2
    payout = quantity if selected == winner else Decimal("0")
    hypothetical_net = payout - entry_cost
    log_sha = str(source_log_sha256 or "").strip().lower()
    if len(log_sha) != 64 or any(
        character not in "0123456789abcdef" for character in log_sha
    ):
        raise ValueError("Polymarket shadow source log digest is invalid")

    source_root = Path(__file__).parent
    artifact: dict[str, object] = {
        "schema_version": SHADOW_SETTLEMENT_SCHEMA_VERSION,
        "opportunity_artifact_sha256": opportunity_sha,
        "source_log_sha256": log_sha,
        "source_commit": _source_commit(source_commit),
        "implementation_sha256": {
            "settlement": _file_sha256(Path(__file__)),
            "opportunity": _file_sha256(
                source_root
                / "polymarket_historical_shadow_opportunity.py"
            ),
        },
        "identity": {
            "gamma_market_id": str(verified["gamma_market_id"]),
            "condition_id": str(verified["condition_id"]),
            "slug": str(verified["slug"]),
            "event_start_ms": int(verified["event_start_ms"]),
            "event_end_ms": int(verified["event_end_ms"]),
            "decision_time_ms": int(verified["decision_time_ms"]),
            "opportunity_observed_at_ms": int(verified["observed_at_ms"]),
            "resolution_observed_at_ms": observed_at,
        },
        "official_resolution": {
            "winner": winner,
            "gamma_payload_sha256": resolution["gamma_payload_sha256"],
            "clob_payload_sha256": resolution["clob_payload_sha256"],
            "gamma_closed": True,
            "clob_closed": True,
            "sources_agree": True,
        },
        "prediction": {
            "probability_up": str(verified["probability_up"]),
            "selected_outcome": selected,
            "selected_outcome_won": selected == winner,
            "log_loss": log_loss,
            "brier_score": brier_score,
        },
        "displayed_depth_counterfactual": {
            "quantity": _decimal_text(quantity),
            "entry_cost_quote": _decimal_text(entry_cost),
            "terminal_payout_quote": _decimal_text(payout),
            "net_pnl_quote": _decimal_text(hypothetical_net),
            "return_on_entry_cost": _decimal_text(
                hypothetical_net / entry_cost
            ),
            "real_order_submitted": False,
            "real_fill_observed": False,
            "fill_assumed_from_displayed_depth": True,
            "interpretation": (
                "Counterfactual only: the pre-settlement displayed-depth "
                "quote is treated as filled exactly; no real order or fill "
                "occurred."
            ),
        },
        "authority": {
            "trading_authority": False,
            "proposal_authority": False,
            "execution_evidence": False,
            "profitability_claim": False,
        },
        "trading_authority": False,
        "profitability_claim": False,
    }
    artifact_sha = _canonical_sha256(artifact)
    return {**artifact, "artifact_sha256": artifact_sha}, artifact_sha


__all__ = [
    "SHADOW_SETTLEMENT_SCHEMA_VERSION",
    "settle_shadow_opportunity",
    "validate_shadow_official_resolution",
]
