from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-portfolio-margin-capital-netting-sensitivity-v1-2026-08-30.json"
)
FUNDING = ROOT / (
    "docs/model-research/action-value/"
    "binance-broad-crypto-funding-carry-preflight-v1-2026-08-27.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_ARTIFACT_HASH = (
    "a2f6c75512ebb3d6b613594cb4cf03ce7473bdcc7a4227cd3a99cbdfd47d7a6d"
)
EXPECTED_FUNDING_HASH = (
    "095009a36a5c6a8a5a2dfdfb3e57ebe6183721bb84600518552ccf6d463617c8"
)
EXPECTED_TERMINAL_SUCCESSOR_HASH = (
    "477a4db3c7f9c594ea8c351ce8f0a766280f062437c8089758d6137fbdd54d86"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _embedded_hash(value: dict[str, object], field: str) -> str:
    payload = dict(value)
    payload.pop(field)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_portfolio_margin_sensitivity_recomputes_from_retained_funding_roles() -> None:
    artifact = _load(ARTIFACT)
    funding = _load(FUNDING)

    assert artifact["result_sha256"] == EXPECTED_ARTIFACT_HASH
    assert _embedded_hash(artifact, "result_sha256") == EXPECTED_ARTIFACT_HASH
    assert funding["result_sha256"] == EXPECTED_FUNDING_HASH
    assert _embedded_hash(funding, "result_sha256") == EXPECTED_FUNDING_HASH

    rows: list[dict[str, object]] = []
    for symbol in funding["symbol_results"]:
        for role in ("training", "validation", "test"):
            metrics = symbol["roles"][role]
            released_hurdle = Decimal(metrics["capital_opportunity_hurdle_bips"]) / 2
            rows.append(
                {
                    "symbol": symbol["future_symbol"],
                    "role": role,
                    "net": Decimal(metrics["net_after_frozen_hurdles_bips"])
                    + released_hurdle,
                    "bootstrap": Decimal(
                        metrics["family_adjusted_bootstrap_net_lower_bound_bips"]
                    )
                    + released_hurdle,
                }
            )

    population = artifact["sensitivity"]["population"]
    assert len(rows) == population["role_count"] == 51
    assert sum(row["net"] > 0 for row in rows) == 1
    assert sum(row["bootstrap"] > 0 for row in rows) == 0

    symbols = {row["symbol"] for row in rows}
    assert len(symbols) == population["selected_symbol_count"] == 17
    assert not any(
        all(row["net"] > 0 for row in rows if row["symbol"] == symbol)
        for symbol in symbols
    )
    assert not any(
        all(row["bootstrap"] > 0 for row in rows if row["symbol"] == symbol)
        for symbol in symbols
    )

    best = artifact["sensitivity"]["best_role_results"]
    for role in ("training", "validation", "test"):
        role_rows = [row for row in rows if row["role"] == role]
        best_net = max(role_rows, key=lambda row: row["net"])
        best_bootstrap = max(role_rows, key=lambda row: row["bootstrap"])
        assert best_net["symbol"] == best[role]["symbol"]
        assert best_bootstrap["symbol"] == best[role]["symbol"]
        assert best_net["net"] == Decimal(
            best[role]["best_net_after_execution_and_one_capital_leg_bips"]
        )
        assert best_bootstrap["bootstrap"] == Decimal(
            best[role]["best_family_adjusted_bootstrap_lower_bound_bips"]
        )


def test_portfolio_margin_sensitivity_is_terminal_and_not_promoted() -> None:
    artifact = _load(ARTIFACT)
    registry = _load(REGISTRY)

    assert _embedded_hash(registry, "result_sha256") == registry["result_sha256"]
    assert artifact["adjudication"]["accepted_edge"] is False
    assert artifact["adjudication"]["stable_profitability_proved"] is False
    assert artifact["adjudication"]["account_or_book_request_justified"] is False
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_broad_crypto_spot_perpetual_funding_carry_portfolio_margin_single_capital_leg_sensitivity_2026_08_30"
    )
    assert terminal["canonical_result_sha256"] == EXPECTED_TERMINAL_SUCCESSOR_HASH
    assert "zero_of_9_BTC_ETH_SOL_roles" in terminal["reason"]
    assert "zero_family_adjusted_bootstrap_bounds" in terminal["reason"]
