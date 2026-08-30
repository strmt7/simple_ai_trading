"""Run one rejection-first pUSD-to-USDT opportunity-cost quote."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import requests

from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-pusd-to-binance-usdt-opportunity-cost-contract-v1-2026-08-30.json"
)
OUTPUT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-pusd-to-binance-usdt-opportunity-cost-gate-v1-2026-08-30.json"
)
RAW_DIR = ROOT / "data/polymarket-pusd-to-binance-usdt-opportunity-cost-v1/raw"
JOURNAL_PATH = (
    ROOT / "data/polymarket-pusd-to-binance-usdt-opportunity-cost-v1/journal.json"
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: Mapping[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    write_bytes_atomic(path, (_canonical_json(payload) + "\n").encode("ascii"))


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _parse_utc(value: object, *, name: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} lacks an explicit offset")
    return parsed.astimezone(timezone.utc)


def _validate_contract(contract: Mapping[str, object]) -> None:
    if contract.get("schema_version") != (
        "polymarket-pusd-to-binance-usdt-opportunity-cost-v1-contract"
    ):
        raise ValueError("contract schema differs")
    if _canonical_hash(contract, "contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise ValueError("contract hash mismatch")
    implementation = _mapping(contract["implementation"], name="implementation")
    if _sha256(Path(__file__).read_bytes()) != implementation["sha256"]:
        raise ValueError("implementation hash mismatch")
    frozen_at = _parse_utc(contract["frozen_at_utc"], name="frozen_at_utc")
    if frozen_at > datetime.now(timezone.utc):
        raise ValueError("frozen_at_utc is in the future")
    if _parse_utc(contract["bonus_end_utc"], name="bonus_end_utc") <= frozen_at:
        raise ValueError("bonus already expired at freeze time")
    for source in contract["retained_sources"]:
        bound = _mapping(source, name="retained source")
        payload = (ROOT / str(bound["path"])).read_bytes()
        if _sha256(payload) != bound["sha256"]:
            raise ValueError(f"retained source hash mismatch: {bound['path']}")


def _record_request(
    journal: dict[str, object],
    *,
    name: str,
    method: str,
    url: str,
    body: bytes,
) -> dict[str, object]:
    request = {
        "name": name,
        "method": method,
        "url": url,
        "request_body_sha256": _sha256(body),
        "state": "planned",
        "planned_at_ms": time.time_ns() // 1_000_000,
    }
    requests_log = journal["requests"]
    if not isinstance(requests_log, list):
        raise ValueError("journal requests must be a list")
    requests_log.append(request)
    _write_json(JOURNAL_PATH, journal)
    return request


def _save_response(
    journal: dict[str, object],
    request: dict[str, object],
    response: requests.Response,
    raw_path: Path,
) -> None:
    write_bytes_atomic(raw_path, response.content)
    request.update(
        {
            "state": "received",
            "completed_at_ms": time.time_ns() // 1_000_000,
            "status_code": response.status_code,
            "response_bytes": len(response.content),
            "response_sha256": _sha256(response.content),
            "raw_path": raw_path.relative_to(ROOT).as_posix(),
        }
    )
    _write_json(JOURNAL_PATH, journal)


def _get(
    session: requests.Session,
    journal: dict[str, object],
    *,
    name: str,
    url: str,
    suffix: str,
) -> requests.Response:
    request = _record_request(
        journal, name=name, method="GET", url=url, body=b""
    )
    response = session.get(url, timeout=30)
    _save_response(journal, request, response, RAW_DIR / f"{name}.{suffix}")
    response.raise_for_status()
    return response


def _post_json(
    session: requests.Session,
    journal: dict[str, object],
    *,
    name: str,
    url: str,
    body: Mapping[str, object],
) -> requests.Response:
    payload = _canonical_json(body).encode("ascii")
    request = _record_request(
        journal, name=name, method="POST", url=url, body=payload
    )
    response = session.post(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    _save_response(journal, request, response, RAW_DIR / f"{name}.raw.json")
    response.raise_for_status()
    return response


def _validate_documentation(contract: Mapping[str, object], pages: list[bytes]) -> None:
    for source, payload in zip(contract["documentation_sources"], pages, strict=True):
        bound = _mapping(source, name="documentation source")
        text = payload.decode("utf-8").casefold()
        missing = [
            phrase
            for phrase in bound["required_text_casefolded"]
            if str(phrase).casefold() not in text
        ]
        if missing:
            raise ValueError(f"{bound['name']} required text missing: {missing}")


def _select_polygon_usdt(payload: object) -> dict[str, Any]:
    root = _mapping(payload, name="supported assets response")
    assets = root.get("supportedAssets")
    if not isinstance(assets, list):
        raise ValueError("supportedAssets must be an array")
    matches: list[dict[str, Any]] = []
    for raw_asset in assets:
        asset = _mapping(raw_asset, name="supported asset")
        token = _mapping(asset.get("token"), name="supported token")
        if str(asset.get("chainId")) == "137" and token.get("symbol") == "USDT":
            matches.append({**asset, "token": token})
    if len(matches) != 1:
        raise ValueError(f"expected one Polygon USDT row, found {len(matches)}")
    match = matches[0]
    token = match["token"]
    if token.get("decimals") != 6:
        raise ValueError("Polygon USDT decimals differ")
    address = str(token.get("address", ""))
    if not address.startswith("0x") or len(address) != 42:
        raise ValueError("Polygon USDT address is invalid")
    return match


def run() -> dict[str, object]:
    if RAW_DIR.exists() or JOURNAL_PATH.exists() or OUTPUT_PATH.exists():
        raise FileExistsError("one-use raw, journal, or output path already exists")
    contract = _mapping(
        json.loads(CONTRACT_PATH.read_text(encoding="ascii")), name="contract"
    )
    _validate_contract(contract)
    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=False, exist_ok=False)
    probe = RAW_DIR / ".write-probe"
    write_bytes_atomic(probe, b"ready\n")
    probe.unlink()
    journal: dict[str, object] = {
        "schema_version": (
            "polymarket-pusd-to-binance-usdt-opportunity-cost-v1-journal"
        ),
        "state": "started",
        "started_at_ms": time.time_ns() // 1_000_000,
        "contract_sha256": contract["contract_sha256"],
        "requests": [],
    }
    _write_json(JOURNAL_PATH, journal)
    session = requests.Session()
    session.headers.update(
        {
            "Accept-Encoding": "identity",
            "User-Agent": "simple-ai-trading-public-edge-research/1.0",
        }
    )
    try:
        pages = []
        for raw_source in contract["documentation_sources"]:
            source = _mapping(raw_source, name="documentation source")
            response = _get(
                session,
                journal,
                name=str(source["name"]),
                url=str(source["url"]),
                suffix="raw.md",
            )
            pages.append(response.content)
        _validate_documentation(contract, pages)

        assets_response = _get(
            session,
            journal,
            name="supported-assets",
            url=str(contract["supported_assets_url"]),
            suffix="raw.json",
        )
        polygon_usdt = _select_polygon_usdt(assets_response.json())
        amount_pusd = Decimal(str(contract["amount_pusd"]))
        if Decimal(str(polygon_usdt["minCheckoutUsd"])) > amount_pusd:
            raise ValueError("quote amount is below the current route minimum")

        quote_template = _mapping(contract["quote_request"], name="quote request")
        quote_body = {
            **quote_template,
            "toTokenAddress": polygon_usdt["token"]["address"],
        }
        quote_response = _post_json(
            session,
            journal,
            name="quote",
            url=str(contract["quote_url"]),
            body=quote_body,
        )
        quote = _mapping(quote_response.json(), name="quote response")
        output_base_units = Decimal(str(quote["estToTokenBaseUnit"]))
        output_usdt = output_base_units / Decimal(10**6)
        estimated_conversion_loss_usdt = amount_pusd - output_usdt
        estimated_conversion_loss_bips = (
            estimated_conversion_loss_usdt / amount_pusd * Decimal(10_000)
        )

        frozen_at = _parse_utc(contract["frozen_at_utc"], name="frozen_at_utc")
        bonus_end = _parse_utc(contract["bonus_end_utc"], name="bonus_end_utc")
        remaining_seconds = Decimal(str((bonus_end - frozen_at).total_seconds()))
        year_seconds = Decimal(365 * 86_400)
        bonus_apr = Decimal(str(contract["fixed_bonus_apr_percent"])) / Decimal(100)
        holding_apr = Decimal(str(contract["holding_yield_apr_percent"])) / Decimal(
            100
        )
        optimistic_bonus_reward = (
            amount_pusd * bonus_apr * remaining_seconds / year_seconds
        )
        holding_yield_opportunity_cost = (
            amount_pusd * holding_apr * remaining_seconds / year_seconds
        )
        optimistic_incremental_reward = (
            optimistic_bonus_reward - holding_yield_opportunity_cost
        )
        rejected_before_account_access = (
            estimated_conversion_loss_usdt > optimistic_bonus_reward
        )

        result: dict[str, object] = {
            "schema_version": (
                "polymarket-pusd-to-binance-usdt-opportunity-cost-gate-v1"
            ),
            "created_at_ms": time.time_ns() // 1_000_000,
            "purpose": (
                "reject_or_retain_the_capped_cross_venue_USDT_bonus_allocation_"
                "before_account_or_transaction_access"
            ),
            "contract": {
                "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
                "sha256": contract["contract_sha256"],
            },
            "route": {
                "from_chain_id": "137",
                "from_asset": "pUSD",
                "from_amount_pusd": _decimal_text(amount_pusd),
                "to_chain_id": "137",
                "to_asset": "USDT",
                "to_token_address": polygon_usdt["token"]["address"],
                "estimated_output_usdt": _decimal_text(output_usdt),
                "estimated_conversion_loss_usdt": _decimal_text(
                    estimated_conversion_loss_usdt
                ),
                "estimated_conversion_loss_bips": _decimal_text(
                    estimated_conversion_loss_bips
                ),
                "quote_est_checkout_time_ms": quote.get("estCheckoutTimeMs"),
                "quote_est_fee_breakdown": quote.get("estFeeBreakdown"),
                "quote_is_estimate_not_execution": True,
            },
            "economics": {
                "fixed_bonus_apr_percent": str(
                    contract["fixed_bonus_apr_percent"]
                ),
                "holding_yield_apr_percent": str(
                    contract["holding_yield_apr_percent"]
                ),
                "remaining_seconds_at_freeze": _decimal_text(remaining_seconds),
                "optimistic_full_fixed_bonus_reward_usdt": _decimal_text(
                    optimistic_bonus_reward
                ),
                "holding_yield_opportunity_cost_pusd": _decimal_text(
                    holding_yield_opportunity_cost
                ),
                "optimistic_incremental_reward_before_external_cost": (
                    _decimal_text(optimistic_incremental_reward)
                ),
                "variable_realtime_USDT_APR_credited": False,
                "return_conversion_cost_credited": False,
                "Binance_deposit_cost_credited": False,
            },
            "adjudication": {
                "status": (
                    "rejected_before_account_access_by_one_way_conversion_cost"
                    if rejected_before_account_access
                    else "not_rejected_by_quote_but_blocked_before_account_access"
                ),
                "accepted_edge_count_change": 0,
                "accepted_edge": False,
                "deployment_ready": False,
                "profitable_switch_proved": False,
                "market_direction_forecast_required": False,
                "pUSD_to_USDT_one_to_one_assumed": False,
                "rejected_before_account_access": rejected_before_account_access,
                "reason": (
                    "the_optimistic_one_way_executable_conversion_estimate_alone_"
                    "exceeds_the_entire_remaining_fixed_bonus_before_charging_"
                    "holding_yield_opportunity_cost_deposit_return_conversion_"
                    "eligibility_capacity_custody_tax_or_operating_cost"
                    if rejected_before_account_access
                    else "the_quote_does_not_reject_the_route_but_Binance_account_"
                    "eligibility_capacity_deposit_and_all_remaining_costs_are_unbound"
                ),
                "next_trigger": (
                    "material_bridge_quote_or_bonus_term_change;_otherwise_do_not_"
                    "move_pUSD_for_this_bonus_and_do_not_repeat_the_quote"
                ),
            },
            "sources": {
                "retained": contract["retained_sources"],
                "journal": JOURNAL_PATH.relative_to(ROOT).as_posix(),
                "documentation_and_public_response_count": len(
                    journal["requests"]
                ),
            },
            "authority": contract["authority"],
            "implementation": contract["implementation"],
        }
        result["result_sha256"] = _canonical_hash(result, "result_sha256")
        _write_json(OUTPUT_PATH, result)
        journal.update(
            {
                "state": "completed",
                "completed_at_ms": time.time_ns() // 1_000_000,
                "result_sha256": result["result_sha256"],
            }
        )
        _write_json(JOURNAL_PATH, journal)
        return result
    except Exception as exc:
        journal.update(
            {
                "state": "failed",
                "failed_at_ms": time.time_ns() // 1_000_000,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json(JOURNAL_PATH, journal)
        raise


def main() -> int:
    result = run()
    print(
        _canonical_json(
            {
                "status": result["adjudication"]["status"],
                "result_sha256": result["result_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
