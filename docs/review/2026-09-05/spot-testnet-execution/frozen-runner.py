"""Isolated BTC/ETH/SOL Spot-testnet lifecycle campaign; never a strategy claim."""

from __future__ import annotations

import argparse
from decimal import Decimal as D
from getpass import getpass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from simple_ai_trading.spot_testnet_evidence import (
    Rules,
    decimal,
    grid,
    owned_trade_cash,
)
from tools.spot_testnet_campaign_transport import Journal, SYMBOLS, Transport

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/review/2026-09-05/spot-testnet-execution"
TERMINAL = {"FILLED", "CANCELED", "EXPIRED", "EXPIRED_IN_MATCH", "REJECTED"}


class Campaign:
    def __init__(self, transport: Transport, plan: dict[str, Any]):
        self.t, self.plan = transport, plan
        self.rules: dict[str, Rules] = {}
        self.cash: dict[str, dict[str, Any]] = {}

    def preflight(self, *, recover: bool = False) -> None:
        """Check exact venue filters, clock, fee basis and virtual capital before orders."""
        before = time.time_ns() // 1_000_000
        clock = self.t.call("GET", "/api/v3/time")
        after = time.time_ns() // 1_000_000
        if (
            type(clock.get("serverTime")) is not int
            or not before - 1000 <= clock["serverTime"] <= after + 1000
        ):
            raise ValueError("clock ambiguity")
        info = self.t.call(
            "GET",
            "/api/v3/exchangeInfo",
            {"symbols": json.dumps(SYMBOLS, separators=(",", ":"))},
        )
        if sorted(x["symbol"] for x in info["symbols"]) != sorted(SYMBOLS):
            raise ValueError("symbol population differs")
        limits = [
            int(x["limit"])
            for x in info["rateLimits"]
            if x["rateLimitType"] == "REQUEST_WEIGHT"
            and x["interval"] == "MINUTE"
            and x["intervalNum"] == 1
        ]
        if len(limits) != 1 or limits[0] < 600:
            raise ValueError("insufficient documented request budget")
        self.t.weight_limit = limits[0]
        for item in info["symbols"]:
            rules = Rules.parse(item)
            self.rules[rules.symbol] = rules
        self.t.journal.add("public_configuration", normalized_response=info)
        account = self.t.call("GET", "/api/v3/account")
        if account.get("canTrade") is not True:
            raise ValueError("test account cannot trade")
        quote = next(x for x in account["balances"] if x["asset"] == "USDT")
        if not recover and decimal(quote["free"]) < D(650):
            raise ValueError("insufficient virtual quote reserve")
        self.t.journal.add(
            "account_preflight",
            can_trade=True,
            virtual_quote_reserve_sufficient=decimal(quote["free"]) >= D(650),
            recovery_only=recover,
        )
        for symbol in SYMBOLS:
            fees = self.t.call("GET", "/api/v3/account/commission", {"symbol": symbol})
            if fees.get("symbol") != symbol:
                raise ValueError("fee symbol mismatch")
            for category in (
                "standardCommission",
                "specialCommission",
                "taxCommission",
            ):
                if any(
                    decimal(fees[category][role]) != 0
                    for role in ("maker", "taker", "buyer", "seller")
                ):
                    raise ValueError(
                        "campaign requires the observed zero-fee testnet schedule"
                    )
            self.t.journal.add(
                "fee_preflight", symbol=symbol, all_twelve_numeric_rates_zero=True
            )

    def book(self, symbol: str) -> tuple[D, D]:
        row = self.t.call("GET", "/api/v3/ticker/bookTicker", {"symbol": symbol})
        if row.get("symbol") != symbol:
            raise ValueError("book symbol mismatch")
        bid, ask = decimal(row["bidPrice"]), decimal(row["askPrice"])
        if bid <= 0 or ask < bid or ask / bid > D("1.01"):
            raise ValueError("invalid or wide book")
        self.t.journal.add(
            "book_observation",
            symbol=symbol,
            bid=str(bid),
            ask=str(ask),
            timestamp_semantics="HTTP observation only, not exchange quote-update time",
        )
        return bid, ask

    def no_foreign_open_orders(self, symbol: str) -> None:
        rows = self.t.call("GET", "/api/v3/openOrders", {"symbol": symbol})
        if not isinstance(rows, list):
            raise ValueError("invalid open orders")
        own = {x["newClientOrderId"] for x in self.t.journal.intents()}
        foreign = sum(row.get("clientOrderId") not in own for row in rows)
        self.t.journal.add(
            "open_order_check", symbol=symbol, total=len(rows), foreign_count=foreign
        )
        if foreign:
            raise ValueError("foreign open orders: leave this symbol untouched")

    def submit(
        self, symbol: str, side: str, qty: D, price: D, *, maker: bool = False
    ) -> dict[str, Any]:
        rule = self.rules[symbol]
        rule.validate(qty, price)
        intents = self.t.journal.intents()
        if len(intents) >= 15:
            raise ValueError("placement ceiling")
        if side == "BUY":
            self.no_foreign_open_orders(symbol)
            spent = sum(
                decimal(x["quantity"]) * decimal(x["price"])
                for x in intents
                if x["side"] == "BUY"
            )
            if qty * price > D(100) or spent + qty * price > D(600):
                raise ValueError("virtual entry notional ceiling")
        elif side == "SELL":
            if qty > self.net(symbol):
                raise ValueError("close exceeds campaign-owned base")
        else:
            raise ValueError("invalid side")
        cid = f"{self.plan['client_id_prefix']}-{len(intents):02d}"
        params = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT_MAKER" if maker else "LIMIT",
            "quantity": format(qty, "f"),
            "price": format(price, "f"),
            "newClientOrderId": cid,
            "newOrderRespType": "FULL",
            "selfTradePreventionMode": "EXPIRE_TAKER",
        }
        if not maker:
            params["timeInForce"] = "IOC"
        self.t.journal.add("order_intent", params=params)
        try:
            reply = self.t.call("POST", "/api/v3/order", params)
            # Intentionally ignore the response body, then reconcile by durable client ID.
            self.t.journal.add(
                "ack_body_discarded_for_reconciliation_exercise",
                client_id=cid,
                received=isinstance(reply, dict),
            )
        except RuntimeError:
            self.t.journal.add("submission_ambiguous", client_id=cid)
        return params

    def finish(self, intent: dict[str, Any]) -> None:
        """Resolve exactly one owned intent, cancel if active, then reconcile its fills."""
        symbol, cid = intent["symbol"], intent["newClientOrderId"]
        known = [
            x
            for x in self.t.journal.rows
            if x["kind"] == "order_identity" and x["original_client_id"] == cid
        ]
        query = {
            "symbol": symbol,
            **(
                {"orderId": known[-1]["orderId"]}
                if known
                else {"origClientOrderId": cid}
            ),
        }
        order = self.t.order_view(self.t.call("GET", "/api/v3/order", query), intent)
        if order["status"] not in TERMINAL:
            try:
                self.t.call(
                    "DELETE",
                    "/api/v3/order",
                    {
                        "symbol": symbol,
                        "orderId": order["orderId"],
                        "newClientOrderId": cid + "c",
                    },
                )
            except RuntimeError:
                self.t.journal.add("cancel_ambiguous", client_id=cid)
            order = self.t.order_view(
                self.t.call(
                    "GET",
                    "/api/v3/order",
                    {"symbol": symbol, "orderId": order["orderId"]},
                ),
                intent,
            )
        if order["status"] not in TERMINAL:
            raise ValueError("order still nonterminal; no new exposure")
        trades = self.t.trade_view(
            self.t.call(
                "GET",
                "/api/v3/myTrades",
                {"symbol": symbol, "orderId": order["orderId"], "limit": 1000},
            ),
            order,
        )
        cash = owned_trade_cash(order, trades, self.rules[symbol].base)
        if cash["third_asset_fees"]:
            raise ValueError("unvalued third-asset commission")
        self.cash[cid] = {"symbol": symbol, **cash}
        self.t.journal.add("order_cash", client_id=cid, **self.cash[cid])

    def net(self, symbol: str) -> D:
        return sum(
            (D(x["base_delta"]) for x in self.cash.values() if x["symbol"] == symbol),
            D(0),
        )

    def flatten(self, symbol: str) -> None:
        rule = self.rules[symbol]
        for _ in range(3):
            net = self.net(symbol)
            if net == 0:
                return
            if net < 0:
                raise ValueError("negative owned base")
            closes = [
                x
                for x in self.t.journal.intents()
                if x["symbol"] == symbol and x["side"] == "SELL"
            ]
            if len(closes) >= 3:
                raise ValueError(
                    "close-attempt ceiling; owned residual requires attention"
                )
            bid, _ = self.book(symbol)
            qty, price = grid(net, rule.step), grid(bid * D("0.995"), rule.tick)
            if qty <= 0 or qty * price < rule.min_notional:
                self.t.journal.add("owned_dust", symbol=symbol, quantity=str(net))
                raise ValueError(
                    "owned dust cannot be rounded up using foreign inventory"
                )
            intent = self.submit(symbol, "SELL", qty, price)
            self.finish(intent)
        if self.net(symbol):
            raise ValueError("owned residual after bounded closes")

    def execute(self, *, recover: bool = False) -> dict[str, Any]:
        self.preflight(recover=recover)
        if recover:
            for intent in self.t.journal.intents():
                self.finish(intent)
            for symbol in SYMBOLS:
                self.flatten(symbol)
        else:
            for symbol in SYMBOLS:
                self.no_foreign_open_orders(symbol)
                rule = self.rules[symbol]
                bid, _ = self.book(symbol)
                price = grid(bid * D("0.99"), rule.tick)
                passive = self.submit(
                    symbol, "BUY", grid(D(100) / price, rule.step), price, maker=True
                )
                # Rebuild durable state while the owned order can still be resting.
                self.t.journal = Journal(self.t.journal.path)
                self.t.journal.add(
                    "journal_reload",
                    client_id=passive["newClientOrderId"],
                    cold_process_restart=False,
                )
                self.finish(passive)
                self.flatten(symbol)
                _, ask = self.book(symbol)
                price = grid(ask * D("1.002"), rule.tick, up=True)
                active = self.submit(
                    symbol, "BUY", grid(D(100) / price, rule.step), price
                )
                self.finish(active)
                self.flatten(symbol)
        residuals = {s: str(self.net(s)) for s in SYMBOLS}
        for symbol in SYMBOLS:
            self.no_foreign_open_orders(symbol)
            if self.t.journal.rows[-1]["total"] != 0:
                raise ValueError("orders remain open")
        coverage = {}
        for symbol in SYMBOLS:
            intents = [x for x in self.t.journal.intents() if x["symbol"] == symbol]
            maker_ids = {
                x["newClientOrderId"] for x in intents if x["type"] == "LIMIT_MAKER"
            }
            observations = [
                x["order"]
                for x in self.t.journal.rows
                if x["kind"] == "order_observation" and x["order"]["symbol"] == symbol
            ]
            coverage[symbol] = {
                "resting_order_observed": any(
                    x["status"] == "NEW" and x["clientOrderId"] in maker_ids
                    for x in observations
                ),
                "cancellation_observed": any(
                    x["status"] == "CANCELED"
                    and x["clientOrderId"] in {cid + "c" for cid in maker_ids}
                    for x in observations
                ),
                "buy_fill_observed": any(
                    x["symbol"] == symbol and D(x["base_delta"]) > 0
                    for x in self.cash.values()
                ),
                "sell_fill_observed": any(
                    x["symbol"] == symbol and D(x["base_delta"]) < 0
                    for x in self.cash.values()
                ),
                "owned_flat": self.net(symbol) == 0,
            }
        return {
            "completed": True,
            "live_case_coverage": coverage,
            "required_live_cases_passed": all(
                all(row.values()) for row in coverage.values()
            ),
            "testnet_only": True,
            "residual_base": residuals,
            "owned_order_intents": len(self.t.journal.intents()),
            "order_cash": self.cash,
            "quote_cash_delta": str(
                sum((D(x["quote_delta"]) for x in self.cash.values()), D(0))
            ),
            "partial_fills_observed": sum(
                x["partial_fill"] for x in self.cash.values()
            ),
            "cold_process_restart_tested": False,
            "profitability_evidence": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("execute", "recover"), required=True)
    args = parser.parse_args()
    plan = json.loads((BASE / "plan.json").read_bytes())
    for path, expected in plan["implementation_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError("frozen implementation changed")
    journal = Journal(BASE / "journal.jsonl", create=args.mode == "execute")
    if args.mode == "recover" and not journal.intents():
        raise ValueError("no owned intents to recover")
    journal.add("run_started", mode=args.mode)
    key, secret = (
        getpass("Spot testnet key (hidden): "),
        getpass("Spot testnet secret (hidden): "),
    )
    campaign = Campaign(Transport(key, secret, journal, plan["client_id_prefix"]), plan)
    key = secret = ""
    try:
        result = campaign.execute(recover=args.mode == "recover")
    except Exception as exc:
        trace = exc.__traceback__
        while trace and trace.tb_next:
            trace = trace.tb_next
        result = {
            "completed": False,
            "failure_type": type(exc).__name__,
            "failure_location": {
                "file": Path(trace.tb_frame.f_code.co_filename).name,
                "line": trace.tb_lineno,
            }
            if trace
            else None,
            "testnet_only": True,
            "new_entries_stopped": True,
            "known_order_cash": campaign.cash,
            "owned_intents_requiring_exact_reconciliation": [
                x["newClientOrderId"] for x in campaign.t.journal.intents()
            ],
            "profitability_evidence": False,
        }
    finally:
        campaign.t.client.session.close()
    campaign.t.journal.add("run_finished", **result)
    output = BASE / (
        "result.json"
        if args.mode == "execute"
        else f"recovery-{sum(x['kind'] == 'run_started' for x in campaign.t.journal.rows)}.json"
    )
    with output.open("x", encoding="ascii") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
