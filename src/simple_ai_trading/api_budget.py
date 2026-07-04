"""Binance API budget parsing and rendering helpers."""

from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass
from typing import Mapping


_HEADER_PATTERN = re.compile(
    r"^x-mbx-(?P<kind>used-weight|order-count)-(?P<num>\d+)(?P<letter>[smhd])$",
    re.IGNORECASE,
)
_INTERVAL_LETTERS = {
    "SECOND": "S",
    "MINUTE": "M",
    "HOUR": "H",
    "DAY": "D",
}
_TYPE_BY_HEADER_KIND = {
    "used-weight": "REQUEST_WEIGHT",
    "order-count": "ORDERS",
}


@dataclass(frozen=True)
class ApiBudgetLine:
    rate_limit_type: str
    interval_num: int
    interval_letter: str
    used: int | None
    limit: int | None
    remaining: int | None
    remaining_pct: float | None
    header: str | None
    status: str

    @property
    def interval_label(self) -> str:
        return f"{self.interval_num}{self.interval_letter}"

    @property
    def interval_ms(self) -> int:
        multiplier = {"S": 1000, "M": 60_000, "H": 3_600_000, "D": 86_400_000}.get(self.interval_letter, 0)
        return max(0, self.interval_num * multiplier)

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["interval_label"] = self.interval_label
        payload["interval_ms"] = self.interval_ms
        return payload


@dataclass(frozen=True)
class ApiBudgetReport:
    status: str
    generated_at_ms: int
    market_type: str
    lines: tuple[ApiBudgetLine, ...]
    retry_after_seconds: float | None = None
    source: str = "binance_response_headers"

    def asdict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "generated_at_ms": self.generated_at_ms,
            "market_type": self.market_type,
            "lines": [line.asdict() for line in self.lines],
            "retry_after_seconds": self.retry_after_seconds,
            "source": self.source,
            "summary": summarize_api_budget(self),
        }


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_int(value: object) -> int | None:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def _safe_float(value: object) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _limits_by_key(exchange_info: Mapping[str, object] | None) -> dict[tuple[str, int, str], int]:
    if not exchange_info:
        return {}
    rate_limits = exchange_info.get("rateLimits")
    if not isinstance(rate_limits, list):
        return {}
    limits: dict[tuple[str, int, str], int] = {}
    for item in rate_limits:
        if not isinstance(item, Mapping):
            continue
        rate_type = str(item.get("rateLimitType") or "").upper()
        interval = str(item.get("interval") or "").upper()
        interval_letter = _INTERVAL_LETTERS.get(interval)
        interval_num = _safe_int(item.get("intervalNum"))
        limit = _safe_int(item.get("limit"))
        if not rate_type or interval_letter is None or interval_num is None or limit is None:
            continue
        limits[(rate_type, interval_num, interval_letter)] = limit
    return limits


def _headers_from_request_info(request_info: Mapping[str, object] | None) -> Mapping[str, object]:
    if not request_info:
        return {}
    headers = request_info.get("rate_limit_headers")
    return headers if isinstance(headers, Mapping) else {}


def build_api_budget_report(
    *,
    market_type: str,
    exchange_info: Mapping[str, object] | None = None,
    request_info: Mapping[str, object] | None = None,
    generated_at_ms: int | None = None,
) -> ApiBudgetReport:
    """Combine Binance exchange limits with last-response usage headers."""

    limits = _limits_by_key(exchange_info)
    lines_by_key: dict[tuple[str, int, str], ApiBudgetLine] = {}
    headers = _headers_from_request_info(request_info)
    for header, value in headers.items():
        match = _HEADER_PATTERN.match(str(header))
        if match is None:
            continue
        header_kind = match.group("kind").lower()
        rate_type = _TYPE_BY_HEADER_KIND[header_kind]
        interval_num = int(match.group("num"))
        interval_letter = match.group("letter").upper()
        used = _safe_int(value)
        limit = limits.get((rate_type, interval_num, interval_letter))
        remaining = None if used is None or limit is None else max(0, limit - used)
        remaining_pct = None if remaining is None or not limit else remaining / limit
        status = "unknown"
        if remaining_pct is not None:
            status = "critical" if remaining_pct < 0.05 else ("warn" if remaining_pct < 0.15 else "ok")
        lines_by_key[(rate_type, interval_num, interval_letter)] = ApiBudgetLine(
            rate_limit_type=rate_type,
            interval_num=interval_num,
            interval_letter=interval_letter,
            used=used,
            limit=limit,
            remaining=remaining,
            remaining_pct=remaining_pct,
            header=str(header),
            status=status,
        )

    for key, limit in limits.items():
        if key in lines_by_key:
            continue
        rate_type, interval_num, interval_letter = key
        lines_by_key[key] = ApiBudgetLine(
            rate_limit_type=rate_type,
            interval_num=interval_num,
            interval_letter=interval_letter,
            used=None,
            limit=limit,
            remaining=None,
            remaining_pct=None,
            header=None,
            status="unknown",
        )

    lines = tuple(
        sorted(
            lines_by_key.values(),
            key=lambda line: (line.rate_limit_type, line.interval_letter, line.interval_num),
        )
    )
    retry_after = _safe_float(request_info.get("retry_after_seconds")) if request_info else None
    statuses = {line.status for line in lines}
    if retry_after is not None:
        status = "blocked"
    elif "critical" in statuses:
        status = "critical"
    elif "warn" in statuses:
        status = "warn"
    elif lines and statuses == {"ok"}:
        status = "ok"
    else:
        status = "unknown"
    return ApiBudgetReport(
        status=status,
        generated_at_ms=_now_ms() if generated_at_ms is None else int(generated_at_ms),
        market_type=str(market_type or "spot"),
        lines=lines,
        retry_after_seconds=retry_after,
    )


def summarize_api_budget(report: ApiBudgetReport | Mapping[str, object] | None) -> str:
    if report is None:
        return "API budget: no sample"
    if isinstance(report, ApiBudgetReport):
        status = report.status
        retry_after = report.retry_after_seconds
        lines = [line.asdict() for line in report.lines]
    else:
        status = str(report.get("status") or "unknown")
        retry_after = _safe_float(report.get("retry_after_seconds"))
        raw_lines = report.get("lines")
        lines = [dict(item) for item in raw_lines if isinstance(item, Mapping)] if isinstance(raw_lines, list) else []
    if retry_after is not None:
        return f"API budget: blocked retry_after={retry_after:.1f}s"
    measured = [line for line in lines if line.get("remaining") is not None and line.get("limit") is not None]
    if not measured:
        return f"API budget: {status} used weight unknown"
    measured.sort(key=lambda line: float(line.get("remaining_pct") or 1.0))
    tightest = measured[0]
    remaining = int(tightest.get("remaining") or 0)
    limit = int(tightest.get("limit") or 0)
    used = int(tightest.get("used") or 0)
    label = str(tightest.get("interval_label") or f"{tightest.get('interval_num', '')}{tightest.get('interval_letter', '')}")
    rate_type = str(tightest.get("rate_limit_type") or "REQUEST_WEIGHT")
    return f"API budget: {status} {rate_type} remaining={remaining}/{limit} used={used} window={label}"


def api_budget_startup_block_reason(
    report: ApiBudgetReport | Mapping[str, object] | None,
    *,
    max_used_ratio: float = 0.80,
) -> str | None:
    """Return a fail-closed live-startup reason when Binance budget is too tight."""

    if report is None:
        return None
    payload = report.asdict() if isinstance(report, ApiBudgetReport) else dict(report)
    retry_after = _safe_float(payload.get("retry_after_seconds"))
    if retry_after is not None:
        return f"Binance API budget guard blocked startup: exchange requested retry-after {retry_after:.1f}s"
    raw_lines = payload.get("lines")
    lines = [dict(item) for item in raw_lines if isinstance(item, Mapping)] if isinstance(raw_lines, list) else []
    threshold = max(0.0, min(1.0, float(max_used_ratio)))
    for line in lines:
        used = _safe_int(line.get("used"))
        limit = _safe_int(line.get("limit"))
        if used is None or limit is None or limit <= 0:
            continue
        used_ratio = used / limit
        if used_ratio >= threshold:
            label = str(line.get("interval_label") or f"{line.get('interval_num', '')}{line.get('interval_letter', '')}")
            rate_type = str(line.get("rate_limit_type") or "REQUEST_WEIGHT")
            return (
                "Binance API budget guard blocked startup: "
                f"{rate_type} {label} usage {used}/{limit} ({used_ratio:.1%}) is at or above "
                f"the {threshold:.0%} live-start threshold"
            )
    return None


def render_api_budget(report: ApiBudgetReport | Mapping[str, object] | None) -> str:
    if report is None:
        return "API budget\nstatus=unknown\nwarning: no cached Binance rate-limit sample"
    payload = report.asdict() if isinstance(report, ApiBudgetReport) else dict(report)
    lines = ["API budget", f"status={payload.get('status') or 'unknown'} market={payload.get('market_type') or 'unknown'}"]
    if payload.get("retry_after_seconds") is not None:
        lines.append(f"retry_after_seconds={float(payload['retry_after_seconds']):.1f}")
    for item in payload.get("lines", []) if isinstance(payload.get("lines"), list) else []:
        if not isinstance(item, Mapping):
            continue
        remaining = item.get("remaining")
        limit = item.get("limit")
        used = item.get("used")
        label = item.get("interval_label") or f"{item.get('interval_num', '')}{item.get('interval_letter', '')}"
        if remaining is None or limit is None:
            lines.append(f"{item.get('rate_limit_type')} {label}: used={used if used is not None else 'unknown'} limit={limit if limit is not None else 'unknown'}")
        else:
            lines.append(f"{item.get('rate_limit_type')} {label}: remaining={remaining}/{limit} used={used} status={item.get('status')}")
    lines.append(str(payload.get("summary") or summarize_api_budget(payload)))
    return "\n".join(lines)


__all__ = [
    "ApiBudgetLine",
    "ApiBudgetReport",
    "api_budget_startup_block_reason",
    "build_api_budget_report",
    "render_api_budget",
    "summarize_api_budget",
]
