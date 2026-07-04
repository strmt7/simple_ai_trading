from __future__ import annotations

from simple_ai_trading.api_budget import (
    api_budget_startup_block_reason,
    build_api_budget_report,
    render_api_budget,
    summarize_api_budget,
)


def test_api_budget_combines_exchange_limits_and_headers() -> None:
    report = build_api_budget_report(
        market_type="spot",
        exchange_info={
            "rateLimits": [
                {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 1200},
                {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 50},
            ]
        },
        request_info={
            "rate_limit_headers": {
                "X-MBX-USED-WEIGHT-1M": "240",
                "X-MBX-ORDER-COUNT-10S": "2",
            }
        },
        generated_at_ms=123,
    )

    payload = report.asdict()

    assert report.status == "ok"
    assert payload["generated_at_ms"] == 123
    assert payload["lines"][0]["interval_ms"] == 10_000
    assert any(line["remaining"] == 960 for line in payload["lines"])
    assert "remaining=960/1200" in summarize_api_budget(report)
    assert "REQUEST_WEIGHT 1M" in render_api_budget(report)


def test_api_budget_startup_guard_blocks_at_eighty_percent_and_retry_after() -> None:
    report = build_api_budget_report(
        market_type="spot",
        exchange_info={"rateLimits": [{"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 1200}]},
        request_info={"rate_limit_headers": {"X-MBX-USED-WEIGHT-1M": "960"}},
    )

    reason = api_budget_startup_block_reason(report, max_used_ratio=0.80)

    assert reason is not None
    assert "960/1200" in reason
    assert "80%" in reason

    retry_report = build_api_budget_report(
        market_type="spot",
        request_info={"retry_after_seconds": 2.5},
    )
    assert "retry-after 2.5s" in str(api_budget_startup_block_reason(retry_report))


def test_api_budget_allows_unknown_limits_but_reports_unknown() -> None:
    report = build_api_budget_report(
        market_type="spot",
        request_info={"rate_limit_headers": {"X-MBX-USED-WEIGHT-1M": "47"}},
    )

    assert report.status == "unknown"
    assert api_budget_startup_block_reason(report) is None
    assert "used weight unknown" in summarize_api_budget(report)
