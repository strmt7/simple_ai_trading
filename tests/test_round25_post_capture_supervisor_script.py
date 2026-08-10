from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_polymarket_round25_post_capture_scheduled.ps1"


def test_round25_scheduled_supervisor_is_bounded_and_fail_closed() -> None:
    text = SCRIPT.read_text(encoding="ascii")

    required = (
        "[ValidatePattern('^[0-9a-f]{40}$')]",
        "MaximumResolutionConditions = 128",
        "MaximumRuntimeHours = 8",
        "MaximumLogMiB = 1",
        "WaitOne(0)",
        "git -C $Repository rev-parse HEAD",
        "git -C $Repository status --porcelain",
        "--maximum-resolution-conditions",
        "source_database_opened",
        "credentials_accessed = $false",
        "orders_submitted = 0",
        "paper_trading_authority = $false",
        "live_trading_authority = $false",
        "Stop-OwnedProcessTree",
        "Write-BoundedLog",
    )
    assert all(value in text for value in required)
    assert "Start-Process" not in text
    assert "Invoke-Expression" not in text
