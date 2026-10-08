from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.reproduce_inventory_cash_actions import ROOT, control_report, main


def test_retained_controls_match_current_sources_and_exact_reproduction():
    assert "docs/review/2026-10-08/inventory-action-*.json text eol=lf" in (
        ROOT / ".gitattributes"
    ).read_text(encoding="utf-8")
    retained = json.loads(
        (ROOT / "docs/review/2026-10-08/inventory-action-controls.json").read_bytes()
    )
    assert retained == control_report()
    assert retained["financially_qualified"] is False
    assert retained["financial_counts_unchanged"]["qualified_stable_edges"] == 0
    for name, expected in retained["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected


def test_control_writer_retains_readback_without_overwriting(tmp_path: Path):
    output = tmp_path / "controls.json"
    assert main(["--output", str(output)]) == 0
    raw = output.read_bytes()
    assert json.loads(raw) == control_report()
    assert raw.endswith(b"\n")
    with pytest.raises(FileExistsError):
        main(["--output", str(output)])
    assert output.read_bytes() == raw


def test_missing_output_directory_is_not_created_implicitly(tmp_path: Path):
    output = tmp_path / "missing" / "controls.json"
    with pytest.raises(FileNotFoundError):
        main(["--output", str(output)])
    assert not output.parent.exists()
