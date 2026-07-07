"""Tests for the optimization-round command wrapper."""

from __future__ import annotations

import argparse
import json

from tools import optimization_round


def test_write_startup_status_before_heavy_optimization_imports(tmp_path):
    args = argparse.Namespace(
        docs_root=tmp_path / "docs",
        round_id="Round Startup Smoke",
        market="futures",
        interval="1s",
        objective="conservative",
        compute_backend="directml",
        require_gpu=True,
    )

    status_path = optimization_round._write_startup_status(args)
    payload = json.loads(status_path.read_text(encoding="utf-8"))

    assert status_path == tmp_path / "docs" / "round-startup-smoke" / "data" / "round-status.json"
    assert payload["phase"] == "process_startup"
    assert payload["status"] == "running"
    assert payload["compute_backend"] == "directml"
    assert payload["require_gpu"] is True
