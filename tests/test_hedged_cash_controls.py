from __future__ import annotations

import hashlib
import json

import pytest

from tools.reproduce_hedged_inventory_cash import ROOT, control_report, main
from tools import reproduce_hedged_inventory_cash as module


def test_controls_reproduce_with_exact_source_and_result_bindings():
    report = json.loads(
        (
            ROOT / "docs/review/2026-11-09/paired-inventory-cash/controls.json"
        ).read_bytes()
    )
    assert report == control_report()
    canonical = {key: value for key, value in report.items() if key != "result_sha256"}
    assert (
        report["result_sha256"]
        == hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    for name, expected in report["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    assert report["market_observations"] == 0
    assert not report["qualified_edge"] and not report["training_admitted"]


def test_control_writer_is_exclusive_and_does_not_print_the_output_path(
    tmp_path, capsys
):
    output = tmp_path / "controls.json"
    assert main(["--output", str(output)]) == 0
    assert str(output) not in capsys.readouterr().out
    raw = output.read_bytes()
    assert json.loads(raw) == control_report()
    with pytest.raises(FileExistsError):
        main(["--output", str(output)])
    assert output.read_bytes() == raw


def test_missing_output_parent_remains_missing(tmp_path):
    output = tmp_path / "missing" / "controls.json"
    with pytest.raises(FileNotFoundError):
        main(["--output", str(output)])
    assert not output.parent.exists()


@pytest.mark.parametrize("failure", ["intent", "source"])
def test_changed_frozen_sources_stop_before_generating_controls(
    tmp_path, monkeypatch, failure
):
    intent = json.loads((ROOT / module.INTENT).read_bytes())
    intent["source_sha256"] = {"fixture.py": "0" * 64}
    (tmp_path / "fixture.py").write_text("changed source", encoding="utf-8")
    if failure == "source":
        intent["sha256"] = hashlib.sha256(
            json.dumps(
                {k: v for k, v in intent.items() if k != "sha256"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    target = tmp_path / module.INTENT
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(intent), encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="intent digest|sources changed"):
        module.control_report()
