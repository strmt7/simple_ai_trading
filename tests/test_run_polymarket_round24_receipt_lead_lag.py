from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "run_polymarket_round24_receipt_lead_lag.py"


def _tool() -> object:
    spec = importlib.util.spec_from_file_location(
        "run_polymarket_round24_receipt_lead_lag",
        TOOL_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Round 24 one-use tool spec is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arguments(tmp_path: Path, acknowledgement: str) -> list[str]:
    return [
        "--core-database",
        str(tmp_path / "core.duckdb"),
        "--terminal-transport-manifest",
        str(tmp_path / "terminal.json"),
        "--sidecar-database",
        str(tmp_path / "sidecar.duckdb"),
        "--sidecar-terminal-manifest",
        str(tmp_path / "sidecar-terminal.json"),
        "--materialization-evidence",
        str(tmp_path / "materialization.json"),
        "--selection-claim",
        str(tmp_path / "claim.json"),
        "--output",
        str(tmp_path / "result.json"),
        "--acknowledgement",
        acknowledgement,
    ]


def test_round24_one_use_tool_rejects_acknowledgement_before_data_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _tool()
    accessed = False

    def loader(_path: Path) -> dict[str, object]:
        nonlocal accessed
        accessed = True
        return {}

    monkeypatch.setattr(tool, "load_round25_terminal_transport_manifest", loader)

    with pytest.raises(ValueError, match="acknowledgement differs"):
        tool.main(_arguments(tmp_path, "wrong"))

    assert accessed is False
    assert not (tmp_path / "materialization.json").exists()
    assert not (tmp_path / "claim.json").exists()
    assert not (tmp_path / "result.json").exists()


def test_round24_one_use_tool_publishes_source_evidence_before_claim_and_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _tool()
    terminal = {"manifest_sha256": "a" * 64}
    sidecar = {"manifest_sha256": "b" * 64}
    evidence = {
        "core_materialization": {"materialization_sha256": "c" * 64},
        "terminal_receipt_audit": {"audit_sha256": "d" * 64},
        "optional_sidecar_joined": True,
        "row_count": 12,
    }
    monkeypatch.setattr(
        tool,
        "load_round25_terminal_transport_manifest",
        lambda _path: terminal,
    )
    monkeypatch.setattr(
        tool,
        "load_round21_sidecar_terminal_manifest",
        lambda _path: sidecar,
    )
    monkeypatch.setattr(
        tool,
        "assemble_round24_receipt_rows",
        lambda **_kwargs: ({"specification_sha256": "e" * 64}, (object(),), evidence),
    )

    def run_result(
        *, spec: object, rows: object, claim_selection: object
    ) -> dict[str, object]:
        assert spec == {"specification_sha256": "e" * 64}
        assert len(rows) == 1
        assert (tmp_path / "materialization.json").exists()
        claim_selection({"claim_sha256": "f" * 64, "status": "frozen"})
        return {
            "conclusion": "fixture_result",
            "mechanism_gate_passed": False,
            "result_sha256": "1" * 64,
        }

    monkeypatch.setattr(tool, "run_round24_receipt_lead_lag", run_result)

    assert tool.main(_arguments(tmp_path, tool.ACKNOWLEDGEMENT)) == 0

    assert (
        json.loads((tmp_path / "materialization.json").read_text("ascii")) == evidence
    )
    assert (
        json.loads((tmp_path / "claim.json").read_text("ascii"))["claim_sha256"]
        == "f" * 64
    )
    assert (
        json.loads((tmp_path / "result.json").read_text("ascii"))["result_sha256"]
        == "1" * 64
    )
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output[0]["event"] == "round24_receipt_holdout_start"
    assert output[-1]["event"] == "round24_receipt_holdout_complete"
    assert output[-1]["materialization_evidence"].endswith("materialization.json")


def test_round24_one_use_tool_refuses_any_preexisting_evidence_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _tool()
    (tmp_path / "materialization.json").write_text("{}\n", encoding="ascii")
    accessed = False

    def loader(_path: Path) -> dict[str, object]:
        nonlocal accessed
        accessed = True
        return {}

    monkeypatch.setattr(tool, "load_round25_terminal_transport_manifest", loader)

    with pytest.raises(FileExistsError, match="must all be new"):
        tool.main(_arguments(tmp_path, tool.ACKNOWLEDGEMENT))

    assert accessed is False
