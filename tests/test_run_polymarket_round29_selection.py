from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_polymarket_round29_selection",
    ROOT / "tools/run_polymarket_round29_selection.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_round29_runner_parser_keeps_resource_use_bounded() -> None:
    args = TOOL._parser().parse_args(
        [
            "--round27-feature-store",
            "feature.duckdb",
            "--round28-overlay-store",
            "overlay.duckdb",
            "--round27-target-store",
            "target.duckdb",
            "--selection-source-database",
            "source.duckdb",
            "--selection-input-manifest",
            "manifest.json",
            "--selection-claim",
            "selection.json",
            "--selection-economic-report",
            "economics.json",
        ]
    )

    assert args.memory_limit == "1GB"
    assert args.threads == 2


def test_round29_runner_rejects_aliases_symlinks_and_open_wal(
    tmp_path: Path,
) -> None:
    inputs = {
        name: tmp_path / f"{name}.duckdb"
        for name in ("feature", "overlay", "target", "source")
    }
    for path in inputs.values():
        path.write_bytes(b"fixture")
    outputs = {
        "manifest": tmp_path / "manifest.json",
        "selection": tmp_path / "selection.json",
        "economics": tmp_path / "economics.json",
    }

    TOOL._validate_paths(inputs, outputs)
    with pytest.raises(ValueError, match="paths differ"):
        TOOL._validate_paths(inputs, {**outputs, "economics": inputs["source"]})
    os.link(inputs["feature"], outputs["manifest"])
    with pytest.raises(ValueError, match="paths differ"):
        TOOL._validate_paths(inputs, outputs)
    outputs["manifest"].unlink()
    Path(f"{inputs['source']}.wal").write_bytes(b"open")
    with pytest.raises(ValueError, match="paths differ"):
        TOOL._validate_paths(inputs, outputs)
