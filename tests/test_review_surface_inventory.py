from pathlib import Path

import pytest

from tools import inventory_review_surface as inventory


def test_parser_does_not_execute_source() -> None:
    result = inventory.inspect_python(
        b"raise RuntimeError('must not run')\n", "safe.py"
    )
    assert result["mechanical_status"] == "ast_parsed_not_semantically_reviewed"
    assert result["lines"] == 1
    assert result["functions"] == 0


@pytest.mark.parametrize("source", [b"def broken(:\n", b"# coding: unknown-encoding\n"])
def test_parser_reports_invalid_input_without_source_text(source: bytes) -> None:
    result = inventory.inspect_python(source, "bad.py")
    assert result["mechanical_status"] == "parse_error"
    assert set(result) == {"mechanical_status", "error_type"}


def test_inventory_keeps_raw_data_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "src" / "ok.py"
    source.parent.mkdir()
    source.write_text("def f():\n    return 1\n", encoding="utf-8")
    # These deliberately nonexistent paths prove contents are not opened.
    paths = [
        "src/ok.py",
        "src/missing.py",
        "data/protected/partial.py",
        "tools/raw/partial.py",
        "docs/plan.md",
    ]
    records = b"".join(f"100644 {'a' * 40} 0\t{p}\0".encode() for p in paths)
    monkeypatch.setattr(inventory, "ROOT", tmp_path)
    monkeypatch.setattr(
        inventory,
        "git",
        lambda *args: records if args[0] == "ls-files" else b"test-head",
    )
    rows, summary = inventory.build_inventory()
    assert summary["tracked_path_count"] == 5
    assert summary["python_functions"] == 1
    assert summary["semantic_review_complete"] is False
    assert rows[0]["working_sha256"]
    assert rows[1]["mechanical_status"] == "missing_worktree_file"
    assert all(
        row["mechanical_status"] == "metadata_only_not_reviewed" for row in rows[2:]
    )
    assert not any("working_sha256" in row for row in rows[2:])


def test_unmerged_index_rejects_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inventory, "git", lambda *args: b"100644 abc 1\tsrc/a.py\0")
    with pytest.raises(ValueError, match="unmerged"):
        inventory.build_inventory()
