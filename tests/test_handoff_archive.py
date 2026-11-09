from hashlib import sha256
import json
import subprocess

import pytest

from tools import read_handoff_archive as reader


def manifest_for(raw):
    item = {
        "source_revision": "1" * 40,
        "source_path": "docs/AGENT_START.md",
        "git_blob_sha1": "2" * 40,
        "sha256": sha256(raw).hexdigest(),
        "bytes": len(raw),
        "lines": raw.count(b"\n") + int(not raw.endswith(b"\n")),
    }
    body = {"schema": "immutable-handoff-archive-v1", "archives": {"agent-start": item}}
    sign(body)
    return body


def sign(value):
    body = {k: v for k, v in value.items() if k != "result_sha256"}
    value["result_sha256"] = sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def install(tmp_path, monkeypatch, raw=b"alpha\nbeta\ngamma\n"):
    manifest = manifest_for(raw)
    path = tmp_path / "archive.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    calls = []

    def git(*args):
        calls.append(args)
        if args[0] == "rev-parse":
            return b"2" * 40 + b"\n"
        if args[1] == "-s":
            return str(len(raw)).encode()
        if args[:2] == ("cat-file", "blob"):
            return raw
        raise AssertionError("unexpected Git operation")

    monkeypatch.setattr(reader, "MANIFEST", path)
    monkeypatch.setattr(reader, "_git", git)
    return manifest, path, calls


def test_verified_archive_window_uses_recorded_revision_and_only_local_reads(
    tmp_path, monkeypatch
):
    _, _, calls = install(tmp_path, monkeypatch)
    result = reader.archive_window("agent-start", first_line=2, count=1)
    assert result["lines"] == ["beta"]
    assert result["first_line"] == 2
    assert calls == [
        ("rev-parse", "--verify", "1" * 40 + ":docs/AGENT_START.md"),
        ("cat-file", "-s", "2" * 40),
        ("cat-file", "blob", "2" * 40),
    ]


def test_find_reports_only_bounded_locations_and_exact_count(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch, raw=b"match\n" * 30)
    result = reader.archive_find("agent-start", literal="match")
    assert result["matching_lines"] == list(range(1, 21))
    assert result["match_count"] == 30 and result["locations_truncated"]
    assert "lines" not in result


@pytest.mark.parametrize("changes", [(0, 1), (1, 121), (True, 1), (1, True)])
def test_invalid_window_stops_before_read(monkeypatch, changes):
    monkeypatch.setattr(
        reader, "load_archive", lambda *_: pytest.fail("unexpected source read")
    )
    with pytest.raises(ValueError, match="1..120"):
        reader.archive_window("agent-start", first_line=changes[0], count=changes[1])


def test_changed_source_is_rejected_before_any_text_is_emitted(
    tmp_path, monkeypatch, capsys
):
    manifest, path, _ = install(tmp_path, monkeypatch)
    manifest["archives"]["agent-start"]["sha256"] = "0" * 64
    sign(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert reader.main(["agent-start", "--line", "1"]) == 2
    output = capsys.readouterr()
    assert not output.out and "digest differs" in output.err
    assert "alpha" not in output.err and str(tmp_path) not in output.err


def test_output_byte_limit_fails_instead_of_truncating_source(
    tmp_path, monkeypatch, capsys
):
    install(tmp_path, monkeypatch, raw=b"x" * (reader.MAX_OUTPUT_BYTES + 1))
    assert reader.main(["agent-start", "--line", "1", "--count", "1"]) == 2
    output = capsys.readouterr()
    assert not output.out and "output byte limit" in output.err


def test_missing_git_object_does_not_fetch_or_substitute_text(
    tmp_path, monkeypatch, capsys
):
    install(tmp_path, monkeypatch)

    def missing(*args):
        raise ValueError(
            "Archive Git object unavailable; explicitly fetch its recorded source revision in a shallow clone"
        )

    monkeypatch.setattr(reader, "_git", missing)
    assert reader.main(["agent-start", "--line", "1"]) == 2
    output = capsys.readouterr()
    assert not output.out and "recorded source revision" in output.err


def test_malformed_manifest_does_not_print_a_traceback_or_source(
    tmp_path, monkeypatch, capsys
):
    _, path, _ = install(tmp_path, monkeypatch)
    path.write_text("[]", encoding="utf-8")
    assert reader.main(["agent-start", "--line", "1"]) == 2
    output = capsys.readouterr()
    assert not output.out and "must be an object" in output.err


def test_changed_manifest_stops_before_git(tmp_path, monkeypatch):
    manifest, path, calls = install(tmp_path, monkeypatch)
    manifest["archives"]["agent-start"]["lines"] = 99
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest digest"):
        reader.load_archive("agent-start")
    assert calls == []


def test_unicode_source_round_trips_without_a_host_code_page_assumption(
    tmp_path, monkeypatch, capsys
):
    install(tmp_path, monkeypatch, raw="delta \u0394 and \u2212\n".encode("utf-8"))
    assert reader.main(["agent-start", "--line", "1", "--count", "1"]) == 0
    output = capsys.readouterr().out
    assert output.isascii()
    assert json.loads(output)["lines"] == ["delta \u0394 and \u2212"]


def test_current_entry_points_stay_compact_and_preserve_archive_routes():
    start = (reader.ROOT / "docs/AGENT_START.md").read_bytes()
    continuation = (reader.ROOT / "docs/CONTINUATION.md").read_bytes()
    assert len(start) + len(continuation) <= 16_384
    assert len(start.splitlines()) <= 80
    for raw in (start, continuation):
        assert b"archive/agent-history/README.md" in raw
        assert b"AGENT_WORKFLOWS.md#progress-blocking-and-resumption" in raw
    assert b"RESEARCH_CAPTURE_BOUNDARIES.md" in start
    assert b"structural-edge-priority-registry-v1.json" in start


def test_git_plumbing_disables_lazy_network_fetch_and_replacement_refs(monkeypatch):
    calls = []
    monkeypatch.setattr(reader.shutil, "which", lambda name: "git")

    def run(command, **options):
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0, "verified", "")

    monkeypatch.setattr(reader, "run_captured_process", run)
    assert reader._git("cat-file", "blob", "2" * 40) == b"verified"
    command, options = calls[0]
    assert command == [
        "git",
        "--no-lazy-fetch",
        "--no-replace-objects",
        "cat-file",
        "blob",
        "2" * 40,
    ]
    assert options["timeout"] == 10 and options["input_text"] == ""
    assert options["inherit_environment"] is False
