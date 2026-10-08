from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path, PurePosixPath
from unittest import mock

import pytest

from tools import cocoindex_agent_search as adapter


@pytest.mark.parametrize(
    "platform,tail", [("nt", "Scripts/python.exe"), ("posix", "bin/python")]
)
def test_venv_interpreter_uses_platform_layout(
    monkeypatch, tmp_path: Path, platform, tail
):
    context = adapter.CocoIndexContext(
        tmp_path, tmp_path / "cache", tmp_path / "mirror", "a" * 32
    )
    monkeypatch.setattr(adapter.os, "name", platform)
    assert context.venv_python == context.venv_dir / tail
    with mock.patch.object(adapter, "checked_command") as command:
        adapter.verify_install(context)
    assert command.call_args_list[0].args[0][0] == str(context.venv_python)


@pytest.mark.parametrize(
    "path",
    [
        "data/protected/partial.json",
        "artifacts/private.md",
        "docs/review/2026-10-08/raw.json",
        "docs/model-research/raw/book.py",
        "docs/archive/old.md",
        "native/icon.png",
        "dump.zip",
        "docs/reference/cocoindex-code-agent-benchmark-2026-07-11-cases.json",
    ],
)
def test_index_scope_excludes_evidence_and_binary_inputs_before_access(path):
    assert not adapter.is_source_path(PurePosixPath(path))


@pytest.mark.parametrize(
    "path",
    [
        "src/strategy.py",
        "tools/capture.py",
        "tests/test_risk.py",
        "native/src/main.cpp",
        "AGENTS.md",
        "docs/CAPITAL_PROTECTION_ARCHITECTURE.md",
        ".github/workflows/ci.yml",
    ],
)
def test_index_scope_includes_code_and_active_instruction_sources(path):
    assert adapter.is_source_path(PurePosixPath(path))


def test_git_scope_filters_paths_without_reading_excluded_contents(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        adapter,
        "checked_git_command",
        lambda *args, **kwargs: mock.Mock(
            stdout="src/main.py\0data/partial.json\0docs/review/raw.json\0"
        ),
    )
    assert adapter.tracked_files(tmp_path) == [PurePosixPath("src/main.py")]


def test_embedding_environment_is_local_and_resource_bounded(tmp_path: Path):
    original_environment = os.environ.copy()
    context = adapter.CocoIndexContext(
        tmp_path, tmp_path / "cache", tmp_path / "mirror", "a" * 32
    )
    environment = adapter.ccc_env(context)
    assert environment["COCOINDEX_DISABLE_USAGE_TRACKING"] == "1"
    assert environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert environment["OMP_NUM_THREADS"] == environment["MKL_NUM_THREADS"] == "2"
    assert environment["TOKENIZERS_PARALLELISM"] == "false"
    assert os.environ == original_environment


@pytest.mark.parametrize("platform", ["nt", "posix"])
def test_cli_preserves_literal_globs_and_query_arguments(
    monkeypatch, tmp_path, platform
):
    context = adapter.CocoIndexContext(
        tmp_path, tmp_path / "cache", tmp_path / "mirror", "a" * 32
    )
    arguments = ["search", "--path", "src/simple_ai_trading/**", "cost * quantity"]
    monkeypatch.setattr(adapter.os, "name", platform)
    command = adapter.ccc_command(context, arguments)
    assert command[-len(arguments) :] == arguments
    if platform == "nt":
        assert command[:2] == [str(context.venv_python), "-c"]
        assert "app(windows_expand_args=False)" in command[2]
    else:
        assert command == [str(context.ccc_bin), *arguments]


def test_automated_commands_never_inherit_interactive_stdin(tmp_path: Path):
    with mock.patch.object(adapter, "run_captured_process") as command:
        adapter.run_command(["ccc", "init", "--force"], cwd=tmp_path)
    assert command.call_args.kwargs["input_text"] == ""
    assert "stdin" not in command.call_args.kwargs


def test_automated_child_observes_noninteractive_stdin(tmp_path: Path):
    import sys

    result = adapter.run_command(
        [
            sys.executable,
            "-c",
            "import sys; print(sys.stdin.isatty()); print(chr(955))",
        ],
        cwd=tmp_path,
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == ["False", "λ"]


def test_benchmark_baseline_reads_only_the_safe_mirror(tmp_path: Path):
    context = adapter.CocoIndexContext(
        tmp_path, tmp_path / "cache", tmp_path / "mirror", "a" * 32
    )
    case = adapter.BenchmarkCase("scope", "cash accounting", "cash", [])
    with mock.patch.object(adapter, "run_command") as command:
        command.return_value.stdout = ""
        adapter.run_rg_baseline(context, "rg", case, [])
    assert command.call_args.kwargs["cwd"] == context.mirror_repo


def receipt_context(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    mirror = tmp_path / "cache" / "mirror"
    for root in (repo, mirror):
        (root / "src").mkdir(parents=True)
        (root / "src" / "cash.py").write_text("cash = 1\n", encoding="utf-8")
    paths = [PurePosixPath("src/cash.py")]
    monkeypatch.setattr(adapter, "tracked_files", lambda _root: paths)
    digest = adapter.file_digest(repo, paths)
    context = adapter.CocoIndexContext(repo, tmp_path / "cache", mirror, digest)
    (mirror.parent / "manifest.json").write_text(
        json.dumps({"digest": digest}), encoding="utf-8"
    )
    return context


def test_actual_search_receipt_retains_raw_output_and_matching_source(
    tmp_path, monkeypatch
):
    context = receipt_context(tmp_path, monkeypatch)
    output = "File: src/cash.py:1 hit\ncash = 1\n"
    path = adapter.retain_search_receipt(context, ["search", "cash"], output)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.is_relative_to(context.artifact_root / "receipts")
    assert payload["raw_output"] == output
    assert payload["raw_output_sha256"] == hashlib.sha256(output.encode()).hexdigest()
    assert payload["index_matches_current_source"] is True
    assert payload["candidate_source_checks"][0]["matches"] is True
    assert payload["financial_evidence"] is False
    assert adapter.retain_search_receipt(context, ["search", "cash"], output) != path


def test_stale_snapshot_rejects_before_search_and_receipt_records_mismatch(
    tmp_path, monkeypatch
):
    context = receipt_context(tmp_path, monkeypatch)
    (context.repo_root / "src" / "cash.py").write_text("cash = 2\n", encoding="utf-8")
    with pytest.raises(adapter.IndexRequiredError, match="stale"):
        adapter.require_current_index(context)
    path = adapter.retain_search_receipt(
        context, ["search", "cash"], "File: src/cash.py:1 hit\n"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["index_matches_current_source"] is False
    assert payload["candidate_source_checks"][0]["matches"] is False


@pytest.mark.parametrize("change_during_search", [False, True])
def test_search_retains_receipt_and_rejects_concurrent_source_changes(
    tmp_path, monkeypatch, change_during_search
):
    context = receipt_context(tmp_path, monkeypatch)
    database = adapter.target_sqlite_db(context)
    database.parent.mkdir(parents=True)
    database.touch()
    output = "File: src/cash.py:1 hit\ncash = 1\n"

    def search(*args, **kwargs):
        if change_during_search:
            (context.repo_root / "src" / "cash.py").write_text(
                "cash = 2\n", encoding="utf-8"
            )
        return mock.Mock(stdout=output)

    monkeypatch.setattr(adapter, "run_ccc_existing", search)
    arguments = dict(
        query=["cash"], limit=5, path=None, langs=[], refresh=False, allow_index=False
    )
    if change_during_search:
        with pytest.raises(adapter.IndexRequiredError, match="Retained query receipt"):
            adapter.run_search(context, **arguments)
    else:
        assert adapter.run_search(context, **arguments) == output
    receipts = list((context.artifact_root / "receipts").glob("search-*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert payload["index_matches_current_source"] is (not change_during_search)


@pytest.mark.parametrize(
    "candidate", ["../escape.py", "data/partial.json", "/escape.py"]
)
def test_receipt_rejects_unsafe_candidate_before_read(tmp_path, monkeypatch, candidate):
    context = receipt_context(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="source scope"):
        adapter.retain_search_receipt(
            context, ["search", "cash"], f"File: {candidate}:1 hit\n"
        )
    assert not (context.artifact_root / "receipts").exists()


@pytest.mark.parametrize("binding", [None, "C:/other-repository"])
def test_mcp_install_preserves_unknown_or_foreign_registration(
    tmp_path, monkeypatch, binding
):
    context = receipt_context(tmp_path, monkeypatch)
    monkeypatch.setattr(adapter, "resolve_mcp_handshake_context", lambda: context)
    monkeypatch.setattr(
        adapter,
        "load_codex_config",
        lambda _path: {
            "mcp_servers": {
                adapter.MCP_SERVER_NAME: {"env": {adapter.REPO_ROOT_ENV: binding}}
            }
        },
    )
    with mock.patch.object(adapter, "ensure_mcp_launcher") as launcher:
        with pytest.raises(RuntimeError, match="Refusing to overwrite"):
            adapter.command_mcp_install(mock.Mock())
    launcher.assert_not_called()


def test_daemon_cleanup_uses_owned_job_not_reported_pid(tmp_path, monkeypatch):
    context = receipt_context(tmp_path, monkeypatch)
    job = mock.Mock()
    process = mock.Mock(_cocoindex_owned_job=job)
    process.poll.return_value = 0
    monkeypatch.setattr(
        adapter,
        "daemon_pid",
        mock.Mock(side_effect=AssertionError("PID is not authority")),
    )
    adapter.stop_owned_daemon(context, process)
    job.close.assert_called_once_with()
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


@pytest.mark.skipif(os.name != "nt", reason="Windows job containment requires Win32")
def test_windows_daemon_starts_suspended_and_containment_failure_has_no_fallback(
    tmp_path, monkeypatch
):
    from simple_ai_trading import windows_owned_job

    context = receipt_context(tmp_path, monkeypatch)
    job = mock.Mock()
    job.enroll_and_resume.side_effect = OSError("containment failed")
    process = mock.Mock()
    process.poll.return_value = 0
    with mock.patch.object(windows_owned_job, "WindowsOwnedJob", return_value=job):
        with mock.patch.object(
            adapter.subprocess, "Popen", return_value=process
        ) as launch:
            with pytest.raises(RuntimeError, match="containment failed"):
                adapter.start_daemon_process(context)
    assert launch.call_count == 1
    assert launch.call_args.kwargs["creationflags"] & windows_owned_job.CREATE_SUSPENDED
    assert (
        launch.call_args.kwargs["creationflags"] & adapter.subprocess.CREATE_NO_WINDOW
    )
    job.enroll_and_resume.assert_called_once_with(process)
    job.close.assert_called_once_with()


@pytest.mark.parametrize("stale", [False, True])
def test_benchmark_reuse_requires_fresh_index_and_never_rebuilds(
    tmp_path, monkeypatch, stale
):
    from contextlib import nullcontext

    context = receipt_context(tmp_path, monkeypatch)
    database = adapter.target_sqlite_db(context)
    database.parent.mkdir(parents=True)
    database.touch()
    if stale:
        (context.repo_root / "src" / "cash.py").write_text(
            "cash = 2\n", encoding="utf-8"
        )
    monkeypatch.setattr(adapter, "require_clean_index_target", mock.Mock())
    monkeypatch.setattr(adapter, "require_disk_budget", mock.Mock())
    monkeypatch.setattr(adapter, "resolve_required_executable", lambda _name: "rg")
    monkeypatch.setattr(adapter, "daemon_session", lambda _context: nullcontext())
    monkeypatch.setattr(adapter, "benchmark_case", mock.Mock())
    monkeypatch.setattr(adapter, "benchmark_summary", lambda _results: {})
    monkeypatch.setattr(
        adapter, "checked_git_command", lambda *args: mock.Mock(stdout="head\n")
    )
    case = adapter.BenchmarkCase("cash", "cash accounting", "cash", ())
    with mock.patch.object(adapter, "run_index") as index:
        if stale:
            with pytest.raises(adapter.IndexRequiredError, match="stale"):
                adapter.run_benchmark(
                    context, [case], None, allow_dirty=True, reuse_current_index=True
                )
        else:
            result = adapter.run_benchmark(
                context, [case], None, allow_dirty=True, reuse_current_index=True
            )
            assert result["reused_current_index"] is True
            assert result["index_elapsed_seconds"] == 0.0
            assert result["timing_qualified"] is False
        index.assert_not_called()


def test_parser_exposes_explicit_benchmark_reuse_without_defaulting_to_it():
    parser = adapter.build_parser()
    assert (
        parser.parse_args(["benchmark", "--cases", "cases.json"]).reuse_current_index
        is False
    )
    assert (
        parser.parse_args(
            ["benchmark", "--cases", "cases.json", "--reuse-current-index"]
        ).reuse_current_index
        is True
    )


@pytest.mark.skipif(
    os.name != "nt", reason="Windows command containment requires Win32"
)
@pytest.mark.parametrize("failure", [None, "timeout", "enrollment"])
def test_captured_command_closes_original_job_on_success_or_failure(tmp_path, failure):
    from simple_ai_trading import windows_owned_job

    job = mock.Mock()
    process = mock.Mock(returncode=0)
    process.communicate.return_value = ("output", "")
    if failure == "timeout":
        process.communicate.side_effect = adapter.subprocess.TimeoutExpired(
            "command", 1
        )
    elif failure == "enrollment":
        job.enroll_and_resume.side_effect = OSError("cannot contain")
    with mock.patch.object(windows_owned_job, "WindowsOwnedJob", return_value=job):
        with mock.patch.object(
            adapter.subprocess, "Popen", return_value=process
        ) as launch:
            if failure is not None:
                with pytest.raises((adapter.subprocess.TimeoutExpired, OSError)):
                    adapter.run_captured_process(
                        ["command"], cwd=tmp_path, env={}, input_text="input", timeout=1
                    )
            else:
                result = adapter.run_captured_process(
                    ["command"], cwd=tmp_path, env={}, input_text="input", timeout=1
                )
                assert result.returncode == 0 and result.stdout == "output"
    assert launch.call_count == 1
    assert launch.call_args.kwargs["creationflags"] & windows_owned_job.CREATE_SUSPENDED
    job.close.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=adapter.timeout_seconds("daemon_stop"))
    for stream in (process.stdin, process.stdout, process.stderr):
        stream.close.assert_called_once_with()
    if failure != "enrollment":
        process.communicate.assert_called_once_with(input="input", timeout=1)


def test_incremental_refresh_preserves_prior_snapshot_and_cache_identity(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "cash.py").write_text("cash = 1\n", encoding="utf-8")
    (repo / "src" / "removed.py").write_text("old = 1\n", encoding="utf-8")
    paths = [PurePosixPath("src/cash.py"), PurePosixPath("src/removed.py")]
    monkeypatch.setattr(adapter, "tracked_files", lambda *args: paths.copy())
    monkeypatch.setattr(adapter, "resolve_repo_root", lambda: repo)
    cache = tmp_path / "cache"
    monkeypatch.setattr(adapter, "default_artifact_root", lambda: cache)
    original_digest = adapter.file_digest(repo, paths)
    original = adapter.CocoIndexContext(
        repo, cache, cache / "mirrors" / original_digest / "repo", original_digest
    )
    adapter.ensure_mirror(original)
    settings = original.mirror_repo / ".cocoindex_code" / "settings.yml"
    settings.parent.mkdir()
    settings.write_text("settings preserved\n", encoding="utf-8")
    database = adapter.target_sqlite_db(original)
    database.parent.mkdir(parents=True)
    database.write_bytes(b"existing cache")
    adapter.write_active_index_metadata(original)
    (repo / "src" / "cash.py").write_text("cash = 2\n", encoding="utf-8")
    (repo / "src" / "removed.py").unlink()
    (repo / "src" / "new.py").write_text("new = 1\n", encoding="utf-8")
    paths.append(PurePosixPath("src/new.py"))
    refreshed = adapter.resolve_reusable_index_context(repo)
    assert refreshed.mirror_digest != original_digest
    assert refreshed.mirror_repo == original.mirror_repo
    assert (
        refreshed.db_dir == original.db_dir
        and refreshed.runtime_dir == original.runtime_dir
    )
    adapter.ensure_mirror(refreshed)
    snapshot = cache / "snapshots" / original_digest / "repo"
    assert (snapshot / "src" / "cash.py").read_text() == "cash = 1\n"
    assert (snapshot / "src" / "removed.py").is_file()
    assert not (snapshot / ".cocoindex_code").exists()
    assert (refreshed.mirror_repo / "src" / "cash.py").read_text() == "cash = 2\n"
    assert not (refreshed.mirror_repo / "src" / "removed.py").exists()
    assert settings.read_text() == "settings preserved\n"
    assert database.read_bytes() == b"existing cache"
    assert adapter.resolve_active_index_context().mirror_digest == original_digest
    adapter.write_active_index_metadata(refreshed)
    assert adapter.resolve_active_index_context() == refreshed
    adapter.require_current_index(refreshed)
    paths.remove(PurePosixPath("src/removed.py"))
    later = adapter.resolve_reusable_index_context(repo)
    adapter.ensure_mirror(later)
    adapter.write_active_index_metadata(later)
    adapter.require_current_index(later)
    # Returning to the original source fingerprint still refreshes storage;
    # a storage key equal to that fingerprint must not mean "already current".
    (repo / "src" / "cash.py").write_text("cash = 1\n", encoding="utf-8")
    (repo / "src" / "removed.py").write_text("old = 1\n", encoding="utf-8")
    (repo / "src" / "new.py").unlink()
    paths[:] = [PurePosixPath("src/cash.py"), PurePosixPath("src/removed.py")]
    reverted = adapter.resolve_reusable_index_context(repo)
    assert reverted.mirror_digest == original_digest
    adapter.ensure_mirror(reverted)
    adapter.write_active_index_metadata(reverted)
    adapter.require_current_index(reverted)


def test_pending_update_or_modified_mirror_blocks_search(tmp_path, monkeypatch):
    context = receipt_context(tmp_path, monkeypatch)
    marker = context.db_dir / "index-update-pending.json"
    marker.parent.mkdir(parents=True)
    marker.touch()
    with pytest.raises(adapter.IndexRequiredError, match="pending or failed"):
        adapter.require_current_index(context)
    marker.unlink()
    (context.mirror_repo / "src" / "cash.py").write_text(
        "cash = 99\n", encoding="utf-8"
    )
    with pytest.raises(adapter.IndexRequiredError, match="stale or modified"):
        adapter.require_current_index(context)


def test_python_routing_control_keeps_frozen_questions_and_expectations():
    repo = Path(__file__).resolve().parents[1]
    original = adapter.load_benchmark_cases(repo / adapter.SEMANTIC_QUERY_FIXTURE)
    controls = adapter.load_benchmark_cases(
        repo / "docs/review/2026-10-08/cocoindex-python-cases.json"
    )
    assert len(original) == len(controls) == 10
    for before, control in zip(original, controls, strict=True):
        assert (before.name, before.query, before.rg, before.expected) == (
            control.name,
            control.query,
            control.rg,
            control.expected,
        )
        assert before.langs == () and control.langs == ("python",)
        assert adapter.benchmark_search_arguments(control) == [
            "search",
            "--limit",
            "5",
            "--lang",
            "python",
            control.query,
        ]


def test_source_routing_control_keeps_frozen_questions_and_expectations():
    repo = Path(__file__).resolve().parents[1]
    original = adapter.load_benchmark_cases(repo / adapter.SEMANTIC_QUERY_FIXTURE)
    controls = adapter.load_benchmark_cases(
        repo / "docs/review/2026-10-08/cocoindex-source-cases.json"
    )
    assert len(original) == len(controls) == 10
    for frozen, control in zip(original, controls, strict=True):
        assert (frozen.name, frozen.query, frozen.rg, frozen.expected) == (
            control.name,
            control.query,
            control.rg,
            control.expected,
        )
        assert adapter.benchmark_search_arguments(control) == [
            "search",
            "--limit",
            "5",
            "--path",
            "src/simple_ai_trading/**",
            control.query,
        ]


@pytest.mark.parametrize(
    "field,value", [("path", ""), ("path", 1), ("langs", "python"), ("langs", [1])]
)
def test_benchmark_rejects_malformed_optional_filters(tmp_path, field, value):
    fixture = tmp_path / "cases.json"
    fixture.write_text(
        json.dumps(
            [
                {
                    "name": "scope",
                    "query": "code",
                    "rg": "code",
                    "expected": ["src/code.py"],
                    field: value,
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match=field):
        adapter.load_benchmark_cases(fixture)
