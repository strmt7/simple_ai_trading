"""Contract tests for the host-side CocoIndex Code agent workflow."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest import mock

import pytest

from tools import cocoindex_agent_search


def test_package_pin_and_hashes_are_exact() -> None:
    """Verify package pin and hashes are exact.

    Inputs: repository fixtures. Output: fails on regressions in package pin and hashes are exact.
    """
    assert cocoindex_agent_search.PACKAGE_REQUIREMENT == (
        "cocoindex-code[full]==0.2.37"
    )
    assert "latest" not in cocoindex_agent_search.PACKAGE_REQUIREMENT


def test_benchmark_doc_records_package_hash_evidence() -> None:
    """Verify benchmark doc records package hash evidence.

    Inputs: repository fixtures. Output: fails on regressions in benchmark doc records package hash evidence.
    """
    repo_root = Path(__file__).resolve().parents[1]
    text = (
        repo_root / "docs/reference/cocoindex-code-agent-benchmark-2026-07-11.md"
    ).read_text(encoding="utf-8")

    assert "Wheel SHA256:" in text
    assert "Source SHA256:" in text


def test_default_artifact_root_uses_xdg_not_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify default artifact root uses xdg not repo.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in default artifact root uses xdg not repo.
    """
    monkeypatch.delenv(cocoindex_agent_search.ARTIFACT_ROOT_ENV, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))

    root = cocoindex_agent_search.default_artifact_root()

    assert root == (tmp_path / "xdg-data" / "agent-cocoindex-code").resolve()


def test_timeout_env_override_is_positive_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify timeout env override is positive integer.

    Inputs: pytest provides `monkeypatch`. Output: fails on regressions in timeout env override is positive integer.
    """
    monkeypatch.setenv("AGENT_COCOINDEX_TIMEOUT_INDEX", "28800")

    assert cocoindex_agent_search.timeout_seconds("index") == 28800


@pytest.mark.parametrize("raw_value", ("0", "-1", "slow"))
def test_timeout_env_override_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, raw_value: str
) -> None:
    """Confirm timeout env override rejects invalid values is rejected at the boundary.

    Inputs: pytest provides `monkeypatch`, `raw_value`. Output: fails on regressions in timeout env override rejects invalid values.
    """
    monkeypatch.setenv("AGENT_COCOINDEX_TIMEOUT_SEARCH", raw_value)

    with pytest.raises(RuntimeError, match="positive integer"):
        cocoindex_agent_search.timeout_seconds("search")


def test_discover_git_root_candidate_walks_from_nested_path(tmp_path: Path) -> None:
    """Verify the discover git root candidate walks from nested path safety boundary.

    Inputs: pytest provides `tmp_path`. Output: fails on regressions when discover git root candidate walks from nested path accepts unsafe input.
    """
    repo = tmp_path / "repo"
    nested = repo / "src" / "package"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()

    assert cocoindex_agent_search.discover_git_root_candidate(nested) == repo.resolve()


def test_resolve_repo_root_uses_command_scoped_safe_directory_from_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify the resolve repo root uses command scoped safe directory from cwd execution contract.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in resolve repo root uses command scoped safe directory from cwd integration.
    """
    repo = (tmp_path / "repo").resolve()
    nested = repo / "docs"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()

    def fake_checked_command(
        args: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Simulate checked command so the surrounding test controls that dependency.

        Inputs: `args` (list[str]) positional arguments, `cwd` (Path) working directory,
        `env` (dict[str, str] | None) environment mapping, `timeout` (int | None)
        timeout seconds. Output: `subprocess.CompletedProcess[str]`.
        """
        assert env is None
        assert timeout is None
        assert cwd == nested
        assert args[:3] == [
            "/usr/bin/git",
            "-c",
            f"safe.directory={repo}",
        ]
        assert args[3:] == ["rev-parse", "--show-toplevel"]
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=f"{repo}\n")

    monkeypatch.chdir(nested)
    monkeypatch.delenv(cocoindex_agent_search.REPO_ROOT_ENV, raising=False)
    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_required_executable",
        mock.Mock(return_value="/usr/bin/git"),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "checked_command",
        fake_checked_command,
    )

    assert cocoindex_agent_search.resolve_repo_root() == repo


def test_resolve_repo_root_rejects_env_override_that_is_not_repo_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Confirm resolve repo root rejects env override that is not repo root is rejected at the boundary.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in resolve repo root rejects env override that is not repo root.
    """
    repo = (tmp_path / "repo").resolve()
    nested = repo / "docs"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()

    monkeypatch.setenv(cocoindex_agent_search.REPO_ROOT_ENV, str(nested))
    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_required_executable",
        mock.Mock(return_value="/usr/bin/git"),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "checked_command",
        mock.Mock(
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout=f"{repo}\n"
            )
        ),
    )

    with pytest.raises(RuntimeError, match="must point at the Git repository root"):
        cocoindex_agent_search.resolve_repo_root()


def test_tracked_files_uses_command_scoped_safe_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify the tracked files uses command scoped safe directory execution contract.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in tracked files uses command scoped safe directory integration.
    """
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()

    def fake_checked_command(
        args: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Simulate checked command so the surrounding test controls that dependency.

        Inputs: `args` (list[str]) positional arguments, `cwd` (Path) working directory,
        `env` (dict[str, str] | None) environment mapping, `timeout` (int | None)
        timeout seconds. Output: `subprocess.CompletedProcess[str]`.
        """
        assert env is None
        assert timeout is None
        assert cwd == repo
        assert args[:3] == [
            "/usr/bin/git",
            "-c",
            f"safe.directory={repo}",
        ]
        assert args[3:] == [
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ]
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="tools/cocoindex_agent_search.py\0",
        )

    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_required_executable",
        mock.Mock(return_value="/usr/bin/git"),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "checked_command",
        fake_checked_command,
    )

    assert cocoindex_agent_search.tracked_files(repo) == [
        PurePosixPath("tools/cocoindex_agent_search.py")
    ]


@pytest.mark.parametrize(
    "raw_path",
    ("", "/abs", "../escape", "dir/../escape", "dir//file", "dir\\file", "bad\nfile"),
)
def test_validate_repo_relative_path_rejects_unsafe_paths(raw_path: str) -> None:
    """Confirm validate repo relative path rejects unsafe paths is rejected at the boundary.

    Inputs: pytest provides `raw_path`. Output: fails on regressions when validate repo relative path rejects unsafe paths accepts unsafe input.
    """
    with pytest.raises(RuntimeError):
        cocoindex_agent_search.validate_repo_relative_path(raw_path)


def test_validate_repo_relative_path_accepts_clean_posix_paths() -> None:
    """Verify the validate repo relative path accepts clean posix paths safety boundary.

    Inputs: repository fixtures. Output: fails on regressions when validate repo relative path accepts clean posix paths accepts unsafe input.
    """
    assert cocoindex_agent_search.validate_repo_relative_path(
        "tools/cocoindex_agent_search.py"
    ) == PurePosixPath("tools/cocoindex_agent_search.py")


@pytest.mark.parametrize(
    "raw_path",
    (".env", "local.env", "env/production.env", ".cocoindex_code/settings.yml"),
)
def test_is_denied_mirror_path_blocks_runtime_artifacts(raw_path: str) -> None:
    """Confirm is denied mirror path blocks runtime artifacts is rejected at the boundary.

    Inputs: pytest provides `raw_path`. Output: fails on regressions when is denied mirror path blocks runtime artifacts accepts unsafe input.
    """
    assert cocoindex_agent_search.is_denied_mirror_path(PurePosixPath(raw_path))


def test_is_denied_mirror_path_allows_example_contracts() -> None:
    """Confirm is denied mirror path allows example contracts is rejected at the boundary.

    Inputs: repository fixtures. Output: fails on regressions when is denied mirror path allows example contracts accepts unsafe input.
    """
    assert not cocoindex_agent_search.is_denied_mirror_path(
        PurePosixPath("env/service_example.env")
    )
    assert not cocoindex_agent_search.is_denied_mirror_path(
        PurePosixPath("env/service.example.env")
    )


def test_load_benchmark_cases_validates_required_schema(tmp_path: Path) -> None:
    """Verify load benchmark cases validates required schema.

    Inputs: pytest provides `tmp_path`. Output: fails on regressions in load benchmark cases validates required schema.
    """
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        '[{"name": "case", "query": "semantic query", "rg": "pattern", '
        '"expected": ["path.py"]}]\n',
        encoding="utf-8",
    )

    assert cocoindex_agent_search.load_benchmark_cases(cases_path) == [
        cocoindex_agent_search.BenchmarkCase(
            name="case",
            query="semantic query",
            rg="pattern",
            expected=("path.py",),
        )
    ]


def test_load_benchmark_cases_rejects_missing_fields(tmp_path: Path) -> None:
    """Confirm load benchmark cases rejects missing fields is rejected at the boundary.

    Inputs: pytest provides `tmp_path`. Output: fails on regressions in load benchmark cases rejects missing fields.
    """
    cases_path = tmp_path / "cases.json"
    cases_path.write_text('[{"name": "case"}]\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing fields"):
        cocoindex_agent_search.load_benchmark_cases(cases_path)


def test_file_digest_refuses_tracked_symlink(tmp_path: Path) -> None:
    """Verify the file digest refuses tracked symlink safety boundary.

    Inputs: pytest provides `tmp_path`. Output: fails on regressions when file digest refuses tracked symlink accepts unsafe input.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.txt").write_text("payload", encoding="utf-8")
    (repo / "link.txt").symlink_to("target.txt")

    with pytest.raises(RuntimeError, match="tracked symlink"):
        cocoindex_agent_search.file_digest_and_mirror_source(
            repo.resolve(), [PurePosixPath("link.txt")]
        )


def test_file_digest_includes_worktree_content(tmp_path: Path) -> None:
    """Verify file digest includes worktree content.

    Inputs: pytest provides `tmp_path`. Output: fails on regressions in file digest includes worktree content.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    path = repo / "tracked.txt"
    path.write_text("first", encoding="utf-8")
    digest_one, files_one = cocoindex_agent_search.file_digest_and_mirror_source(
        repo.resolve(), [PurePosixPath("tracked.txt")]
    )
    path.write_text("second", encoding="utf-8")
    digest_two, files_two = cocoindex_agent_search.file_digest_and_mirror_source(
        repo.resolve(), [PurePosixPath("tracked.txt")]
    )

    assert digest_one != digest_two
    assert files_one["tracked.txt"] == b"first"
    assert files_two["tracked.txt"] == b"second"


def test_file_digest_preserves_package_markers(tmp_path: Path) -> None:
    """Check that file digest preserves package markers remains stable.

    Inputs: pytest provides `tmp_path`. Output: fails on regressions in file digest preserves package markers.
    """
    repo = tmp_path / "repo"
    package = repo / "pkg"
    package.mkdir(parents=True)
    payload = '"""Package marker with useful search context."""\n'
    (package / "__init__.py").write_bytes(payload.encode("utf-8"))

    _digest, files = cocoindex_agent_search.file_digest_and_mirror_source(
        repo.resolve(), [PurePosixPath("pkg/__init__.py")]
    )

    assert files["pkg/__init__.py"] == payload.encode()


def test_repo_relative_path_if_inside_validates_repo_member(tmp_path: Path) -> None:
    """Verify the repo relative path if inside validates repo member safety boundary.

    Inputs: pytest provides `tmp_path`. Output: fails on regressions when repo relative path if inside validates repo member accepts unsafe input.
    """
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    cases_path = repo / "docs" / "cases.json"
    cases_path.parent.mkdir()
    cases_path.write_text("[]", encoding="utf-8")

    assert cocoindex_agent_search.repo_relative_path_if_inside(
        repo, cases_path
    ) == PurePosixPath("docs/cases.json")
    assert (
        cocoindex_agent_search.repo_relative_path_if_inside(
            repo, tmp_path / "outside.json"
        )
        is None
    )


def test_ccc_env_maps_database_and_display_paths_outside_repo(tmp_path: Path) -> None:
    """Verify ccc env maps database and display paths outside repo.

    Inputs: pytest provides `tmp_path`. Output: fails on regressions in ccc env maps database and display paths outside repo.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=(tmp_path / "repo").resolve(),
        artifact_root=(tmp_path / "artifacts").resolve(),
        mirror_repo=(tmp_path / "artifacts" / "mirrors" / "abc" / "repo").resolve(),
        mirror_digest="abc",
    )

    env = cocoindex_agent_search.ccc_env(context)

    assert env["COCOINDEX_CODE_DIR"] == str(context.settings_dir)
    assert env["COCOINDEX_CODE_RUNTIME_DIR"] == str(context.runtime_dir)
    assert env["COCOINDEX_CODE_DB_PATH_MAPPING"] == (
        f"{context.mirror_repo}={context.db_dir}"
    )
    assert "COCOINDEX_CODE_HOST_PATH_MAPPING" not in env


def test_verify_install_executes_console_entrypoint(tmp_path: Path) -> None:
    """Verify verify install executes console entrypoint.

    Inputs: pytest provides `tmp_path`. Output: fails on regressions in verify install executes console entrypoint.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )

    with mock.patch("tools.cocoindex_agent_search.checked_command") as mocked_checked:
        cocoindex_agent_search.verify_install(context)

    assert mocked_checked.call_args_list[-1].args[0] == [str(context.ccc_bin), "--help"]


def test_project_settings_match_generic_mirror_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify project settings match generic mirror policy.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in project settings match generic mirror policy.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    project_settings = SimpleNamespace(
        include_patterns=["**/*.py"],
        exclude_patterns=["**/.*"],
        language_overrides=[],
        chunkers=[],
    )
    settings_module = SimpleNamespace(
        load_project_settings=mock.Mock(return_value=project_settings),
        save_project_settings=mock.Mock(),
    )

    monkeypatch.setattr(
        cocoindex_agent_search, "prepend_venv_site_package_paths", mock.Mock()
    )
    monkeypatch.setattr(
        cocoindex_agent_search.importlib,
        "import_module",
        mock.Mock(return_value=settings_module),
    )

    changed = cocoindex_agent_search.ensure_project_settings_match_mirror(context)

    assert changed
    assert project_settings.include_patterns == list(
        cocoindex_agent_search.MIRROR_INCLUDE_PATTERNS
    )
    assert project_settings.exclude_patterns == list(
        cocoindex_agent_search.MIRROR_EXCLUDE_PATTERNS
    )
    assert project_settings.language_overrides == []
    assert project_settings.chunkers == []
    settings_module.save_project_settings.assert_called_once_with(
        context.mirror_repo, project_settings
    )


def test_project_settings_match_mirror_policy_idempotently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify project settings match mirror policy idempotently.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in project settings match mirror policy idempotently.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    project_settings = SimpleNamespace(
        include_patterns=list(cocoindex_agent_search.MIRROR_INCLUDE_PATTERNS),
        exclude_patterns=list(cocoindex_agent_search.MIRROR_EXCLUDE_PATTERNS),
    )
    settings_module = SimpleNamespace(
        load_project_settings=mock.Mock(return_value=project_settings),
        save_project_settings=mock.Mock(),
    )

    monkeypatch.setattr(
        cocoindex_agent_search, "prepend_venv_site_package_paths", mock.Mock()
    )
    monkeypatch.setattr(
        cocoindex_agent_search.importlib,
        "import_module",
        mock.Mock(return_value=settings_module),
    )

    changed = cocoindex_agent_search.ensure_project_settings_match_mirror(context)

    assert not changed
    settings_module.save_project_settings.assert_not_called()


def test_require_clean_index_target_rejects_dirty_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Confirm require clean index target rejects dirty worktree is rejected at the boundary.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in require clean index target rejects dirty worktree.
    """
    monkeypatch.setattr(
        cocoindex_agent_search,
        "repo_status_porcelain",
        mock.Mock(return_value=" M tools/cocoindex_agent_search.py\n"),
    )

    with pytest.raises(RuntimeError, match="dirty worktree"):
        cocoindex_agent_search.require_clean_index_target(
            tmp_path,
            allow_dirty=False,
        )


def test_repo_status_porcelain_includes_untracked_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify repo status porcelain includes untracked files.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in repo status porcelain includes untracked files.
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    def fake_checked_git_command(
        repo_root: Path,
        args: list[str],
        *,
        cwd: Path | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Simulate checked git command so the surrounding test controls that dependency.

        Inputs: `repo_root` (Path), `args` (list[str]) positional arguments, `cwd` (Path
        | None) working directory, `timeout` (int | None) timeout seconds. Output:
        `subprocess.CompletedProcess[str]`.
        """
        assert repo_root == repo
        assert args == ["status", "--porcelain=v1"]
        assert cwd is None
        assert timeout == cocoindex_agent_search.timeout_seconds("rg")
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="?? generated.txt\n",
            stderr="",
        )

    monkeypatch.setattr(
        cocoindex_agent_search,
        "checked_git_command",
        fake_checked_git_command,
    )

    assert cocoindex_agent_search.repo_status_porcelain(repo) == "?? generated.txt\n"


def test_require_disk_budget_uses_host_scaled_free_space(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify require disk budget uses host scaled free space.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in require disk budget uses host scaled free space.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    usage = SimpleNamespace(
        total=100 * 1024 * 1024 * 1024,
        used=99 * 1024 * 1024 * 1024,
        free=1 * 1024 * 1024 * 1024,
    )
    monkeypatch.setattr(
        cocoindex_agent_search.shutil, "disk_usage", lambda _path: usage
    )

    with pytest.raises(RuntimeError, match="Refusing CocoIndex index"):
        cocoindex_agent_search.require_disk_budget(context, "index")


def test_require_mirror_write_budget_counts_source_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify require mirror write budget counts source bytes.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in require mirror write budget counts source bytes.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    usage = SimpleNamespace(total=100, used=0, free=40)
    monkeypatch.setattr(
        cocoindex_agent_search.shutil,
        "disk_usage",
        lambda _path: usage,
    )
    monkeypatch.setattr(cocoindex_agent_search, "env_bytes", lambda _name, _default: 25)

    with pytest.raises(RuntimeError, match="Refusing CocoIndex mirror"):
        cocoindex_agent_search.require_mirror_write_budget(context, source_bytes=20)


def test_multiple_repositories_share_one_install_but_use_separate_indexes(
    tmp_path: Path,
) -> None:
    """Verify multiple repositories share one install but use separate indexes.

    Inputs: pytest provides `tmp_path`. Output: fails on regressions in multiple repositories share one install but use separate indexes.
    """
    artifact_root = tmp_path / "artifacts"
    repo_one = (tmp_path / "repo-one").resolve()
    repo_two = (tmp_path / "repo-two").resolve()
    context_one = cocoindex_agent_search.CocoIndexContext(
        repo_root=repo_one,
        artifact_root=artifact_root,
        mirror_repo=artifact_root / "mirrors" / "digest-one" / "repo",
        mirror_digest="digest-one",
    )
    context_two = cocoindex_agent_search.CocoIndexContext(
        repo_root=repo_two,
        artifact_root=artifact_root,
        mirror_repo=artifact_root / "mirrors" / "digest-two" / "repo",
        mirror_digest="digest-two",
    )

    assert context_one.venv_dir == context_two.venv_dir
    assert context_one.settings_dir == context_two.settings_dir
    assert context_one.runtime_dir != context_two.runtime_dir
    assert context_one.mirror_repo != context_two.mirror_repo
    assert context_one.db_dir != context_two.db_dir
    assert context_one.venv_dir == artifact_root / "venv" / (
        f"{cocoindex_agent_search.PACKAGE_NAME}-"
        f"{cocoindex_agent_search.PACKAGE_VERSION}"
    )
    assert context_one.mcp_launcher == (
        artifact_root / "bin" / cocoindex_agent_search.MCP_LAUNCHER_NAME
    )
    assert context_one.runtime_dir == artifact_root / "runtime" / "digest-one"
    assert context_two.runtime_dir == artifact_root / "runtime" / "digest-two"
    assert context_one.db_dir == artifact_root / "db" / "digest-one"
    assert context_two.db_dir == artifact_root / "db" / "digest-two"
    assert cocoindex_agent_search.lock_path(
        artifact_root, f"mirror-{context_one.mirror_digest}"
    ) != cocoindex_agent_search.lock_path(
        artifact_root, f"mirror-{context_two.mirror_digest}"
    )
    assert cocoindex_agent_search.lock_path(
        artifact_root, f"init-{context_one.mirror_digest}"
    ) != cocoindex_agent_search.lock_path(
        artifact_root, f"init-{context_two.mirror_digest}"
    )
    assert cocoindex_agent_search.lock_path(
        artifact_root, f"daemon-{context_one.mirror_digest}"
    ) != cocoindex_agent_search.lock_path(
        artifact_root, f"daemon-{context_two.mirror_digest}"
    )
    assert (
        cocoindex_agent_search.ccc_env(context_one)["COCOINDEX_CODE_DB_PATH_MAPPING"]
        != cocoindex_agent_search.ccc_env(context_two)["COCOINDEX_CODE_DB_PATH_MAPPING"]
    )
    assert (
        cocoindex_agent_search.ccc_env(context_one)["COCOINDEX_CODE_RUNTIME_DIR"]
        != cocoindex_agent_search.ccc_env(context_two)["COCOINDEX_CODE_RUNTIME_DIR"]
    )


def test_mcp_config_is_workspace_agnostic_by_default(tmp_path: Path) -> None:
    """Verify MCP config is workspace agnostic by default.

    Inputs: pytest provides `tmp_path`. Output: fails on regressions in MCP config is workspace agnostic by default.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )

    payload = cocoindex_agent_search.mcp_config_payload(context, pin_repo=False)

    assert payload["name"] == "cocoindex-code"
    assert payload["transport"] == "stdio"
    assert payload["command"] == cocoindex_agent_search.MCP_PYTHON_COMMAND
    assert payload["args"] == ["tools/cocoindex_agent_search.py", "mcp"]
    assert payload["startup_timeout_sec"] == 600
    assert payload["startup_timeout_sec"] == (
        cocoindex_agent_search.MCP_STARTUP_TIMEOUT_SECONDS
    )
    assert payload["tool_timeout_sec"] == (
        cocoindex_agent_search.MCP_TOOL_TIMEOUT_SECONDS
    )
    assert payload["env"] == {
        cocoindex_agent_search.ARTIFACT_ROOT_ENV: str(context.artifact_root)
    }
    assert "cwd" not in payload
    assert cocoindex_agent_search.REPO_ROOT_ENV in str(
        payload["working_directory_contract"]
    )
    serialized = str(payload)
    assert str(context.repo_root) not in serialized
    assert str(context.mirror_repo) not in serialized
    assert str(context.db_dir) not in serialized


def test_mcp_config_can_pin_repo_for_static_clients(tmp_path: Path) -> None:
    """Verify MCP config can pin repo for static clients.

    Inputs: pytest provides `tmp_path`. Output: fails on regressions in MCP config can pin repo for static clients.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )

    payload = cocoindex_agent_search.mcp_config_payload(context, pin_repo=True)

    assert payload["command"] == cocoindex_agent_search.MCP_PYTHON_COMMAND
    assert payload["args"] == [
        str(Path(cocoindex_agent_search.__file__).resolve()),
        "mcp",
    ]
    assert payload["env"] == {
        cocoindex_agent_search.ARTIFACT_ROOT_ENV: str(context.artifact_root),
        cocoindex_agent_search.REPO_ROOT_ENV: str(context.repo_root),
    }
    assert "cwd" not in payload


def test_mcp_launcher_is_host_stable_and_environment_driven(tmp_path: Path) -> None:
    """Verify the MCP launcher stays host-stable and environment-driven.

    Inputs: pytest provides `tmp_path`. Output: fails on stale absolute launcher regressions.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )

    launcher = cocoindex_agent_search.ensure_mcp_launcher(context)

    assert launcher == context.mcp_launcher
    launcher_text = launcher.read_text(encoding="utf-8")
    assert cocoindex_agent_search.REPO_ROOT_ENV in launcher_text
    assert '"tools" / "cocoindex_agent_search.py"' in launcher_text
    assert str(context.repo_root) not in launcher_text
    assert str(context.mirror_repo) not in launcher_text
    if sys.platform != "win32":
        assert launcher.stat().st_mode & 0o777 == 0o700


def test_install_command_does_not_hash_worktree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify the install command does not hash worktree execution contract.

    Inputs: pytest provides `monkeypatch`, `tmp_path`, `capsys`. Output: fails on regressions in install command does not hash worktree integration.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "mcp-handshake" / "repo",
        mirror_digest="mcp-handshake",
    )
    install = mock.Mock()
    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_mcp_handshake_context",
        mock.Mock(return_value=context),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_context",
        mock.Mock(side_effect=AssertionError("install must not hash worktree files")),
    )
    monkeypatch.setattr(cocoindex_agent_search, "ensure_installed", install)

    cocoindex_agent_search.command_install(mock.Mock())

    install.assert_called_once_with(context)
    assert str(context.venv_dir) in capsys.readouterr().out


def test_mcp_config_command_does_not_hash_worktree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify the MCP config command does not hash worktree execution contract.

    Inputs: pytest provides `monkeypatch`, `tmp_path`, `capsys`. Output: fails on regressions in MCP config command does not hash worktree integration.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "mcp-handshake" / "repo",
        mirror_digest="mcp-handshake",
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_mcp_handshake_context",
        mock.Mock(return_value=context),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_context",
        mock.Mock(
            side_effect=AssertionError("mcp-config must not hash worktree files")
        ),
    )

    cocoindex_agent_search.command_mcp_config(SimpleNamespace(pin_repo=False))

    payload = json.loads(capsys.readouterr().out)
    assert payload["args"] == ["tools/cocoindex_agent_search.py", "mcp"]


def test_active_index_metadata_resolves_without_hashing_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify active index metadata resolves without hashing worktree.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in active index metadata resolves without hashing worktree.
    """
    repo_root = (tmp_path / "repo").resolve()
    artifact_root = (tmp_path / "artifacts").resolve()
    mirror_digest = "abc12345" * 4
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=repo_root,
        artifact_root=artifact_root,
        mirror_repo=artifact_root / "mirrors" / mirror_digest / "repo",
        mirror_digest=mirror_digest,
    )
    cocoindex_agent_search.write_active_index_metadata(context)
    monkeypatch.setattr(cocoindex_agent_search, "resolve_repo_root", lambda: repo_root)
    monkeypatch.setattr(
        cocoindex_agent_search, "default_artifact_root", lambda: artifact_root
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "file_digest_and_mirror_source",
        mock.Mock(side_effect=AssertionError("active search must not hash files")),
    )

    resolved = cocoindex_agent_search.resolve_active_index_context()

    assert resolved == context


def test_mcp_search_schema_does_not_allow_refresh_index() -> None:
    """Verify MCP search schema does not allow refresh index.

    Inputs: repository fixtures. Output: fails on regressions in MCP search schema does not allow refresh index.
    """
    schema = cocoindex_agent_search.mcp_search_tool_definition()["inputSchema"]

    assert isinstance(schema, dict)
    assert "refresh_index" not in schema["properties"]


def test_mcp_install_does_not_duplicate_existing_codex_server(tmp_path: Path) -> None:
    """Verify MCP install does not duplicate existing codex server.

    Inputs: pytest provides `tmp_path`. Output: fails on regressions in MCP install does not duplicate existing codex server.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    completed = subprocess.CompletedProcess(
        args=["codex", "mcp", "get", "cocoindex-code"],
        returncode=0,
        stdout="configured",
        stderr="",
    )

    with (
        mock.patch(
            "tools.cocoindex_agent_search.resolve_mcp_handshake_context",
            return_value=context,
        ),
        mock.patch(
            "tools.cocoindex_agent_search.resolve_required_executable",
            return_value="codex",
        ),
        mock.patch(
            "tools.cocoindex_agent_search.run_command",
            return_value=completed,
        ) as mocked_run,
        mock.patch(
            "tools.cocoindex_agent_search.load_codex_config",
            return_value={},
        ),
        mock.patch(
            "tools.cocoindex_agent_search.codex_mcp_server_matches_expected",
            return_value=True,
        ),
        mock.patch("tools.cocoindex_agent_search.checked_command") as mocked_checked,
    ):
        cocoindex_agent_search.command_mcp_install(mock.Mock())

    mocked_run.assert_called_once()
    mocked_checked.assert_not_called()


def test_mcp_install_repairs_stale_existing_codex_server(tmp_path: Path) -> None:
    """Verify MCP install repairs stale existing codex server.

    Inputs: pytest provides `tmp_path`. Output: fails on regressions in MCP install repairs stale existing codex server.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    completed = subprocess.CompletedProcess(
        args=["codex", "mcp", "get", "cocoindex-code"],
        returncode=0,
        stdout="configured",
        stderr="",
    )

    with (
        mock.patch(
            "tools.cocoindex_agent_search.resolve_mcp_handshake_context",
            return_value=context,
        ),
        mock.patch(
            "tools.cocoindex_agent_search.resolve_required_executable",
            return_value="codex",
        ),
        mock.patch(
            "tools.cocoindex_agent_search.run_command",
            return_value=completed,
        ),
        mock.patch(
            "tools.cocoindex_agent_search.load_codex_config",
            return_value={"mcp_servers": {}},
        ),
        mock.patch(
            "tools.cocoindex_agent_search.codex_mcp_server_matches_expected",
            return_value=False,
        ),
        mock.patch("tools.cocoindex_agent_search.checked_command") as mocked_checked,
        mock.patch("tools.cocoindex_agent_search.ensure_codex_mcp_timeouts"),
    ):
        cocoindex_agent_search.command_mcp_install(mock.Mock())

    commands = [call.args[0] for call in mocked_checked.call_args_list]
    assert commands[0] == ["codex", "mcp", "remove", "cocoindex-code"]
    assert commands[1][:3] == [
        "codex",
        "mcp",
        "add",
    ]


def test_mcp_install_uses_host_stable_codex_launcher(
    tmp_path: Path,
) -> None:
    """Verify MCP install uses host-stable Codex launcher.

    Inputs: pytest provides `tmp_path`. Output: fails on stale absolute wrapper regressions.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    missing = subprocess.CompletedProcess(
        args=["codex", "mcp", "get", "cocoindex-code"],
        returncode=1,
        stdout="",
        stderr="Error: No MCP server named 'cocoindex-code' found.",
    )

    with (
        mock.patch(
            "tools.cocoindex_agent_search.resolve_mcp_handshake_context",
            return_value=context,
        ),
        mock.patch(
            "tools.cocoindex_agent_search.resolve_required_executable",
            return_value="codex",
        ),
        mock.patch(
            "tools.cocoindex_agent_search.run_command",
            return_value=missing,
        ),
        mock.patch("tools.cocoindex_agent_search.checked_command") as mocked_checked,
        mock.patch("tools.cocoindex_agent_search.ensure_codex_mcp_timeouts"),
    ):
        cocoindex_agent_search.command_mcp_install(mock.Mock())

    command = mocked_checked.call_args.args[0]
    assert (
        f"{cocoindex_agent_search.ARTIFACT_ROOT_ENV}={context.artifact_root}" in command
    )
    assert f"{cocoindex_agent_search.REPO_ROOT_ENV}={context.repo_root}" in command
    assert command[command.index("--") + 1] == (
        cocoindex_agent_search.MCP_PYTHON_COMMAND
    )
    assert command[command.index("--") + 2] == str(context.mcp_launcher)
    assert command[command.index("--") + 3] == "mcp"


def test_codex_mcp_server_matches_expected_requires_pinned_args_env_and_timeouts(
    tmp_path: Path,
) -> None:
    """Verify codex MCP server matches expected requires pinned args env and timeouts.

    Inputs: pytest provides `tmp_path`. Output: fails on regressions in codex MCP server matches expected requires pinned args env and timeouts.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    expected = cocoindex_agent_search.expected_codex_mcp_server(context)
    assert expected["command"] == cocoindex_agent_search.MCP_PYTHON_COMMAND
    assert expected["args"] == [
        str(context.mcp_launcher),
        "mcp",
    ]
    assert expected["env"] == {
        cocoindex_agent_search.ARTIFACT_ROOT_ENV: str(context.artifact_root),
        cocoindex_agent_search.REPO_ROOT_ENV: str(context.repo_root),
    }
    config = {"mcp_servers": {"cocoindex-code": expected.copy()}}

    assert cocoindex_agent_search.codex_mcp_server_matches_expected(config, expected)

    config["mcp_servers"]["cocoindex-code"]["tool_timeout_sec"] = 60
    assert not cocoindex_agent_search.codex_mcp_server_matches_expected(
        config, expected
    )

    config["mcp_servers"]["cocoindex-code"]["tool_timeout_sec"] = expected[
        "tool_timeout_sec"
    ]
    config["mcp_servers"]["cocoindex-code"]["env"] = {
        cocoindex_agent_search.ARTIFACT_ROOT_ENV: str(context.artifact_root)
    }
    assert not cocoindex_agent_search.codex_mcp_server_matches_expected(
        config, expected
    )

    config["mcp_servers"]["cocoindex-code"] = expected.copy()
    config["mcp_servers"]["cocoindex-code"]["cwd"] = "/other/repo"
    assert not cocoindex_agent_search.codex_mcp_server_matches_expected(
        config, expected
    )

    stale_launcher = (
        context.repo_root.parent
        / "removed-clone"
        / "tools"
        / "cocoindex_agent_search.py"
    )
    config["mcp_servers"]["cocoindex-code"] = {
        "command": "python3",
        "args": [str(stale_launcher), "mcp"],
        "env": expected["env"],
        "startup_timeout_sec": expected["startup_timeout_sec"],
        "tool_timeout_sec": expected["tool_timeout_sec"],
    }
    assert not cocoindex_agent_search.codex_mcp_server_matches_expected(
        config, expected
    )


def test_upsert_toml_table_scalars_preserves_env_subtable() -> None:
    """Check that upsert toml table scalars preserves env subtable remains stable.

    Inputs: repository fixtures. Output: fails on regressions in upsert toml table scalars preserves env subtable.
    """
    text = (
        'model = "gpt-5.5"\n'
        "\n"
        "[mcp_servers.cocoindex-code]\n"
        'command = "python3"\n'
        "\n"
        "[mcp_servers.cocoindex-code.env]\n"
        'AGENT_COCOINDEX_HOME = "/tmp/home"\n'
    )

    updated = cocoindex_agent_search.upsert_toml_table_scalars(
        text,
        "mcp_servers.cocoindex-code",
        {"startup_timeout_sec": 7200, "tool_timeout_sec": 14400},
    )

    assert "startup_timeout_sec = 7200\n" in updated
    assert "tool_timeout_sec = 14400\n" in updated
    assert "[mcp_servers.cocoindex-code.env]\n" in updated
    assert 'AGENT_COCOINDEX_HOME = "/tmp/home"\n' in updated


def test_toml_value_formats_string_lists_and_rejects_other_items() -> None:
    assert cocoindex_agent_search.format_toml_value(["launcher.py", "mcp"]) == (
        '["launcher.py", "mcp"]'
    )
    with pytest.raises(TypeError, match="Only string lists"):
        cocoindex_agent_search.format_toml_value([1])  # type: ignore[list-item]


def test_direct_codex_config_preserves_other_tables_and_removes_cwd(
    tmp_path: Path,
) -> None:
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    config_path = tmp_path / "config.toml"
    config_path.write_bytes(
        (
            'model = "gpt-test"\n\n'
            "[mcp_servers.other]\n"
            'command = "other"\n\n'
            "[mcp_servers.cocoindex-code]\n"
            'command = "stale"\n'
            'args = ["removed.py", "mcp"]\n'
            'cwd = "C:/removed"\n\n'
            "[mcp_servers.cocoindex-code.env]\n"
            'AGENT_COCOINDEX_HOME = "C:/stale"\n'
        ).encode("utf-8")
    )
    expected = cocoindex_agent_search.expected_codex_mcp_server(context)

    cocoindex_agent_search.ensure_codex_mcp_config(config_path, expected)

    text = config_path.read_text(encoding="utf-8")
    parsed = cocoindex_agent_search.load_codex_config(config_path)
    assert 'model = "gpt-test"' in text
    assert "[mcp_servers.other]" in text
    assert "cwd =" not in text
    assert cocoindex_agent_search.codex_mcp_server_matches_expected(parsed, expected)


def test_mcp_install_falls_back_to_direct_config_when_codex_cli_is_blocked(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    config_path = tmp_path / "codex" / "config.toml"
    with (
        mock.patch(
            "tools.cocoindex_agent_search.resolve_mcp_handshake_context",
            return_value=context,
        ),
        mock.patch(
            "tools.cocoindex_agent_search.codex_config_path", return_value=config_path
        ),
        mock.patch("tools.cocoindex_agent_search.ensure_mcp_launcher"),
        mock.patch(
            "tools.cocoindex_agent_search.resolve_required_executable",
            side_effect=RuntimeError("Codex executable is unavailable"),
        ),
    ):
        cocoindex_agent_search.command_mcp_install(mock.Mock())

    expected = cocoindex_agent_search.expected_codex_mcp_server(context)
    assert cocoindex_agent_search.codex_mcp_server_matches_expected(
        cocoindex_agent_search.load_codex_config(config_path), expected
    )
    assert "configured directly" in capsys.readouterr().out


def test_repo_benchmark_cases_file_is_valid() -> None:
    """Verify repo benchmark cases file is valid.

    Inputs: repository fixtures. Output: fails on regressions in repo benchmark cases file is valid.
    """
    cases_path = (
        Path(__file__).resolve().parents[1]
        / "docs/reference/cocoindex-code-agent-benchmark-2026-07-11-cases.json"
    )

    cases = cocoindex_agent_search.load_benchmark_cases(cases_path)

    assert len(cases) == 10
    assert all(case.expected for case in cases)


def test_mcp_command_starts_without_preparing_cocoindex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the MCP command starts without preparing cocoindex execution contract.

    Inputs: pytest provides `monkeypatch`. Output: fails on regressions in MCP command starts without preparing cocoindex integration.
    """
    server = mock.Mock()
    monkeypatch.setattr(
        cocoindex_agent_search,
        "run_lightweight_mcp_server",
        server,
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "ensure_ready",
        mock.Mock(side_effect=AssertionError("startup must stay lightweight")),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "daemon_session",
        mock.Mock(side_effect=AssertionError("startup must not launch daemon")),
    )

    cocoindex_agent_search.command_mcp(mock.Mock())

    server.assert_called_once_with()


def test_lightweight_mcp_lists_tools_without_preparing_cocoindex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify lightweight MCP lists tools without preparing cocoindex.

    Inputs: pytest provides `monkeypatch`. Output: fails on regressions in lightweight MCP lists tools without preparing cocoindex.
    """
    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_context",
        mock.Mock(side_effect=AssertionError("startup must not resolve repo")),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "run_search",
        mock.Mock(side_effect=AssertionError("tools/list must not search")),
    )
    input_stream = io.StringIO(
        "\n".join(
            [
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {"protocolVersion": "2025-06-18"},
                    }
                ),
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized",
                        "params": {},
                    }
                ),
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/list",
                        "params": {},
                    }
                ),
            ]
        )
        + "\n"
    )
    output_stream = io.StringIO()

    cocoindex_agent_search.run_lightweight_mcp_server(input_stream, output_stream)

    responses = [
        json.loads(line) for line in output_stream.getvalue().splitlines() if line
    ]
    assert [response["id"] for response in responses] == [1, 2]
    assert responses[0]["result"]["serverInfo"] == {
        "name": "cocoindex-code",
        "version": cocoindex_agent_search.PACKAGE_VERSION,
    }
    assert [tool["name"] for tool in responses[1]["result"]["tools"]] == ["search"]


def test_lightweight_mcp_search_tool_defers_cocoindex_work_until_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify lightweight MCP search tool defers cocoindex work until call.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in lightweight MCP search tool defers cocoindex work until call.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    mocked_search = mock.Mock(return_value="File: AGENTS.md:1 result\n")
    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_active_index_context",
        mock.Mock(return_value=context),
    )
    monkeypatch.setattr(cocoindex_agent_search, "run_search", mocked_search)
    input_stream = io.StringIO(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {
                        "query": "semantic routing",
                        "limit": 3,
                        "path": "docs/**",
                        "lang": ["Markdown"],
                    },
                },
            }
        )
        + "\n"
    )
    output_stream = io.StringIO()

    cocoindex_agent_search.run_lightweight_mcp_server(input_stream, output_stream)

    response = json.loads(output_stream.getvalue())
    assert response["id"] == 1
    assert response["result"] == {
        "content": [{"type": "text", "text": "File: AGENTS.md:1 result\n"}],
        "isError": False,
    }
    mocked_search.assert_called_once_with(
        context,
        query=["semantic routing"],
        limit=3,
        path="docs/**",
        langs=["Markdown"],
        refresh=False,
        allow_index=False,
    )


def test_mcp_search_limit_is_bounded() -> None:
    """Reject semantic result sets large enough to inflate agent context."""
    definition = cocoindex_agent_search.mcp_search_tool_definition()
    limit_schema = definition["inputSchema"]["properties"]["limit"]
    assert limit_schema["default"] == cocoindex_agent_search.DEFAULT_SEARCH_LIMIT
    assert limit_schema["maximum"] == cocoindex_agent_search.MAX_SEARCH_LIMIT

    with pytest.raises(
        cocoindex_agent_search.JsonRpcError,
        match=f"limit must not exceed {cocoindex_agent_search.MAX_SEARCH_LIMIT}",
    ):
        cocoindex_agent_search.mcp_search_arguments(
            {
                "query": "semantic routing",
                "limit": cocoindex_agent_search.MAX_SEARCH_LIMIT + 1,
            }
        )


def test_lightweight_mcp_rejects_refresh_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Confirm lightweight MCP rejects refresh index is rejected at the boundary.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in lightweight MCP rejects refresh index.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    mocked_search = mock.Mock()
    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_context",
        mock.Mock(return_value=context),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_active_index_context",
        mock.Mock(side_effect=cocoindex_agent_search.IndexRequiredError()),
    )
    monkeypatch.setattr(cocoindex_agent_search, "run_search", mocked_search)
    input_stream = io.StringIO(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {
                        "query": "semantic routing",
                        "refresh_index": True,
                    },
                },
            }
        )
        + "\n"
    )
    output_stream = io.StringIO()

    cocoindex_agent_search.run_lightweight_mcp_server(input_stream, output_stream)

    response = json.loads(output_stream.getvalue())
    assert response["id"] == 1
    assert response["error"]["code"] == -32602
    assert "refresh_index is not supported" in response["error"]["message"]
    mocked_search.assert_not_called()


def test_mcp_search_tool_uses_active_index_without_resolving_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify MCP search tool uses active index without resolving content.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in MCP search tool uses active index without resolving content.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    mocked_search = mock.Mock(return_value="File: AGENTS.md:1 result\n")
    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_active_index_context",
        mock.Mock(return_value=context),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_context",
        mock.Mock(side_effect=AssertionError("MCP must not scan worktree content")),
    )
    monkeypatch.setattr(cocoindex_agent_search, "run_search", mocked_search)

    output = cocoindex_agent_search.run_mcp_search_tool(
        {"query": "semantic routing", "limit": 2}
    )

    assert output == "File: AGENTS.md:1 result\n"
    mocked_search.assert_called_once_with(
        context,
        query=["semantic routing"],
        limit=2,
        path=None,
        langs=[],
        refresh=False,
        allow_index=False,
    )


def test_lightweight_mcp_search_errors_redact_local_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify lightweight MCP search errors redact local paths.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in lightweight MCP search errors redact local paths.
    """
    repo_root = (tmp_path / "repo").resolve()
    artifact_root = (tmp_path / "artifacts").resolve()
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=repo_root,
        artifact_root=artifact_root,
        mirror_repo=artifact_root / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_active_index_context",
        mock.Mock(return_value=context),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "run_search",
        mock.Mock(
            side_effect=RuntimeError(
                f"failed under {context.repo_root} and {context.artifact_root}"
            )
        ),
    )
    input_stream = io.StringIO(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {"query": "semantic routing"},
                },
            }
        )
        + "\n"
    )
    output_stream = io.StringIO()

    cocoindex_agent_search.run_lightweight_mcp_server(input_stream, output_stream)

    response = json.loads(output_stream.getvalue())
    text = response["result"]["content"][0]["text"]
    assert response["result"]["isError"] is True
    assert str(tmp_path) not in text
    assert str(repo_root) not in text
    assert str(artifact_root) not in text
    assert cocoindex_agent_search.REPO_ROOT_ENV in text
    assert cocoindex_agent_search.ARTIFACT_ROOT_ENV in text


def test_lightweight_mcp_unknown_tool_does_not_echo_name() -> None:
    """Verify lightweight MCP unknown tool does not echo name.

    Inputs: repository fixtures. Output: fails on regressions in lightweight MCP unknown tool does not echo name.
    """
    sensitive_tool_name = "/private/operator/path"
    input_stream = io.StringIO(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": sensitive_tool_name, "arguments": {}},
            }
        )
        + "\n"
    )
    output_stream = io.StringIO()

    cocoindex_agent_search.run_lightweight_mcp_server(input_stream, output_stream)

    response = json.loads(output_stream.getvalue())
    assert response["error"]["code"] == -32602
    assert response["error"]["message"] == "Unknown MCP tool."
    assert sensitive_tool_name not in response["error"]["message"]


def test_run_ccc_uses_supervised_daemon_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify run ccc uses supervised daemon session.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in run ccc uses supervised daemon session.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    mocked_checked = mock.Mock(return_value=completed)
    session_events: list[str] = []

    class FakeDaemonSession:
        """Test double for the daemon session context manager."""

        def __enter__(self) -> None:
            """Enter the fake daemon session.

            Inputs: no arguments. Output: records the context-manager entry.
            """
            session_events.append("enter")

        def __exit__(self, *_exc: object) -> None:
            """Exit the fake daemon session.

            Inputs: optional exception details. Output: records context exit.
            """
            session_events.append("exit")

    monkeypatch.setattr(cocoindex_agent_search, "ensure_ready", mock.Mock())
    monkeypatch.setattr(
        cocoindex_agent_search,
        "daemon_session",
        mock.Mock(return_value=FakeDaemonSession()),
    )
    monkeypatch.setattr(cocoindex_agent_search, "checked_command", mocked_checked)

    assert cocoindex_agent_search.run_ccc(context, ["status"]) == completed
    assert session_events == ["enter", "exit"]

    kwargs = mocked_checked.call_args.kwargs
    assert kwargs["env"]["COCOINDEX_CODE_DAEMON_SUPERVISED"] == "1"
    assert kwargs["cwd"] == context.mirror_repo


def test_daemon_session_stops_only_daemon_it_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify daemon session stops only daemon it starts.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in daemon ownership cleanup.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    proc = SimpleNamespace(pid=12345)
    events: list[str] = []

    monkeypatch.setattr(
        cocoindex_agent_search,
        "daemon_handshake_succeeds",
        mock.Mock(return_value=False),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "cleanup_stale_daemon_files",
        mock.Mock(side_effect=lambda _context: events.append("cleanup")),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "start_daemon_process",
        mock.Mock(side_effect=lambda _context: events.append("start") or proc),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "wait_for_daemon_handshake",
        mock.Mock(side_effect=lambda _context, _proc: events.append("wait")),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "stop_owned_daemon",
        mock.Mock(side_effect=lambda _context, _proc: events.append("stop")),
    )

    with cocoindex_agent_search.daemon_session(context):
        events.append("body")

    assert events == ["cleanup", "start", "wait", "body", "stop"]
    cocoindex_agent_search.stop_owned_daemon.assert_called_once_with(context, proc)


def test_daemon_session_reuses_preexisting_daemon_without_stopping_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify daemon session reuses preexisting daemon without stopping it.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in daemon reuse cleanup.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    start = mock.Mock()
    stop = mock.Mock()

    monkeypatch.setattr(
        cocoindex_agent_search,
        "daemon_handshake_succeeds",
        mock.Mock(return_value=True),
    )
    monkeypatch.setattr(cocoindex_agent_search, "start_daemon_process", start)
    monkeypatch.setattr(cocoindex_agent_search, "stop_owned_daemon", stop)

    with cocoindex_agent_search.daemon_session(context):
        pass

    start.assert_not_called()
    stop.assert_not_called()


def test_daemon_session_reaps_started_daemon_when_handshake_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify daemon session reaps started daemon when handshake fails.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in failed-start cleanup.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    proc = SimpleNamespace(
        pid=12345,
        poll=mock.Mock(return_value=None),
        terminate=mock.Mock(),
        wait=mock.Mock(return_value=None),
        kill=mock.Mock(),
    )

    monkeypatch.setattr(
        cocoindex_agent_search,
        "daemon_handshake_succeeds",
        mock.Mock(return_value=False),
    )
    monkeypatch.setattr(
        cocoindex_agent_search, "cleanup_stale_daemon_files", mock.Mock()
    )
    monkeypatch.setattr(
        cocoindex_agent_search, "start_daemon_process", mock.Mock(return_value=proc)
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "wait_for_daemon_handshake",
        mock.Mock(side_effect=RuntimeError("handshake failed")),
    )

    with (
        pytest.raises(RuntimeError, match="handshake failed"),
        cocoindex_agent_search.daemon_session(context),
    ):
        pass

    proc.terminate.assert_called_once_with()
    proc.wait.assert_called_once_with(
        timeout=cocoindex_agent_search.timeout_seconds("daemon_stop")
    )
    proc.kill.assert_not_called()


def test_reap_started_daemon_process_escalates_after_timeout() -> None:
    """Verify reap started daemon process escalates after timeout.

    Inputs: no external fixtures. Output: fails on regressions in daemon reap escalation.
    """
    proc = SimpleNamespace(
        poll=mock.Mock(return_value=None),
        terminate=mock.Mock(),
        wait=mock.Mock(
            side_effect=[
                subprocess.TimeoutExpired("daemon", 1),
                subprocess.TimeoutExpired("daemon", 1),
                None,
            ]
        ),
        kill=mock.Mock(),
    )

    cocoindex_agent_search.reap_started_daemon_process(proc)

    assert proc.terminate.call_count == 1
    proc.kill.assert_called_once_with()
    assert proc.wait.call_count == 3


def test_stop_owned_daemon_waits_for_started_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify stop owned daemon waits for started process.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in owned daemon reaping.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    proc = SimpleNamespace(
        pid=12345,
        poll=mock.Mock(return_value=None),
        terminate=mock.Mock(),
        wait=mock.Mock(return_value=None),
        kill=mock.Mock(),
    )
    stop_daemon = mock.Mock()

    monkeypatch.setattr(
        cocoindex_agent_search, "daemon_pid", mock.Mock(return_value=12345)
    )
    monkeypatch.setattr(
        cocoindex_agent_search, "prepend_venv_site_package_paths", mock.Mock()
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "importlib",
        SimpleNamespace(
            import_module=mock.Mock(
                return_value=SimpleNamespace(stop_daemon=stop_daemon)
            )
        ),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "daemon_handshake_succeeds",
        mock.Mock(return_value=False),
    )

    cocoindex_agent_search.stop_owned_daemon(context, proc)

    stop_daemon.assert_called_once_with()
    proc.wait.assert_called_once_with(
        timeout=cocoindex_agent_search.timeout_seconds("daemon_stop")
    )
    proc.terminate.assert_not_called()
    proc.kill.assert_not_called()


def test_run_coco_search_uses_existing_artifact_daemon_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify run coco search uses existing artifact daemon path.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in benchmark search daemon ownership.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    case = cocoindex_agent_search.BenchmarkCase(
        name="daemon",
        query="daemon ownership",
        rg="daemon",
        expected=("tools/cocoindex_agent_search.py",),
    )
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="File: tools/cocoindex_agent_search.py:1 hit\n",
        stderr="",
    )
    mocked_run_ccc_existing = mock.Mock(return_value=completed)
    monkeypatch.setattr(
        cocoindex_agent_search, "run_ccc_existing", mocked_run_ccc_existing
    )

    result, _elapsed_ms, files = cocoindex_agent_search.run_coco_search(context, case)

    assert result == completed
    assert files == ["tools/cocoindex_agent_search.py"]
    mocked_run_ccc_existing.assert_called_once_with(
        context,
        ["search", "--limit", "5", "daemon ownership"],
        timeout=cocoindex_agent_search.timeout_seconds("search"),
        manage_daemon=True,
    )


def test_run_benchmark_reuses_one_daemon_for_cocoindex_cases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify run benchmark reuses one daemon for cocoindex cases.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in benchmark daemon efficiency.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    cases = [
        cocoindex_agent_search.BenchmarkCase(
            name="one", query="one", rg="one", expected=("one.py",)
        ),
        cocoindex_agent_search.BenchmarkCase(
            name="two", query="two", rg="two", expected=("two.py",)
        ),
    ]
    session_events: list[str] = []

    class FakeDaemonSession:
        """Test double for the benchmark daemon session."""

        def __enter__(self) -> None:
            """Enter the fake daemon session.

            Inputs: no arguments. Output: records the context-manager entry.
            """
            session_events.append("enter")

        def __exit__(self, *_exc: object) -> None:
            """Exit the fake daemon session.

            Inputs: optional exception details. Output: records context exit.
            """
            session_events.append("exit")

    def fake_benchmark_case(
        _context: cocoindex_agent_search.CocoIndexContext,
        _rg_bin: str,
        case: cocoindex_agent_search.BenchmarkCase,
        _exclude_args: list[str],
        *,
        manage_daemon: bool,
    ) -> cocoindex_agent_search.BenchmarkResult:
        """Return a deterministic benchmark result.

        Inputs: benchmark context and daemon management flag. Output: fixed result.
        """
        assert manage_daemon is False
        return cocoindex_agent_search.BenchmarkResult(
            case=case.name,
            rg_ms=1.0,
            rg_returncode=0,
            rg_chars=10,
            rg_bytes=10,
            rg_line_count=1,
            rg_unique_files=1,
            rg_first_files=[case.expected[0]],
            rg_expected_rank=1,
            coco_ms=1.0,
            coco_chars=5,
            coco_bytes=5,
            coco_line_count=1,
            coco_unique_files=1,
            coco_first_files=[case.expected[0]],
            coco_expected_rank=1,
            focused_rg_ms=1.0,
            focused_rg_returncode=0,
            focused_rg_chars=3,
            focused_rg_bytes=3,
            focused_rg_line_count=1,
            focused_rg_unique_files=1,
            hybrid_chars=8,
            hybrid_bytes=8,
        )

    context.db_dir.mkdir(parents=True)
    cocoindex_agent_search.target_sqlite_db(context).write_bytes(b"sqlite")
    monkeypatch.setattr(
        cocoindex_agent_search, "require_clean_index_target", mock.Mock()
    )
    monkeypatch.setattr(cocoindex_agent_search, "require_disk_budget", mock.Mock())
    monkeypatch.setattr(cocoindex_agent_search, "ensure_installed", mock.Mock())
    monkeypatch.setattr(cocoindex_agent_search, "ensure_mirror", mock.Mock())
    monkeypatch.setattr(
        cocoindex_agent_search, "ensure_project_initialized", mock.Mock()
    )
    monkeypatch.setattr(cocoindex_agent_search, "run_index", mock.Mock())
    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_required_executable",
        mock.Mock(return_value="rg"),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "checked_git_command",
        mock.Mock(
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="HEAD\n", stderr=""
            )
        ),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "daemon_session",
        mock.Mock(return_value=FakeDaemonSession()),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "benchmark_case",
        mock.Mock(side_effect=fake_benchmark_case),
    )

    payload = cocoindex_agent_search.run_benchmark(
        context, cases, output_path=None, allow_dirty=True
    )

    assert session_events == ["enter", "exit"]
    assert payload["benchmark_schema"] == 2
    assert payload["summary"]["cases"] == 2
    assert payload["summary"]["rg_total_bytes"] == 20
    assert payload["summary"]["hybrid_total_bytes"] == 16
    assert payload["summary"]["hybrid_minus_rg_bytes"] == -4
    assert payload["summary"]["hybrid_to_rg_output_ratio"] == 0.8
    assert cocoindex_agent_search.benchmark_case.call_count == 2


def test_benchmark_case_records_exact_utf8_output_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Measure encoded context volume without pretending it is a token count."""
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    case = cocoindex_agent_search.BenchmarkCase(
        name="unicode", query="risk", rg="risk", expected=("risk.py",)
    )
    rg_stdout = "risk.py:1:caf\u00e9\n"
    coco_stdout = "File: risk.py:1 r\u00e9sum\u00e9\n"
    focused_stdout = "risk.py:1:na\u00efve\n"
    monkeypatch.setattr(
        cocoindex_agent_search,
        "run_rg_baseline",
        mock.Mock(
            return_value=(
                subprocess.CompletedProcess([], 0, rg_stdout, ""),
                1.0,
                ["risk.py"],
            )
        ),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "run_coco_search",
        mock.Mock(
            return_value=(
                subprocess.CompletedProcess([], 0, coco_stdout, ""),
                2.0,
                ["risk.py"],
            )
        ),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "run_focused_rg",
        mock.Mock(
            return_value=(
                subprocess.CompletedProcess([], 0, focused_stdout, ""),
                3.0,
                ["risk.py"],
            )
        ),
    )

    result = cocoindex_agent_search.benchmark_case(
        context, "rg", case, [], manage_daemon=False
    )

    assert result.rg_bytes == len(rg_stdout.encode("utf-8"))
    assert result.coco_bytes == len(coco_stdout.encode("utf-8"))
    assert result.focused_rg_bytes == len(focused_stdout.encode("utf-8"))
    assert result.hybrid_bytes == result.coco_bytes + result.focused_rg_bytes


def test_run_search_rejects_unbounded_context_before_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reject oversized direct calls before indexing or daemon interaction."""
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "run_ccc_existing",
        mock.Mock(side_effect=AssertionError("search side effect must not run")),
    )

    with pytest.raises(ValueError, match="limit must lie"):
        cocoindex_agent_search.run_search(
            context,
            query=["semantic routing"],
            limit=cocoindex_agent_search.MAX_SEARCH_LIMIT + 1,
            path=None,
            langs=[],
            refresh=False,
            allow_index=False,
        )


def test_run_index_preserves_excluded_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Check that run index preserves excluded paths remains stable.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in run index preserves excluded paths.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    excluded_paths = frozenset({PurePosixPath("benchmark_cases.json")})
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="indexed\n",
        stderr="",
    )
    mocked_run_ccc = mock.Mock(return_value=completed)
    monkeypatch.setattr(
        cocoindex_agent_search, "require_clean_index_target", mock.Mock()
    )
    monkeypatch.setattr(cocoindex_agent_search, "require_disk_budget", mock.Mock())
    monkeypatch.setattr(
        cocoindex_agent_search, "emit_cold_index_notice_if_needed", mock.Mock()
    )
    monkeypatch.setattr(
        cocoindex_agent_search, "write_active_index_metadata", mock.Mock()
    )
    monkeypatch.setattr(cocoindex_agent_search, "run_ccc", mocked_run_ccc)

    assert (
        cocoindex_agent_search.run_index(
            context,
            allow_dirty=False,
            excluded_paths=excluded_paths,
        )
        == "indexed\n"
    )

    mocked_run_ccc.assert_called_once_with(
        context,
        ["index"],
        timeout=cocoindex_agent_search.timeout_seconds("index"),
        excluded_paths=excluded_paths,
    )


def test_cleanup_stale_daemon_files_logs_unlink_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Check cleanup stale daemon files logs unlink failures cleanup behavior.

    Inputs: `monkeypatch` (pytest.MonkeyPatch) pytest monkeypatch fixture, `tmp_path`
    (Path) temporary path fixture, `caplog` (pytest.LogCaptureFixture) pytest log
    capture fixture. Output: None. Raises: OSError for the exercised failure path.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    context.runtime_dir.mkdir(parents=True)
    socket_path = context.runtime_dir / "daemon.sock"
    pid_path = context.runtime_dir / "daemon.pid"
    socket_path.write_text("stale", encoding="utf-8")
    pid_path.write_text("stale", encoding="utf-8")
    original_unlink = Path.unlink

    def fake_unlink(self: Path, *, missing_ok: bool = False) -> None:
        """Simulate unlink so the surrounding test controls that dependency.

        Inputs: `missing_ok` (bool). Output: None. Raises: OSError when validation or
        external operations fail.
        """
        if self == socket_path:
            raise OSError("simulated busy socket")
        original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fake_unlink)

    with caplog.at_level("DEBUG", logger=cocoindex_agent_search.__name__):
        cocoindex_agent_search.cleanup_stale_daemon_files(context)

    assert socket_path.exists()
    assert not pid_path.exists()
    assert "Could not remove stale CocoIndex daemon runtime file." in caplog.text


def test_command_index_rejects_dirty_worktree_before_hashing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Confirm command index rejects dirty worktree before hashing is rejected at the boundary.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in command index rejects dirty worktree before hashing integration.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(cocoindex_agent_search, "resolve_repo_root", lambda: repo_root)
    monkeypatch.setattr(
        cocoindex_agent_search,
        "repo_status_porcelain",
        mock.Mock(return_value="?? untracked.txt\n"),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "file_digest",
        mock.Mock(side_effect=AssertionError("dirty rejection must happen first")),
    )

    with pytest.raises(RuntimeError, match="dirty worktree"):
        cocoindex_agent_search.command_index(SimpleNamespace(allow_dirty_index=False))


def test_command_search_refuses_missing_index_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify the command search refuses missing index by default execution contract.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in command search refuses missing index by default integration.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_context",
        mock.Mock(return_value=context),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_active_index_context",
        mock.Mock(side_effect=cocoindex_agent_search.IndexRequiredError()),
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "run_index",
        mock.Mock(side_effect=AssertionError("search must not index by default")),
    )

    with pytest.raises(cocoindex_agent_search.IndexRequiredError):
        cocoindex_agent_search.command_search(
            SimpleNamespace(
                refresh=False,
                index_if_missing=False,
                allow_dirty_index=False,
                limit=5,
                path=None,
                lang=[],
                query=["semantic routing"],
            )
        )


def test_command_search_indexes_only_with_explicit_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify the command search indexes only with explicit flag execution contract.

    Inputs: pytest provides `monkeypatch`, `tmp_path`, `capsys`. Output: fails on regressions in command search indexes only with explicit flag integration.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="File: pkg/runtime.py:1 result\n", stderr=""
    )

    def fake_index(
        _context: cocoindex_agent_search.CocoIndexContext, *, allow_dirty: bool
    ) -> str:
        """Simulate index so the surrounding test controls that dependency.

        Inputs: `_context` (cocoindex_agent_search.CocoIndexContext), `allow_dirty`
        (bool). Output: `str`.
        """
        cocoindex_agent_search.target_sqlite_db(context).parent.mkdir(parents=True)
        cocoindex_agent_search.target_sqlite_db(context).write_bytes(b"sqlite")
        return ""

    mocked_index = mock.Mock(side_effect=fake_index)
    mocked_existing = mock.Mock(return_value=completed)

    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_context",
        mock.Mock(return_value=context),
    )
    monkeypatch.setattr(cocoindex_agent_search, "run_index", mocked_index)
    monkeypatch.setattr(cocoindex_agent_search, "run_ccc_existing", mocked_existing)

    cocoindex_agent_search.command_search(
        SimpleNamespace(
            refresh=False,
            index_if_missing=True,
            allow_dirty_index=True,
            limit=5,
            path=None,
            lang=[],
            query=["semantic routing"],
        )
    )

    captured = capsys.readouterr()
    assert captured.out == completed.stdout
    mocked_index.assert_called_once_with(context, allow_dirty=True)
    assert mocked_existing.call_args.args[1] == [
        "search",
        "--limit",
        "5",
        "semantic routing",
    ]


def test_command_search_skips_cold_notice_when_index_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify the command search skips cold notice when index exists execution contract.

    Inputs: pytest provides `monkeypatch`, `tmp_path`, `capsys`. Output: fails on regressions in command search skips cold notice when index exists integration.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    cocoindex_agent_search.target_sqlite_db(context).parent.mkdir(parents=True)
    cocoindex_agent_search.target_sqlite_db(context).write_bytes(b"sqlite")
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="File: pkg/runtime.py:1 result\n", stderr=""
    )
    mocked_run_ccc = mock.Mock(return_value=completed)

    monkeypatch.setattr(
        cocoindex_agent_search,
        "resolve_active_index_context",
        mock.Mock(return_value=context),
    )
    monkeypatch.setattr(cocoindex_agent_search, "run_ccc_existing", mocked_run_ccc)

    cocoindex_agent_search.command_search(
        SimpleNamespace(
            refresh=False,
            index_if_missing=False,
            allow_dirty_index=False,
            limit=5,
            path=None,
            lang=[],
            query=["semantic routing"],
        )
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    commands = [call.args[1] for call in mocked_run_ccc.call_args_list]
    assert commands == [["search", "--limit", "5", "semantic routing"]]


def test_mcp_smoke_uses_workspace_root_and_minimal_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify MCP smoke uses workspace root and minimal env.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in MCP smoke uses workspace root and minimal env.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            '{"jsonrpc":"2.0","id":1,"result":'
            '{"serverInfo":{"name":"cocoindex-code","version":"0.2.37"}}}\n'
            '{"jsonrpc":"2.0","id":2,"result":'
            '{"tools":[{"name":"search"},{"name":"status"}]}}\n'
        ),
        stderr="",
    )
    mocked_run = mock.Mock(return_value=completed)
    monkeypatch.setattr(
        cocoindex_agent_search,
        "run_command_with_input",
        mocked_run,
    )

    assert cocoindex_agent_search.run_mcp_stdio_smoke(
        context,
        include_search=False,
    ) == {
        "server_name": "cocoindex-code",
        "server_version": "0.2.37",
        "tools": ["search", "status"],
    }
    assert mocked_run.call_args.args[0] == [
        sys.executable,
        str(Path(cocoindex_agent_search.__file__).resolve()),
        "mcp",
    ]
    kwargs = mocked_run.call_args.kwargs
    assert kwargs["cwd"] == context.repo_root
    assert kwargs["env"] == {
        cocoindex_agent_search.ARTIFACT_ROOT_ENV: str(context.artifact_root),
        cocoindex_agent_search.REPO_ROOT_ENV: str(context.repo_root),
    }
    assert '"method": "tools/list"' in kwargs["input_text"]
    assert '"method": "tools/call"' not in kwargs["input_text"]


def test_mcp_stdio_smoke_include_search_fails_on_tool_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify MCP stdio smoke include search fails on tool error.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in MCP stdio smoke include search failure detection.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            '{"jsonrpc":"2.0","id":1,"result":'
            '{"serverInfo":{"name":"cocoindex-code","version":"0.2.37"}}}\n'
            '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"search"}]}}\n'
            '{"jsonrpc":"2.0","id":3,"result":'
            '{"content":[{"type":"text","text":"missing index"}],"isError":true}}\n'
        ),
        stderr="",
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "run_command_with_input",
        mock.Mock(return_value=completed),
    )

    with pytest.raises(RuntimeError, match="MCP search smoke failed"):
        cocoindex_agent_search.run_mcp_stdio_smoke(context, include_search=True)


def test_mcp_stdio_smoke_include_search_records_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify MCP stdio smoke include search records success.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in MCP stdio smoke include search success handling.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            '{"jsonrpc":"2.0","id":1,"result":'
            '{"serverInfo":{"name":"cocoindex-code","version":"0.2.37"}}}\n'
            '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"search"}]}}\n'
            '{"jsonrpc":"2.0","id":3,"result":'
            '{"content":[{"type":"text","text":"File: AGENTS.md:1"}],"isError":false}}\n'
        ),
        stderr="",
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "run_command_with_input",
        mock.Mock(return_value=completed),
    )

    assert (
        cocoindex_agent_search.run_mcp_stdio_smoke(context, include_search=True)[
            "search_tool_content_items"
        ]
        == 1
    )


def test_mcp_jsonrpc_protocol_probe_uses_raw_stdio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify MCP jsonrpc protocol probe uses raw stdio.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in MCP jsonrpc protocol probe uses raw stdio.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            '{"jsonrpc":"2.0","id":1,"result":'
            '{"protocolVersion":"2025-06-18"}}\n'
            '{"jsonrpc":"2.0","id":2,"result":'
            '{"tools":[{"name":"search"},{"name":"status"}]}}\n'
        ),
        stderr="",
    )
    mocked_run = mock.Mock(return_value=completed)
    monkeypatch.setattr(
        cocoindex_agent_search,
        "run_command_with_input",
        mocked_run,
    )

    result = cocoindex_agent_search.run_mcp_jsonrpc_protocol_probe(
        context, "2025-06-18"
    )

    assert result == {
        "protocol_version": "2025-06-18",
        "negotiated_protocol_version": "2025-06-18",
        "tools": ["search", "status"],
    }
    kwargs = mocked_run.call_args.kwargs
    assert kwargs["cwd"] == context.repo_root
    assert kwargs["env"] == {
        cocoindex_agent_search.ARTIFACT_ROOT_ENV: str(context.artifact_root),
        cocoindex_agent_search.REPO_ROOT_ENV: str(context.repo_root),
    }
    assert '"method": "initialize"' in kwargs["input_text"]
    assert '"method": "tools/list"' in kwargs["input_text"]
    assert '"method": "tools/call"' not in kwargs["input_text"]


def test_mcp_jsonrpc_protocol_probe_retries_empty_tool_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify MCP jsonrpc protocol probe retries empty tool list.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in MCP jsonrpc protocol probe retries empty tool list.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    empty_tools = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            '{"jsonrpc":"2.0","id":1,"result":'
            '{"protocolVersion":"2025-06-18"}}\n'
            '{"jsonrpc":"2.0","id":2,"result":{"tools":[]}}\n'
        ),
        stderr="",
    )
    search_tool = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            '{"jsonrpc":"2.0","id":1,"result":'
            '{"protocolVersion":"2025-06-18"}}\n'
            '{"jsonrpc":"2.0","id":2,"result":'
            '{"tools":[{"name":"search"}]}}\n'
        ),
        stderr="",
    )
    mocked_run = mock.Mock(side_effect=[empty_tools, search_tool])
    monkeypatch.setattr(
        cocoindex_agent_search,
        "run_command_with_input",
        mocked_run,
    )
    monkeypatch.setattr(cocoindex_agent_search.time, "sleep", mock.Mock())

    assert cocoindex_agent_search.run_mcp_jsonrpc_protocol_probe(
        context, "2025-06-18"
    ) == {
        "protocol_version": "2025-06-18",
        "negotiated_protocol_version": "2025-06-18",
        "tools": ["search"],
    }
    assert mocked_run.call_count == 2
    cocoindex_agent_search.time.sleep.assert_called_once_with(
        cocoindex_agent_search.MCP_PROTOCOL_PROBE_RETRY_DELAY_SECONDS
    )


def test_mcp_jsonrpc_protocol_probe_rejects_protocol_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Confirm MCP jsonrpc protocol probe rejects protocol mismatch is rejected at the boundary.

    Inputs: pytest provides `monkeypatch`, `tmp_path`. Output: fails on regressions in MCP jsonrpc protocol probe rejects protocol mismatch.
    """
    context = cocoindex_agent_search.CocoIndexContext(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
        mirror_repo=tmp_path / "artifacts" / "mirrors" / "abc" / "repo",
        mirror_digest="abc",
    )
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            '{"jsonrpc":"2.0","id":1,"result":'
            '{"protocolVersion":"1900-01-01"}}\n'
            '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"search"}]}}\n'
        ),
        stderr="",
    )
    monkeypatch.setattr(
        cocoindex_agent_search,
        "run_command_with_input",
        mock.Mock(return_value=completed),
    )

    with pytest.raises(RuntimeError, match="unsupported protocolVersion"):
        cocoindex_agent_search.run_mcp_jsonrpc_protocol_probe(context, "2025-06-18")


def test_cross_agent_surfaces_describe_generic_cocoindex_workflow() -> None:
    """Verify the cross agent surfaces describe generic cocoindex workflow execution contract.

    Inputs: repository fixtures. Output: fails on regressions in cross agent surfaces describe generic cocoindex workflow integration.
    """
    repo_root = Path(__file__).resolve().parents[1]
    canonical_path = "docs/AGENT_WORKFLOWS.md"
    canonical_text = (repo_root / canonical_path).read_text(encoding="utf-8")
    for required in (
        "cocoindex-code-search",
        "MCP",
        "cocoindex-code",
        "mandatory",
        "semantic routing",
        "mcp-smoke",
        "rg",
        "AGENT_COCOINDEX_HOME",
        "cold",
        "external cache",
        "text-decodable",
        "--refresh",
        "--allow-dirty-index",
        "MCP search itself never refreshes",
        "stale active-index text",
        ".cocoindex_code/",
    ):
        assert required in canonical_text, required

    for relative_path in ("AGENTS.md", "README.md"):
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        assert "cocoindex-code-search" in text, relative_path
        assert canonical_path in text, relative_path


def test_no_copied_ccc_skill_trees_are_tracked() -> None:
    """Verify no copied ccc skill trees are tracked.

    Inputs: repository fixtures. Output: fails on regressions in no copied ccc skill trees are tracked.
    """
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            cocoindex_agent_search.resolve_required_executable("git"),
            "ls-files",
            "-z",
            ":(glob)**/skills/ccc/**",
            "skills-lock.json",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    assert completed.stdout == b""
    skill_text = (
        repo_root / ".agents" / "skills" / "cocoindex-code-search" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "npx skills add cocoindex-io/cocoindex-code --all --copy" not in skill_text
    assert "project-local copies" not in skill_text


def test_repo_has_no_stale_omero_specific_cocoindex_install_name() -> None:
    """Verify repo has no stale OMERO specific cocoindex install name.

    Inputs: repository fixtures. Output: fails on regressions in repo has no stale OMERO specific cocoindex install name.
    """
    repo_root = Path(__file__).resolve().parents[1]
    stale_name = "omero" + "-agent-cocoindex"
    completed = subprocess.run(
        [
            cocoindex_agent_search.resolve_required_executable("git"),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    checked_paths = [
        repo_root / raw_path.decode("utf-8")
        for raw_path in completed.stdout.split(b"\0")
        if raw_path
    ]

    for path in checked_paths:
        if not path.exists():
            continue
        if path.suffix in {".pyc", ".png", ".jpg", ".jpeg", ".ico", ".sqlite3"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert stale_name not in text, path


def test_cli_supports_help() -> None:
    """Verify the CLI supports help execution contract.

    Inputs: repository fixtures. Output: fails on regressions in CLI supports help.
    """
    result = subprocess.run(
        [sys.executable, "tools/cocoindex_agent_search.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
