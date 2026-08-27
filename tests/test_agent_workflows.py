"""Contracts for imported agent skills and quality workflows."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import tomllib
from unittest import mock

from tools import vulture_check

ROOT = Path(__file__).resolve().parents[1]
OMERO_COMMIT = "246110b1045cfd4ca318b4e870b5a38d213399b6"
KARPATHY_COMMIT = "2c606141936f1eeef17fa3043a72095b4765b9c2"
PUBLISHED_SKILLS = (
    "ai-regression-testing",
    "cocoindex-code-search",
    "docs-knowledge-maintainer",
    "karpathy-guidelines",
    "search-first",
    "source-audit",
)
OMERO_SKILLS = (
    "ai-regression-testing",
    "cocoindex-code-search",
    "context-budget",
    "docker-patterns",
    "docs-knowledge-maintainer",
    "python-patterns",
    "search-first",
    "security-review",
    "source-audit",
    "tdd-workflow",
    "verification-loop",
)


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_imported_skills_have_valid_surfaces_and_provenance() -> None:
    for skill in PUBLISHED_SKILLS:
        root = ROOT / ".agents" / "skills" / skill
        text = (root / "SKILL.md").read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert f"name: {skill}\n" in text
        assert "description:" in text
        assert "metadata:\n  origin:" in text
        openai = (root / "agents" / "openai.yaml").read_text(encoding="utf-8")
        assert "display_name:" in openai
        assert "short_description:" in openai
        assert "allow_implicit_invocation: true" in openai

    assert KARPATHY_COMMIT in _read(".agents/skills/karpathy-guidelines/SKILL.md")
    for skill in OMERO_SKILLS:
        text = _read(f".agents/skills/{skill}/SKILL.md")
        assert text.startswith("---\n")
        assert f"name: {skill}\n" in text
        assert "metadata:\n  origin:" in text
        assert OMERO_COMMIT in text
        assert "ECC v2.0.0 reviewed" in text


def test_agent_workflow_doc_records_exact_tool_versions() -> None:
    text = _read("docs/AGENT_WORKFLOWS.md")
    for expected in (
        OMERO_COMMIT,
        KARPATHY_COMMIT,
        "`2.0.0`",
        "`0.2.37`",
        "`v4.37.8`",
        "`0.15.22`",
        "`2.16`",
        "`v8.7.0`",
    ):
        assert expected in text


def test_agent_verification_skills_enforce_input_aware_efficiency() -> None:
    context = _read(".agents/skills/context-budget/SKILL.md")
    verification = _read(".agents/skills/verification-loop/SKILL.md")
    search = _read(".agents/skills/search-first/SKILL.md")
    assert "verification ledger" in context
    assert "until its code, configuration, fixtures" in context
    assert "final repository-wide matrix" in verification
    assert "Tool Availability Preflight" in search
    assert 'Never report "nothing found"' in search


def test_root_agent_context_is_compact_without_dropping_hard_routes() -> None:
    text = _read("AGENTS.md")
    compact = " ".join(text.split())
    # The hard routes and safety boundaries now occupy about 11 KiB. Keep a
    # finite bloat guard without enforcing the obsolete pre-expansion budget.
    assert len(text.encode("utf-8")) <= 12000
    for required in (
        "docs/AGENT_START.md",
        "docs/AI_COMMIT_IDENTITY.md",
        "AI agent <>\u0060 for author and committer",
        "Work in this session only",
        "No live-money authority exists",
        "Never print, prompt, log, serialize, test, document, or commit credentials",
        "cocoindex-code-search",
        "generated native-contract parity",
    ):
        assert required in compact
    assert "\u00e2" not in text
    assert (ROOT / "docs" / "AI_COMMIT_IDENTITY.md").is_file()


def test_ci_enforces_financial_terminology_audit() -> None:
    workflow = _read(".github/workflows/ci.yml")
    documentation = _read("docs/AGENT_WORKFLOWS.md")

    assert "uv run --locked python tools/audit_financial_terminology.py" in workflow
    assert "tools/audit_financial_terminology.py" in documentation
    assert "diff-cover coverage.xml" in workflow
    assert "--fail-under=95" in workflow
    assert "name: coverage-html" in workflow
    assert "codecov/codecov-action" not in workflow
    assert "CODECOV_TOKEN" not in workflow


def test_push_workflows_cancel_superseded_runs_on_the_same_ref() -> None:
    for relative in (
        ".github/workflows/ci.yml",
        ".github/workflows/codeql.yml",
        ".github/workflows/ruff.yml",
        ".github/workflows/super-linter.yml",
        ".github/workflows/vulture.yml",
    ):
        workflow = _read(relative)
        assert "group: ${{ github.workflow }}-${{ github.ref }}" in workflow
        assert (
            "group: ${{ github.workflow }}-${{ github.ref }}-${{ github.sha }}"
            not in workflow
        )
        assert "cancel-in-progress: true" in workflow


def test_posix_launchers_start_with_bom_free_shebangs() -> None:
    for relative in ("run-shell.sh", "run-gui.sh"):
        payload = (ROOT / relative).read_bytes()
        assert payload.startswith(b"#!/usr/bin/env sh\n")
        assert not payload.startswith(b"\xef\xbb\xbf")


def test_ruff_workflow_is_pinned_and_checks_changed_format_scope() -> None:
    text = _read(".github/workflows/ruff.yml")
    assert "pull_request:" in text
    assert "branches:\n      - main" in text
    assert "workflow_dispatch:" in text
    assert "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0" in text
    assert "astral-sh/ruff-action@278981a28ce3188b1e39527901f38254bf3aac89" in text
    assert "version: 0.15.22" in text
    assert "run: ruff check ." in text
    assert "git diff --name-only --diff-filter=ACMR -z" in text
    assert 'cp -- "${file}" "${formatted}"' in text
    assert "ruff format --config pyproject.toml --" in text
    assert "git diff --no-index --unified=0" in text
    assert 'comm -12 "${changed_lines}" "${format_lines}"' in text


def test_pre_commit_toolchain_matches_reviewed_ruff_release() -> None:
    text = _read(".pre-commit-config.yaml")
    assert "rev: v0.15.22" in text
    assert "- id: ruff-check" in text
    assert "- id: ruff-format" in text
    assert "rev: v6.0.0" in text


def test_vulture_workflow_and_requirements_are_hash_pinned() -> None:
    workflow = _read(".github/workflows/vulture.yml")
    requirements = _read(".github/requirements/vulture-ci.txt")
    assert "pull_request:" in workflow
    assert "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0" in workflow
    assert "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1" in workflow
    assert "python-version: '3.12'" in workflow
    assert "python tools/vulture_check.py" in workflow
    assert "vulture==2.16" in requirements
    assert requirements.count("--hash=sha256:") == 2


def test_super_linter_workflow_is_digest_pinned_and_scoped() -> None:
    text = _read(".github/workflows/super-linter.yml")
    assert "pull_request:" in text
    assert (
        "ghcr.io/super-linter/super-linter:v8.7.0@sha256:"
        "c05768164eed53bac7c82aade7a14a76955206d4962cd41be97118db96fa5996" in text
    )
    for gate in (
        "VALIDATE_GIT_MERGE_CONFLICT_MARKERS: 'true'",
        "VALIDATE_GITHUB_ACTIONS: 'true'",
        "VALIDATE_GITHUB_ACTIONS_ZIZMOR: 'true'",
        "VALIDATE_MARKDOWN: 'true'",
        "VALIDATE_YAML: 'true'",
    ):
        assert gate in text
    assert "third_party" not in text


def test_codeql_scans_python_and_native_windows_with_least_privilege() -> None:
    text = _read(".github/workflows/codeql.yml")
    action = "github/codeql-action/{}@db488ddef3bf6cb639b32c2e9a7c0a7ea8271d28"

    assert "schedule:" in text
    assert "workflow_dispatch:" in text
    assert "security-events: write" in text
    assert "contents: write" not in text
    assert "pull-requests: write" not in text
    assert "- language: python\n            build-mode: none" in text
    assert "- language: c-cpp\n            build-mode: manual" in text
    assert "os: ubuntu-24.04" in text
    assert "os: windows-2025" in text
    assert "fail-fast: false" in text
    assert "queries: security-extended" in text
    assert action.format("init") in text
    assert action.format("analyze") in text
    assert "astral-sh/setup-uv@11f9893b081a58869d3b5fccaea48c9e9e46f990" in text
    assert "version: '0.12.1'" in text
    assert "uv sync --locked --group test" in text
    assert "pip install" not in text
    assert "tools\\build_native_windows.ps1" in text
    assert "persist-credentials: false" in text


def test_all_remote_github_actions_are_commit_pinned() -> None:
    workflow_roots = (
        ROOT / ".github" / "workflows",
        ROOT / ".github" / "actions",
    )
    for root in workflow_roots:
        for path in (*root.rglob("*.yml"), *root.rglob("*.yaml")):
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped.startswith("uses:"):
                    continue
                action = stripped.removeprefix("uses:").strip().split(" #", 1)[0]
                if action.startswith("./"):
                    continue
                assert "@" in action, (path, action)
                reference = action.rsplit("@", 1)[1]
                assert len(reference) == 40, (path, action)
                assert all(
                    character in "0123456789abcdef" for character in reference
                ), (
                    path,
                    action,
                )


def test_full_test_workflows_use_locked_uv_environments() -> None:
    ci = _read(".github/workflows/ci.yml")
    release = _read(".github/actions/windows-beta-release/action.yml")
    setup_uv = "astral-sh/setup-uv@11f9893b081a58869d3b5fccaea48c9e9e46f990"
    assert ci.count(setup_uv) == 2
    assert setup_uv in release
    assert ci.count('version: "0.12.1"') == 2
    assert 'version: "0.12.1"' in release
    assert (
        "uv sync --locked --extra foundation-ai --extra gpu --extra polymarket-live "
        "--extra reporting --group test" in ci
    )
    assert (
        "uv sync --locked --extra foundation-ai --extra gpu --extra polymarket-live "
        "--extra reporting --group test --group release" in release
    )
    assert "pip install" not in ci
    assert "pip install" not in release
    assert '$ErrorActionPreference = "Stop"' in ci
    assert "$PSNativeCommandUseErrorActionPreference = $true" in ci


def test_vulture_scope_keeps_only_production_python() -> None:
    for path in (
        "src/simple_ai_trading/risk_controls.py",
        "tools/vulture_check.py",
    ):
        assert vulture_check.is_vulture_target(PurePosixPath(path))
    for path in (
        "tests/test_risk_controls.py",
        "docs/conf.py",
        ".github/scripts/helper.py",
        "src/simple_ai_trading/tests/test_helper.py",
        "src/simple_ai_trading/conftest.py",
    ):
        assert not vulture_check.is_vulture_target(PurePosixPath(path))


def test_vulture_target_listing_uses_git_visible_files() -> None:
    stdout = "\n".join(
        (
            "tests/test_agent_workflows.py",
            "tools/vulture_check.py",
            "src/simple_ai_trading/risk_controls.py",
            "docs/conf.py",
        )
    )
    with (
        mock.patch(
            "tools.vulture_check.resolve_required_executable", return_value="git"
        ),
        mock.patch(
            "tools.vulture_check.subprocess.run",
            return_value=mock.Mock(stdout=stdout),
        ) as run,
    ):
        assert vulture_check.list_vulture_targets(ROOT) == [
            "tools/vulture_check.py",
            "src/simple_ai_trading/risk_controls.py",
        ]
    command = run.call_args.args[0]
    assert command[-3:] == ["ls-files", "--", "*.py"]
    assert f"safe.directory={ROOT.resolve()}" in command


def test_vulture_command_uses_active_interpreter_and_strict_confidence() -> None:
    assert vulture_check.build_vulture_command(
        ["src/simple_ai_trading/risk_controls.py"], min_confidence=100
    ) == [
        vulture_check.sys.executable,
        "-m",
        "vulture",
        "--min-confidence",
        "100",
        "src/simple_ai_trading/risk_controls.py",
    ]


def test_vulture_scan_falls_back_to_api_on_windows_command_length() -> None:
    scanner = mock.Mock()
    module = mock.Mock()
    module.Vulture.return_value = scanner
    scanner.get_unused_code.return_value = []
    error = OSError("command line is too long")
    error.winerror = 206

    with (
        mock.patch("tools.vulture_check.subprocess.run", side_effect=error),
        mock.patch("tools.vulture_check.importlib.import_module", return_value=module),
    ):
        assert (
            vulture_check.run_vulture(
                ROOT,
                ["src/simple_ai_trading/risk_controls.py"],
                min_confidence=100,
            )
            == 0
        )

    scanner.scavenge.assert_called_once_with(["src/simple_ai_trading/risk_controls.py"])
    scanner.get_unused_code.assert_called_once_with(min_confidence=100)


def test_vulture_scan_ignores_only_exact_protocol_false_positives() -> None:
    expected = {
        (
            "src/simple_ai_trading/round75_continuous_capture.py",
            185,
            "unused variable 'exc_type'",
        ),
        (
            "src/simple_ai_trading/polymarket_round25_resolution_store.py",
            116,
            "unused variable 'allow_redirects'",
        ),
        (
            "src/simple_ai_trading/polymarket_historical_l2.py",
            108,
            "unused variable 'allow_redirects'",
        ),
        (
            "src/simple_ai_trading/polymarket_round22_ingestion.py",
            94,
            "unused variable 'allow_redirects'",
        ),
        (
            "src/simple_ai_trading/polymarket_round22_targets.py",
            90,
            "unused variable 'allow_redirects'",
        ),
        (
            "src/simple_ai_trading/polymarket_round23_binance.py",
            76,
            "unused variable 'allow_redirects'",
        ),
    }
    assert vulture_check.IGNORED_FINDINGS == expected
    findings = "".join(
        f"{path.replace('/', chr(92))}:{line}: {message} (100% confidence)\n"
        for path, line, message in sorted(expected)
    )
    completed = mock.Mock(returncode=3, stdout=findings, stderr="")

    with mock.patch("tools.vulture_check.subprocess.run", return_value=completed):
        assert (
            vulture_check.run_vulture(
                ROOT,
                ["src/simple_ai_trading/polymarket_round25_resolution_store.py"],
                min_confidence=100,
            )
            == 0
        )


def test_vulture_scan_preserves_invalid_input_failure() -> None:
    completed = mock.Mock(returncode=1, stdout="", stderr="")

    with mock.patch("tools.vulture_check.subprocess.run", return_value=completed):
        assert (
            vulture_check.run_vulture(
                ROOT,
                ["src/simple_ai_trading/risk_controls.py"],
                min_confidence=100,
            )
            == 1
        )


def test_accelerator_lock_excludes_vulnerable_legacy_directml_stack() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))

    assert project["tool"]["uv"]["required-version"] == ">=0.12.1,<0.13"
    assert project["project"]["optional-dependencies"]["directml"] == [
        "onnxruntime-directml==1.24.4; platform_system == 'Windows'"
    ]

    versions: dict[str, set[str]] = {}
    for package in lock["package"]:
        versions.setdefault(package["name"], set()).add(package["version"])
    assert "torch-directml" not in versions
    assert versions["torch"] == {"2.13.0"}
    assert versions["h2"] == {"4.4.1"}
    assert versions["onnxruntime-directml"] == {"1.24.4"}
