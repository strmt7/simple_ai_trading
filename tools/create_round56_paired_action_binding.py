"""Bind the committed Round 56 implementation and external evidence inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 56
DESIGN_SCHEMA = "round-056-paired-action-distributional-design-v1"
BINDING_SCHEMA = "round-056-paired-action-execution-binding-v1"
AI_REPORT_SCHEMA = "round-056-ai-factor-research-report-v1"
AI_LEDGER_SCHEMA = "round-056-action-conditioned-factor-program-ledger-v1"
SOURCE_CERTIFICATE_SHA256 = (
    "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
)
PATHS = (
    "docs/model-research/action-value/round-056-paired-action-distributional-design.json",
    "src/simple_ai_trading/ai_factor_programs.py",
    "src/simple_ai_trading/bounded_alpha_lightgbm.py",
    "src/simple_ai_trading/compute.py",
    "src/simple_ai_trading/cross_asset_cost_data.py",
    "src/simple_ai_trading/derivatives_hurdle_data.py",
    "src/simple_ai_trading/lightgbm_backend.py",
    "src/simple_ai_trading/paired_action_lightgbm.py",
    "src/simple_ai_trading/stop_time_payoff_data.py",
    "src/simple_ai_trading/storage.py",
    "tests/test_ai_factor_programs.py",
    "tests/test_bounded_alpha_lightgbm.py",
    "tests/test_paired_action_lightgbm.py",
    "tests/test_stop_time_payoff_data.py",
    "tools/create_round56_paired_action_binding.py",
    "tools/run_round56_ai_factor_research.py",
    "tools/run_round56_paired_action.py",
)
CACHE_FILES = (
    "features.npy",
    "forward_return_bps.npy",
    "hourly_return_bps.npy",
    "metadata.json",
    "timestamps_ms.npy",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} root is not an object")
    return value


def _git(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Round 56 binding Git command failed") from exc


def _manifest(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def _validate_canonical_report(
    report: dict[str, object],
    *,
    design_sha: str,
    label: str,
) -> str:
    canonical = dict(report)
    claimed = str(canonical.pop("report_sha256", ""))
    if (
        report.get("schema_version") != AI_REPORT_SCHEMA
        or report.get("round") != ROUND
        or report.get("design_sha256") != design_sha
        or report.get("market_values_read") is not False
        or report.get("timestamps_read") is not False
        or report.get("outcomes_read") is not False
        or report.get("trading_authority") is not False
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError(f"Round 56 {label} identity is invalid")
    return claimed


def run(arguments: argparse.Namespace) -> int:
    design = _read_object(arguments.design.resolve(), "Round 56 design")
    canonical = dict(design)
    design_sha = str(canonical.pop("design_sha256", ""))
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or design_sha != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 56 design identity is invalid")
    implementation_commit = _git("rev-parse", "HEAD")
    blobs = [
        {
            "path": path,
            "git_blob_oid": _git("rev-parse", f"{implementation_commit}:{path}"),
        }
        for path in PATHS
    ]
    cache_root = arguments.derived_cache.resolve()
    cache = []
    for name in CACHE_FILES:
        row = _manifest(cache_root / name)
        row["name"] = name
        cache.append(row)
    source_certificate = _read_object(
        arguments.source_certificate.resolve(), "Round 38 source certificate"
    )
    if source_certificate.get("source_certificate_sha256") != SOURCE_CERTIFICATE_SHA256:
        raise ValueError("Round 56 source certificate identity differs")

    ai_report = _read_object(arguments.ai_report.resolve(), "Round 56 AI report")
    _validate_canonical_report(
        ai_report, design_sha=design_sha, label="accepted AI report"
    )
    if ai_report.get("status") != "complete":
        raise ValueError("Round 56 accepted AI report did not complete")
    rejected_ai_report = _read_object(
        arguments.rejected_ai_report.resolve(), "Round 56 rejected AI report"
    )
    _validate_canonical_report(
        rejected_ai_report, design_sha=design_sha, label="rejected AI report"
    )
    ai_ledger = _read_object(arguments.ai_ledger.resolve(), "Round 56 AI ledger")
    ledger_canonical = dict(ai_ledger)
    ledger_sha = str(ledger_canonical.pop("ledger_sha256", ""))
    if (
        ai_ledger.get("schema_version") != AI_LEDGER_SCHEMA
        or ai_ledger.get("design_sha256") != design_sha
        or ledger_sha != _canonical_sha256(ledger_canonical)
    ):
        raise ValueError("Round 56 AI ledger identity is invalid")
    ai_implementation_commit = str(ai_report.get("implementation_commit", ""))
    if not ai_implementation_commit or subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            ai_implementation_commit,
            implementation_commit,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        raise ValueError("Round 56 AI evidence commit is not an ancestor")
    ai_blobs = ai_report.get("implementation_blobs")
    if not isinstance(ai_blobs, dict) or not ai_blobs:
        raise ValueError("Round 56 AI implementation blobs are absent")
    for source_path, expected_oid in ai_blobs.items():
        if (
            _git("rev-parse", f"{ai_implementation_commit}:{source_path}")
            != str(expected_oid)
            or _git("rev-parse", f"{implementation_commit}:{source_path}")
            != str(expected_oid)
        ):
            raise ValueError(f"Round 56 AI implementation blob changed: {source_path}")

    payload: dict[str, object] = {
        "schema_version": BINDING_SCHEMA,
        "round": ROUND,
        "design_sha256": design_sha,
        "implementation_commit": implementation_commit,
        "blobs": blobs,
        "derived_cache": cache,
        "external_evidence": {
            "source_certificate": _manifest(arguments.source_certificate.resolve()),
            "ai_report": _manifest(arguments.ai_report.resolve()),
            "ai_ledger": _manifest(arguments.ai_ledger.resolve()),
            "rejected_ai_report": _manifest(
                arguments.rejected_ai_report.resolve()
            ),
        },
        "command": (
            ".venv311\\Scripts\\python.exe tools\\run_round56_paired_action.py "
            "--binding docs\\model-research\\action-value\\round-056-paired-action-execution-binding.json "
            "--source-certificate <external-certificate.json> "
            "--derived-cache <round45-derived-cache> --ai-report <round56-v2-report.json> "
            "--ai-ledger <round56-v2-factor-program-ledger.json> "
            "--rejected-ai-report <round56-v1-report.json> "
            "--evidence-root <new-external-evidence-root> --compute-backend directml"
        ),
    }
    payload["binding_sha256"] = _canonical_sha256(payload)
    write_json_atomic(arguments.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs/model-research/action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-056-paired-action-distributional-design.json",
    )
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--derived-cache", type=Path, required=True)
    parser.add_argument("--ai-report", type=Path, required=True)
    parser.add_argument("--ai-ledger", type=Path, required=True)
    parser.add_argument("--rejected-ai-report", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=research / "round-056-paired-action-execution-binding.json",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))
