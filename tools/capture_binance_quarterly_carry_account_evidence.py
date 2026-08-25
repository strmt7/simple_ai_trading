"""Capture frozen read-only Binance quarterly-carry account evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import requests


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.quarterly_carry_account_evidence import (  # noqa: E402
    capture_quarterly_carry_account_evidence,
    canonical_sha256,
)
from tools._round74_public_evidence_capture import write_artifact  # noqa: E402


CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-quarterly-carry-account-evidence-contract-v1.json"
)
API_KEY_VARIABLE = "SIMPLE_AI_TRADING_BINANCE_MAINNET_API_KEY"
API_SECRET_VARIABLE = "SIMPLE_AI_TRADING_BINANCE_MAINNET_API_SECRET"
JOURNAL_SCHEMA_VERSION = "binance-quarterly-carry-account-evidence-journal-v1"
ARTIFACT_SCHEMA_VERSION = "binance-quarterly-carry-account-evidence-artifact-v1"


def _normalized_sha256(path: Path) -> str:
    body = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(body).hexdigest()


def _load_contract() -> Mapping[str, object]:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("quarterly-carry account-evidence contract differs")
    body = dict(payload)
    claimed = body.pop("result_sha256", None)
    if claimed != canonical_sha256(body):
        raise ValueError("quarterly-carry account-evidence contract hash differs")
    implementation = payload.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("quarterly-carry implementation contract differs")
    expected = {
        "module_sha256": _normalized_sha256(
            SOURCE / "simple_ai_trading" / "quarterly_carry_account_evidence.py"
        ),
        "tool_sha256": _normalized_sha256(Path(__file__)),
    }
    if any(implementation.get(key) != value for key, value in expected.items()):
        raise ValueError("quarterly-carry implementation hash differs")
    return payload


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise ValueError("execution git identity differs")
    return commit


def _require_clean_tracked_worktree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise ValueError("capture requires a clean tracked worktree")


class _DurableJournal:
    def __init__(self, *, path: Path, contract: Mapping[str, object]) -> None:
        self.path = path
        self.payload: dict[str, object] = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "status": "active",
            "contract_result_sha256": contract["result_sha256"],
            "events": [],
            "orders_submitted": False,
            "credential_material_persisted": False,
        }
        self._write()

    def _write(self) -> None:
        body = dict(self.payload)
        body.pop("journal_sha256", None)
        self.payload["journal_sha256"] = canonical_sha256(body)
        write_artifact(self.path, self.payload)

    def __call__(self, event: Mapping[str, object]) -> None:
        events = list(self.payload["events"])
        events.append(dict(event))
        self.payload["events"] = events
        self._write()

    def complete(self, *, result_sha256: str) -> None:
        self.payload["status"] = "complete"
        self.payload["capture_result_sha256"] = result_sha256
        self._write()

    def fail(self, error: Exception) -> None:
        self.payload["status"] = "terminal_failure_without_retry"
        self.payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        self._write()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture six exact Binance commission responses plus minimal "
            "futures account configuration using signed GET requests only."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    contract = _load_contract()
    _require_clean_tracked_worktree()
    api_key = os.environ.get(API_KEY_VARIABLE, "")
    api_secret = os.environ.get(API_SECRET_VARIABLE, "")
    if not api_key or not api_secret:
        raise RuntimeError(
            "read-only Binance credentials are absent from the required "
            "ephemeral environment variables; no request was attempted"
        )
    journal = _DurableJournal(path=arguments.journal, contract=contract)
    try:
        with requests.Session() as session:
            capture = capture_quarterly_carry_account_evidence(
                api_key=api_key,
                api_secret=api_secret,
                timeout_seconds=arguments.timeout_seconds,
                request=session.request,
                journal=journal,
            )
        result = capture.as_dict()
        journal.complete(result_sha256=str(result["result_sha256"]))
        artifact: dict[str, object] = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "capture": result,
            "source": {
                "contract_path": str(CONTRACT.relative_to(ROOT)).replace("\\", "/"),
                "contract_result_sha256": contract["result_sha256"],
                "journal_path": str(arguments.journal.relative_to(ROOT)).replace(
                    "\\", "/"
                ),
                "journal_sha256": journal.payload["journal_sha256"],
                "execution_git_commit": _git_commit(),
            },
            "credential_transport": {
                "source": "process_environment_only",
                "api_key_variable": API_KEY_VARIABLE,
                "api_secret_variable": API_SECRET_VARIABLE,
                "credential_values_persisted": False,
            },
        }
        artifact["result_sha256"] = canonical_sha256(artifact)
        write_artifact(arguments.output, artifact)
    except Exception as exc:
        journal.fail(exc)
        raise
    print(
        json.dumps(
            {
                "result_sha256": artifact["result_sha256"],
                "signed_get_count": 7,
                "orders_submitted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
