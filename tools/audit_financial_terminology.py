"""Reject informal or superseded terminology in authored repository surfaces."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_AUTHORED_FILES = {
    "AGENTS.md",
    "LIVE_TESTNET_RUNBOOK.md",
    "PLANNING.md",
    "README.md",
}
AUTHORED_PREFIXES = (
    ".github/",
    "docs/",
    "native/windows/",
    "src/simple_ai_trading/",
    "tests/",
    "tools/",
)
AUTHORED_SUFFIXES = (
    ".cpp",
    ".h",
    ".hpp",
    ".json",
    ".md",
    ".py",
    ".svg",
    ".toml",
    ".yaml",
    ".yml",
)
MACHINE_EVIDENCE_JSON_PREFIXES = ("docs/ai/risk-review/latest/",)
TEXT_EXCLUSIONS = {"tools/audit_financial_terminology.py"}

# Build phrases from fragments so this enforcement module cannot flag itself.
_BANNED_PHRASE_SPECS = (
    (("action", " funnel"), "signal selection"),
    (("eligibility", " funnel"), "pre-trade control results"),
    (("profile gate", " funnel"), "pre-trade risk controls"),
    (("profitable", "-outcome"), "probability of profit"),
    (("mean", "-action"), "expected net return"),
    (("policy", " replay"), "policy-validation simulation"),
    (("policy", " trade"), "policy-validation simulated trade"),
    (("calibration threshold", " trace"), "threshold-selection simulation"),
    (("stress", " gates"), "stress-test acceptance criteria"),
    (("pressure", "-capacity"), "depth-normalized order flow"),
    (("model", " zoo"), "candidate-model set"),
    (("model", "-zoo"), "candidate-model set"),
    (("template", " zoo"), "strategy-template library"),
    (("profile", " zoo"), "candidate-profile set"),
    (("low-base", " rescue"), "reduced-base-model-weight fallback"),
    (("ai", " reviewer"), "LLM risk-assessment overlay"),
    (("magic", " alpha"), "validated alpha evidence"),
    (("trade", " anything"), "trade supported instruments"),
    (("action", " gates"), "pre-trade risk controls"),
    (("mandatory", " gates"), "binding risk controls"),
    (("rejected", " safely"), "rejected without trading authority"),
    (("10x", " maximum app-level leverage cap"), "20x maximum app-level futures leverage cap"),
    (("default leverage", ": `1x`"), "profile-specific futures defaults; spot remains 1x"),
    (("no static", " allowlist"), "hard BTC/ETH/SOL scope with dynamic liquidity checks"),
    (("across all eligible", " symbols"), "across supported BTC/ETH/SOL symbols"),
    (("future stock-market", " schedule changes"), "changing venue participation"),
    (("multi-asset by", " design"), "BTC/ETH/SOL-only by design"),
    (("testnet-first multi-asset", " day-trading"), "testnet-first BTC/ETH/SOL day trading"),
    (
        ("out-of-sample simulated trades", " across trained candidates"),
        "policy-window simulated trades with contamination status",
    ),
    (
        ("out-of-sample rows with positive", " predicted net return"),
        "policy-window rows with contamination status",
    ),
    (("free signal source inventory for", " btcusdc"), "free signal source inventory for BTC, ETH, and SOL"),
    (("current", " btcusdc workflow"), "supported BTC/ETH/SOL workflow"),
    (("simple", " bitcoin trading"), "Simple AI Trading"),
    (("simple", "_bitcoin_trading"), "simple_ai_trading"),
    (("simple", "-bitcoin-trading"), "simple-ai-trading"),
    (
        ("latest local btc-only", " profitability"),
        "retained legacy BTCUSDT profitability experiment",
    ),
    (
        ("current retained per-iteration", " evidence is"),
        "explicitly named latest-only evidence tracks",
    ),
    (("positive calibration", " traces"), "positive threshold-selection simulations"),
    (
        ("threshold-selection stress", " traces"),
        "threshold-selection stress simulations",
    ),
    (("best policy", " trace"), "best policy simulation"),
    (("executable trade", " traces"), "executable-trade simulations"),
    (("predominantly horizon", " exits"), "predominantly time exits"),
    (
        ("even the best", " trace"),
        "even the best threshold-selection simulation",
    ),
)
_BANNED_FILENAME_SPECS = (
    (("action", "-funnel"), "signal-selection"),
    (("profile-gate", "-funnel"), "pre-trade-risk-controls"),
    (("pressure", "-capacity"), "depth-normalized-order-flow"),
    (("simple", "_bitcoin_trading"), "simple_ai_trading"),
    (("simple", "-bitcoin-trading"), "simple-ai-trading"),
)


@dataclass(frozen=True)
class TerminologyFinding:
    path: str
    line: int | None
    term: str
    replacement: str

    def format(self) -> str:
        location = self.path if self.line is None else f"{self.path}:{self.line}"
        return f"{location}: replace {self.term!r} with {self.replacement!r}"


def _specs(
    values: Iterable[tuple[tuple[str, ...], str]],
) -> tuple[tuple[str, str], ...]:
    return tuple(("".join(parts), replacement) for parts, replacement in values)


BANNED_PHRASES = _specs(_BANNED_PHRASE_SPECS)
BANNED_FILENAME_PARTS = _specs(_BANNED_FILENAME_SPECS)


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def _is_authored_text(path: str) -> bool:
    normalized = _normalize(path)
    if normalized in TEXT_EXCLUSIONS:
        return False
    if not normalized.lower().endswith(AUTHORED_SUFFIXES):
        return False
    if normalized.lower().endswith(".json") and normalized.startswith(
        MACHINE_EVIDENCE_JSON_PREFIXES
    ):
        return False
    return normalized in ROOT_AUTHORED_FILES or normalized.startswith(AUTHORED_PREFIXES)


def audit_entries(entries: Iterable[tuple[str, str]]) -> list[TerminologyFinding]:
    findings: list[TerminologyFinding] = []
    for raw_path, text in entries:
        path = _normalize(raw_path)
        folded_path = path.casefold()
        for term, replacement in BANNED_FILENAME_PARTS:
            if term in folded_path:
                findings.append(
                    TerminologyFinding(
                        path=path,
                        line=None,
                        term=term,
                        replacement=replacement,
                    )
                )
        if not _is_authored_text(path):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            folded_line = line.casefold()
            for term, replacement in BANNED_PHRASES:
                if term in folded_line:
                    findings.append(
                        TerminologyFinding(
                            path=path,
                            line=line_number,
                            term=term,
                            replacement=replacement,
                        )
                    )
    return sorted(
        findings,
        key=lambda item: (item.path.casefold(), item.line or 0, item.term),
    )


def git_visible_files(repo_root: Path = REPO_ROOT) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(
        {
            _normalize(line.strip())
            for line in result.stdout.splitlines()
            if line.strip()
        }
    )


def audit_repository(repo_root: Path = REPO_ROOT) -> list[TerminologyFinding]:
    entries: list[tuple[str, str]] = []
    for relative_path in git_visible_files(repo_root):
        path = repo_root / relative_path
        if not path.is_file():
            continue
        text = (
            path.read_text(encoding="utf-8") if _is_authored_text(relative_path) else ""
        )
        entries.append((relative_path, text))
    return audit_entries(entries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit authored surfaces for superseded financial terminology."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root containing the Git index (default: detected root).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    findings = audit_repository(args.repo_root.resolve())
    if findings:
        for finding in findings:
            print(finding.format())
        print(f"financial terminology audit failed: {len(findings)} finding(s)")
        return 1
    print("financial terminology audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
