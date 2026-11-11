"""Reject informal or superseded terminology in authored repository surfaces."""

from __future__ import annotations

import argparse
import hashlib
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
# Exact historical content at 982bd9ac is evidence, not current editorial copy.
# Hash UTF-8 text with CRLF normalized to LF for portable Git checkouts. A path
# alone never exempts modified content; filename checks remain unconditional.
# Provenance and the four newly classified before/audit records are documented
# in docs/review/2026-11-11/feedback-repair/review.md.
IMMUTABLE_EVIDENCE_TEXT_SHA256 = {
    "docs/model-research/action-value/accepted-edge-profitability-durability-audit-v1-2026-08-30.json": "eea73f41a823c9914c26f1339c7d839ef927d1262eddaa1745415fff482a4100",
    "docs/model-research/action-value/polymarket-us-combo-rfq-overview-adjudication-v1-2026-09-01.json": "ff077d8165ecba526a395d2c02a024b3eb99f3b389f25cf36f28e7325c7fefe9",
    "docs/model-research/action-value/round-040-causal-meta-label-capacity-ai-design.json": "85a8ee86d42905f1a7875132794a38fac921201aa22e79b6c9a8be4c7af4ad8b",
    "docs/model-research/action-value/round-041-prequential-meta-label-ai-design.json": "a7d5df6ac4b848ec8ab195a182f2686fb1aeaff526bd1009ddd96e3b630de41c",
    "docs/model-research/action-value/round-047-cost-aware-utility-tcn-design.json": "31965b3544674a1f82a07f2a4f6c7d90df49a5194c6718eaeebb8d23e4044c98",
    "docs/model-research/action-value/round-048-minute-logistic-mixture-tcn-design.json": "5ee9414c12fd3aaf5c5064d3cfea0b369526f3cd3cecbad69b8e974a736e2efd",
    "docs/model-research/action-value/round-049-cost-aware-action-hurdle-tcn-design.json": "96d325dc8636550a9b78d27a61467ba05143739eb5fc7cb425d79f000e71f7a9",
    "docs/model-research/action-value/round-074-event-sequence-model-design-v108.json": "473e672687a733cba4d2ffe932547165323e9941ad539d53038cb67ef3bdea4c",
    "docs/model-research/action-value/round-074-event-sequence-model-design-v116.json": "9112cc3486aa30a3f53207606d804b7d9e7271547b4282b76d1a3d9069d3ddd3",
    "docs/review/2026-09-04/nyc-sep5-frontier/durability-before.json": "a727031f643003aab7516742d9ef49fb855c396b9289d5303f78de53f8151253",
    "docs/review/2026-09-04/paradex-index-publication/durability-before.json": "5412354d9dc5861e9e49acfa7fc4025db39e1227a2c56fea62c5a6398549ab61",
    "docs/review/2026-10-07/model-integration/audit-before.json": "9359919129109c78651272955bd8a558f686aa188bf65e431cc19c49b020d052",
    "docs/review/2026-10-07/model-integration/review.md": "bf81672840e13babcfb0d521dec1211cda203635cfe410b8aa781a6031d84cf5",
    "src/simple_ai_trading/impact_absorption_ai_contract_screen.py": "652a84da10794dcd53b45befc5c1164ac50d6c3c0ce744b94601d69dda423767",
    "src/simple_ai_trading/polymarket_round17_publication.py": "3f75bbd348cf7d29cdcbacccef58cc4d1866efd7123cfe130456591ef467eeaa",
    "src/simple_ai_trading/round74_ai_qualification_operator.py": "239ba4afc7ace70ebb0023bb94b4edd9fafdc78d6f11e232f71f5467139bae61",
}

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
    (
        ("10x", " maximum app-level leverage cap"),
        "20x maximum app-level futures leverage cap",
    ),
    (
        ("default leverage", ": `1x`"),
        "profile-specific futures defaults; spot remains 1x",
    ),
    (
        ("no static", " allowlist"),
        "hard BTC/ETH/SOL scope with dynamic liquidity checks",
    ),
    (("across all eligible", " symbols"), "across supported BTC/ETH/SOL symbols"),
    (("future stock-market", " schedule changes"), "changing venue participation"),
    (("multi-asset by", " design"), "BTC/ETH/SOL-only by design"),
    (
        ("testnet-first multi-asset", " day-trading"),
        "testnet-first BTC/ETH/SOL day trading",
    ),
    (
        ("out-of-sample simulated trades", " across trained candidates"),
        "policy-window simulated trades with contamination status",
    ),
    (
        ("out-of-sample rows with positive", " predicted net return"),
        "policy-window rows with contamination status",
    ),
    (
        ("free signal source inventory for", " btcusdc"),
        "free signal source inventory for BTC, ETH, and SOL",
    ),
    (("current", " btcusdc workflow"), "supported BTC/ETH/SOL workflow"),
    (("simple", " bitcoin trading"), "Simple AI Trading"),
    (("simple", "_bitcoin_trading"), "simple_ai_trading"),
    (("simple", "-bitcoin-trading"), "simple-ai-trading"),
    (
        ("feature contract `l1-tape-causal-", "v7`"),
        "Feature contract `l1-tape-causal-v8`",
    ),
    (("current v16/", "v7"), "current v16/v8"),
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
    (("executed", " mean"), "simulated-trade mean"),
    (("executed trade", " mean"), "simulated-trade mean"),
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
    if normalized.startswith("docs/model-research/") and "/raw/" in normalized:
        return False
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
        historical_digest = IMMUTABLE_EVIDENCE_TEXT_SHA256.get(path)
        if historical_digest is not None:
            text_digest = hashlib.sha256(
                text.replace("\r\n", "\n").encode("utf-8")
            ).hexdigest()
            if historical_digest == text_digest:
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
            path.read_bytes().decode("utf-8")
            if _is_authored_text(relative_path)
            else ""
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
