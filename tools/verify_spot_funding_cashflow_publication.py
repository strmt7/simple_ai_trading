"""Verify the consumed audit from published bytes, never local data/ or network."""

from pathlib import Path
import json

from tools.review_spot_funding_cashflows import (
    D,
    ROOT,
    canonical,
    cashflow_metrics,
    digest,
    normalize_history,
)

BASE = Path("docs/review/2026-09-04")
RAW_PREFIX = Path("data/binance-broad-crypto-funding-carry-preflight-v1/raw")


def published_bytes(binding: dict, root: Path = ROOT) -> bytes:
    path = Path(binding["path"])
    if path.is_relative_to(RAW_PREFIX):
        suffix = path.relative_to(RAW_PREFIX)
        if len(suffix.parts) != 1:
            raise ValueError("unexpected raw-source suffix")
        path = BASE / "spot-funding-sources" / suffix
    resolved = (root / path).resolve()
    resolved.relative_to(root.resolve())
    raw = resolved.read_bytes()
    if digest(raw) != binding["sha256"]:
        raise ValueError(f"published input hash differs: {path}")
    return raw


def verify(root: Path = ROOT) -> int:
    result = json.loads(
        (root / BASE / "spot-funding-cashflow-review.json").read_bytes()
    )
    if canonical(result, "result_sha256") != result["result_sha256"]:
        raise ValueError("published result self-hash differs")
    inputs = {
        entry["path"]: published_bytes(entry, root) for entry in result["bindings"]
    }
    prior_path = "docs/model-research/action-value/binance-broad-crypto-funding-carry-preflight-v1-2026-08-27.json"
    prior = json.loads(inputs[prior_path])
    receipts = {row["name"]: row for row in prior["sources"]["responses"]}
    histories = {}
    for row in result["rows"]:
        symbol = row["symbol"]
        if symbol not in histories:
            path = receipts["funding-" + symbol.lower()]["raw_path"]
            histories[symbol] = normalize_history(json.loads(inputs[path]), symbol)
        history = histories[symbol]
        selected = [
            entry
            for entry in history
            if row["first_time_ms"] <= entry["time"] <= row["last_time_ms"]
        ]
        reference = next(
            entry["mark"]
            for entry in history
            if entry["time"] == row["reference_time_ms"]
        )
        metrics = cashflow_metrics(
            selected, reference, D(row["duration_days"]), D(32), D(2), D(1000)
        )
        if metrics != row["metrics"] or len(selected) != row["row_count"]:
            raise ValueError("published scenario reconstruction differs")
    if len(result["rows"]) != 51 or len(histories) != 17:
        raise ValueError("published population differs")
    return len(result["rows"])


if __name__ == "__main__":
    print(f"Reconstructed {verify()} published roles with zero data/ or network reads")
