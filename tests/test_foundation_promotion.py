from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tools import promote_foundation_benchmark as promotion


_REPOSITORY = Path(__file__).resolve().parents[1]
_EVIDENCE = _REPOSITORY / "docs" / "ai" / "foundation" / "latest"


def test_promoted_bundle_is_exact_hash_bound_and_path_sanitized(tmp_path: Path) -> None:
    output = tmp_path / "latest"

    promotion.promote(
        _EVIDENCE / "report.json",
        _EVIDENCE / "observations.csv",
        _EVIDENCE / "benchmark.svg",
        output,
    )

    manifest = promotion.verify_promoted_bundle(output)
    assert manifest["observation_count"] == 1536
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["trading_authority"] is False
    assert report["config"]["database_path"] == "data/market_data.sqlite"
    assert "C:\\" not in json.dumps(report)

    with (output / "observations.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        promotion.verify_promoted_bundle(output)


def test_promotion_recomputes_observation_accounting_after_hash_validation(
    tmp_path: Path,
) -> None:
    observations = tmp_path / "observations.csv"
    report_path = tmp_path / "report.json"
    shutil.copyfile(_EVIDENCE / "observations.csv", observations)
    report = json.loads((_EVIDENCE / "report.json").read_text(encoding="utf-8"))
    rows: list[dict[str, str]]
    with observations.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    rows[0]["absolute_error"] = str(float(rows[0]["absolute_error"]) + 1.0)
    with observations.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report["observations_sha256"] = hashlib.sha256(observations.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="inconsistent error"):
        promotion._validated_payload(
            report_path,
            observations,
            _EVIDENCE / "benchmark.svg",
        )


def test_promotion_rejects_unknown_stale_artifacts_and_host_paths(tmp_path: Path) -> None:
    output = tmp_path / "latest"
    output.mkdir()
    (output / "old-round.svg").write_text("stale", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected stale artifacts"):
        promotion.promote(
            _EVIDENCE / "report.json",
            _EVIDENCE / "observations.csv",
            _EVIDENCE / "benchmark.svg",
            output,
        )

    payload = json.loads((_EVIDENCE / "report.json").read_text(encoding="utf-8"))
    payload["unrecognized_local_artifact"] = r"worker failed at C:\private\result.json"
    with pytest.raises(RuntimeError, match="host-local path"):
        promotion._sanitized_payload(payload, source_report_sha256="0" * 64)
