"""Regenerate the host-agnostic Polymarket current-status manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "polymarket"
LATEST = RESEARCH / "latest"
MANIFEST = LATEST / "publication-integrity.json"
_TEXT_SUFFIXES = {".csv", ".json", ".md", ".svg"}
_SOURCE_TABLES = {
    "charts/optimization-progress.svg": ["optimization-progress.csv"],
    "charts/round14-cumulative-log-loss-advantage.svg": ["round14-conditions.csv"],
    "charts/round14-decision-offset-log-loss.svg": ["round14-decision-offsets.csv"],
    "charts/round14-held-out-metrics.svg": ["round14-candidates.csv"],
}
_CURRENT_RESEARCH_ARTIFACTS = (
    "round-022-diagnostic-results-v1.json",
    "round-023-binance-ingestion-result-v1.json",
    "round-023-binance-lead-lag-source-v1.json",
    "round-023-binance-lead-lag-source-v2.json",
    "round-023-lead-lag-model-spec-v1.json",
    "round-023-lead-lag-performance.svg",
    "round-023-lead-lag-results-v1.json",
    "round-023-source-qualification-attempt1-2026-08-03.json",
    "round-023-source-qualification-v2-2026-08-03.json",
    "round-024-preregistration-publication-2026-08-03.json",
    "round-024-preregistration-publication-v2-2026-08-10.json",
    "round-024-prospective-receipt-lead-lag-spec-v1.json",
    "round-024-prospective-receipt-lead-lag-spec-v2.json",
    "round-024-prospective-receipt-lead-lag-spec-v3.json",
    "round-025-ai-risk-advisory-contract-v1.json",
    "round-025-ai-risk-advisory-contract-v2.json",
    "round-025-ai-risk-advisory-contract-v3.json",
    "round-025-ai-risk-advisory-contract-v4.json",
    "round-025-ai-risk-advisory-contract-v5.json",
    "round-025-ai-risk-advisory-contract-v6.json",
    "round-025-ai-risk-advisory-host-probe-2026-08-10.json",
    "round-025-ai-risk-advisory-host-probe-v2-2026-08-10.json",
    "round-025-ai-risk-advisory-host-probe-v3-2026-08-10.json",
    "round-025-ai-risk-advisory-host-probe-v4-2026-08-10.json",
    "round-025-ai-risk-advisory-host-probe-v5-2026-08-10.json",
    "round-025-ai-risk-advisory-host-probe-v6-2026-08-10.json",
    "round-025-ai-risk-advisory-host-probe-v7-2026-08-10.json",
    "round-025-ai-risk-scenario-contract-v1.json",
    "round-025-ai-risk-scenario-contract-v2.json",
    "round-025-ai-risk-scenario-contract-v3.json",
    "round-025-ai-risk-scenario-correction-contract-v1.json",
    "round-025-ai-risk-scenario-host-probe-2026-08-10.json",
    "round-025-ai-risk-scenario-host-probe-v2-2026-08-10.json",
    "round-025-ai-risk-scenario-host-probe-v3-2026-08-10.json",
    "round-025-ai-risk-scenario-host-probe-v3-correction-2026-08-10.json",
    "round-025-ai-uplift-evaluation-contract-v1.json",
    "round-025-ai-uplift-evaluation-contract-v2.json",
    "round-025-control-fit-contract-v1.json",
    "round-025-lightgbm-fit-contract-v1.json",
    "round-025-predictive-evaluation-contract-v1.json",
    "round-025-sequence-materialization-contract-v1.json",
    "round-025-tcn-directml-host-probe-2026-08-10.json",
    "round-025-tcn-fit-contract-v1.json",
    "round-025-terminal-receipt-materialization-design-v1.json",
    "round-025-twap-native-candidate-selection-amendment-v2.json",
    "round-025-twap-native-candidate-selection-design-v1.json",
    "round-025-twap-native-model-design-v1.json",
    "round-025-twap-core-campaign-plan-publication-2026-08-10.json",
    "round-025-twap-core-capture-design-v1.json",
    "round-025-twap-source-qualification-2026-08-10.json",
)


def canonical_artifact_bytes(path: Path) -> bytes:
    payload = path.read_bytes()
    if path.suffix.lower() in _TEXT_SUFFIXES:
        payload = payload.replace(b"\r\n", b"\n")
    return payload


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return _sha256(payload)


def _artifact_paths() -> tuple[Path, ...]:
    selected = [
        LATEST / "README.md",
        LATEST / "ai-risk-models-rejected.json",
        LATEST / "public-clob-live-probe-2026-08-10.json",
        LATEST / "public-predictor-live-probe.json",
        LATEST / "round-014-evaluation.json",
        LATEST / "round-014-pretest.json",
    ]
    selected.extend(sorted((LATEST / "charts").glob("*")))
    selected.extend(sorted((LATEST / "tables").glob("*")))
    selected.extend(RESEARCH / name for name in _CURRENT_RESEARCH_ARTIFACTS)
    if any(not path.is_file() for path in selected):
        raise FileNotFoundError("Polymarket latest publication artifact is missing")
    return tuple(
        sorted(
            set(selected),
            key=lambda path: path.relative_to(RESEARCH).as_posix(),
        )
    )


def _artifact(path: Path) -> dict[str, object]:
    relative = path.relative_to(RESEARCH).as_posix()
    payload = canonical_artifact_bytes(path)
    result: dict[str, object] = {
        "path": relative,
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            result["row_count"] = sum(1 for _row in csv.DictReader(handle))
    latest_relative = (
        relative.removeprefix("latest/") if relative.startswith("latest/") else ""
    )
    source_tables = _SOURCE_TABLES.get(latest_relative)
    if source_tables is not None:
        result["source_tables"] = source_tables
    if relative == "round-023-lead-lag-performance.svg":
        result["source_artifacts"] = ["round-023-lead-lag-results-v1.json"]
    return result


def build_manifest() -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "polymarket-current-status-publication-v1",
        "status": "round25_v2_waiting_ai_behavior_qualified_uplift_evaluator_implemented",
        "asset": "BTC",
        "market_variant": "fiveminute",
        "latest_research_round": 25,
        "latest_evaluated_round": 23,
        "latest_graph_round": 23,
        "accepted_historical_predictive_edge": False,
        "round21_result_available": False,
        "round23_result_available": True,
        "round24_result_available": False,
        "round25_result_available": False,
        "round25_capture_revision": 2,
        "round25_v1_capture_eligible": False,
        "round25_v2_campaign_start_utc": "2026-08-11T00:00:00Z",
        "round25_twap_native_model_design_frozen": True,
        "round25_feature_contract_implemented": True,
        "round25_candidate_ledger_frozen": True,
        "round25_development_dataset_contract_implemented": True,
        "round25_control_fit_contract_frozen": True,
        "round25_control_fit_implemented": True,
        "round25_control_model_fitted": False,
        "round25_lightgbm_fit_contract_frozen": True,
        "round25_lightgbm_operator_implemented": True,
        "round25_lightgbm_model_fitted": False,
        "round25_sequence_materialization_contract_frozen": True,
        "round25_sequence_materializer_implemented": True,
        "round25_sequence_corpus_materialized": False,
        "round25_tcn_fit_contract_frozen": True,
        "round25_tcn_operator_implemented": True,
        "round25_tcn_host_directml_mechanics_verified": True,
        "round25_tcn_seed_model_fitted": False,
        "round25_tcn_ensemble_fitted": False,
        "round25_tcn_model_fitted": False,
        "round25_predictive_evaluation_contract_frozen": True,
        "round25_predictive_evaluator_implemented": True,
        "round25_prediction_panel_frozen": False,
        "round25_predictive_result_available": False,
        "round25_ai_risk_contract_frozen": True,
        "round25_ai_operator_implemented": True,
        "round25_ai_host_mechanics_verified": True,
        "round25_ai_safety_behavior_mechanics_verified": True,
        "round25_ai_uplift_contract_frozen": True,
        "round25_ai_uplift_contract_revision": 2,
        "round25_ai_uplift_evaluator_implemented": True,
        "round25_ai_uplift_population_available": False,
        "round25_ai_uplift_result_available": False,
        "round25_ai_uplift_verified": False,
        "round25_sealed_test_campaign_planned": False,
        "round25_settlement_rule_verified": False,
        "round25_terminal_design_frozen": False,
        "round25_terminal_receipt_audit_available": False,
        "round25_materialization_result_available": False,
        "profitability_claim": False,
        "paper_authority": False,
        "live_trading_authority": False,
        "manual_chart_edits_permitted": False,
        "source_contract_sha256": "9275cfb5cb95427ee9565e3196c87b1c91ce114f713eae2561ad8f6144ddc6f0",
        "source_dataset_sha256": "a6fd7f55ec7705d391a035b923c4440c3f92ddfb5b1b169ee6b7df3d67aa7665",
        "source_evaluation_artifact_sha256": "c3385d4894ca430a91442d1023bf01f61c650f6d208a4d0f89d11007ee5a11c0",
        "source_pretest_artifact_sha256": "03e4fae64f34f835cac338dee111701c73419929e97a22939e12037eba8dfec9",
        "evaluated_test_start_utc": "2026-07-06T00:00:00Z",
        "evaluated_test_end_utc": "2026-07-06T01:00:00Z",
        "artifacts": [_artifact(path) for path in _artifact_paths()],
    }
    body["manifest_sha256"] = _canonical_sha256(body)
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    expected = json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n"
    if args.check:
        return 0 if MANIFEST.read_text(encoding="utf-8") == expected else 1
    MANIFEST.write_text(expected, encoding="utf-8", newline="\n")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
