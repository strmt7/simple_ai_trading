from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.polymarket_round22_feature_store import (  # noqa: E402
    Round22FeatureStore,
)
from simple_ai_trading.polymarket_round22_features import (  # noqa: E402
    POLYMARKET_ROUND22_FEATURE_NAMES_SHA256,
    POLYMARKET_ROUND22_FEATURE_POLICY_SHA256,
)
from simple_ai_trading.polymarket_round22_pilot import (  # noqa: E402
    Round22PilotStore,
    load_round22_pilot_contract,
)


DEFAULT_DATABASE = ROOT / "data" / "polymarket-round22-pilot-v1.duckdb"
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-022-diagnostic-mini-cohort-v1.json"
)
ROLE_ORDER = {"train": 0, "tune_calibration": 1, "tune_selection": 2}


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


def _build_artifact(store: Round22PilotStore) -> dict[str, object]:
    state = str(
        store.connection.execute(
            "SELECT state FROM feature.pilot_manifest WHERE singleton"
        ).fetchone()[0]
    )
    if state != "feature_ingestion" or store.target_row_count() != 0:
        raise ValueError("Round 22 diagnostic must be frozen before target access")
    rows = store.connection.execute(
        """
        SELECT source.condition_id, source.slug, source.role,
               source.manifest_sha256, feature.manifest_sha256,
               feature.row_count, feature.available_count,
               feature.sequence_complete_count,
               feature.tabular_history_complete_count,
               feature.compressed_size_bytes
        FROM feature.condition_manifest AS source
        JOIN feature.condition_feature_chunk AS feature
          ON feature.condition_id = source.condition_id
        ORDER BY CASE source.role
                   WHEN 'train' THEN 0
                   WHEN 'tune_calibration' THEN 1
                   WHEN 'tune_selection' THEN 2
                   ELSE 3
                 END,
                 source.slug
        """
    ).fetchall()
    if len(rows) != 36:
        raise ValueError("Round 22 diagnostic requires exactly 36 feature conditions")
    feature_store = Round22FeatureStore(store)
    role_counts = Counter(str(row[2]) for row in rows)
    if role_counts != Counter(
        {"train": 12, "tune_calibration": 12, "tune_selection": 12}
    ):
        raise ValueError("Round 22 diagnostic role counts differ")
    conditions: list[dict[str, object]] = []
    aggregates: dict[str, dict[str, int]] = {}
    for row in rows:
        condition_id = str(row[0])
        audit = feature_store.audit_condition(condition_id)
        if audit["manifest_sha256"] != str(row[4]) or audit["target_row_count"] != 0:
            raise ValueError("Round 22 diagnostic feature audit differs")
        conditions.append(
            {
                "condition_id": condition_id,
                "feature_manifest_sha256": str(row[4]),
                "role": str(row[2]),
                "slug": str(row[1]),
                "source_manifest_sha256": str(row[3]),
            }
        )
        aggregate = aggregates.setdefault(
            str(row[2]),
            {
                "available_rows": 0,
                "compressed_bytes": 0,
                "condition_count": 0,
                "rows": 0,
                "sequence_complete_rows": 0,
                "tabular_history_complete_rows": 0,
            },
        )
        aggregate["condition_count"] += 1
        aggregate["rows"] += int(row[5])
        aggregate["available_rows"] += int(row[6])
        aggregate["sequence_complete_rows"] += int(row[7])
        aggregate["tabular_history_complete_rows"] += int(row[8])
        aggregate["compressed_bytes"] += int(row[9])
    body: dict[str, object] = {
        "authority": {
            "binance_private_api": False,
            "live_trading": False,
            "paper_trading": False,
            "polymarket_authentication": False,
            "polymarket_order_submission": False,
        },
        "diagnostic_design": {
            "candidate_count": 1,
            "cluster_unit": "polymarket_condition",
            "decision": "pipeline_falsification_only_not_an_edge_or_profitability_claim",
            "model": "executable_market_prior_plus_l2_regularized_logistic_residual",
            "partitions": {
                "calibration": "tune_calibration_only",
                "fit": "train_only",
                "final_diagnostic_selection": "tune_selection_only",
            },
            "primary_predictive_metrics": [
                "condition_clustered_out_of_sample_brier_score",
                "condition_clustered_out_of_sample_log_loss",
                "calibration_intercept",
                "calibration_slope",
            ],
            "reference": "executable_market_prior",
            "seed": 2201,
            "statistical_limits": {
                "economic_promotion_allowed": False,
                "independent_conditions_per_partition": 12,
                "sealed_test_access_allowed": False,
                "wider_backfill_required_for_edge_claim": True,
            },
        },
        "feature_evidence": {
            "feature_names_sha256": POLYMARKET_ROUND22_FEATURE_NAMES_SHA256,
            "feature_policy_sha256": POLYMARKET_ROUND22_FEATURE_POLICY_SHA256,
            "role_aggregates": {
                role: aggregates[role]
                for role in sorted(aggregates, key=ROLE_ORDER.__getitem__)
            },
        },
        "parents": {
            "pilot_design_sha256": store.contract.design_sha256,
            "source_qualification_sha256": store.contract.qualification_sha256,
        },
        "population": {
            "condition_count": len(conditions),
            "conditions": conditions,
            "role_counts": dict(
                sorted(role_counts.items(), key=lambda item: ROLE_ORDER[item[0]])
            ),
        },
        "schema_version": "polymarket-round22-diagnostic-mini-cohort-v1",
        "status": "frozen_after_features_before_any_target_access",
        "target_contract": {
            "collection_limit": 36,
            "gamma_source": "https://gamma-api.polymarket.com/markets/{market_id}",
            "clob_source": "https://clob.polymarket.com/markets/{condition_id}",
            "dual_source_terminal_agreement_required": True,
            "only_registered_conditions_allowed": True,
            "target": "official_binary_up_or_down_resolution",
        },
    }
    return {**body, "preregistration_sha256": _canonical_sha256(body)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the target-blind Round 22 diagnostic cohort."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    contract = load_round22_pilot_contract(ROOT)
    with Round22PilotStore(args.database, contract=contract, read_only=True) as store:
        artifact = _build_artifact(store)
    rendered = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    output = args.output.resolve()
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise ValueError("Round 22 diagnostic preregistration already differs")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8", newline="\n")
    print(
        _canonical_json(
            {
                "condition_count": artifact["population"]["condition_count"],
                "output": str(output),
                "preregistration_sha256": artifact["preregistration_sha256"],
                "target_accessed": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
