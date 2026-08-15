from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Mapping


ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.polymarket_historical_l2 import (  # noqa: E402
    PolymarketHistoricalL2Client,
)
from simple_ai_trading.polymarket_round22_ingestion import (  # noqa: E402
    POLYMARKET_ROUND22_MAXIMUM_CONDITIONS_PER_RUN,
    Round22GammaIdentityClient,
    ingest_round22_development_conditions,
)
from simple_ai_trading.polymarket_round22_feature_operator import (  # noqa: E402
    materialize_round22_development_features,
)
from simple_ai_trading.polymarket_round22_feature_store import (  # noqa: E402
    Round22FeatureStore,
)
from simple_ai_trading.polymarket_round22_pilot import (  # noqa: E402
    Round22PilotStore,
    development_conditions,
    load_round22_pilot_contract,
)


DEFAULT_DATABASE = ROOT / "data" / "polymarket-round22-pilot-v1.duckdb"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _emit(event: str, values: Mapping[str, object]) -> None:
    print(_canonical_json({"event": event, **dict(values)}), flush=True)


def _status(store: Round22PilotStore) -> dict[str, object]:
    completed = store.completed_slugs()
    development = {item.slug for item in development_conditions(store.contract)}
    feature_table_exists = bool(
        store.connection.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'feature'
              AND table_name = 'condition_feature_chunk'
            """
        ).fetchone()[0]
    )
    feature_count = 0
    feature_compressed_bytes = 0
    if feature_table_exists:
        feature_count, feature_compressed_bytes = store.connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(compressed_size_bytes), 0)
            FROM feature.condition_feature_chunk
            """
        ).fetchone()
    return {
        "authentication_used": False,
        "binance_used": False,
        "database_bytes": store.path.stat().st_size,
        "design_sha256": store.contract.design_sha256,
        "development_complete_count": len(completed & development),
        "development_remaining_count": len(development - completed),
        "feature_counts": store.feature_counts(),
        "feature_materialized_condition_count": int(feature_count),
        "feature_payload_compressed_bytes": int(feature_compressed_bytes),
        "live_trading_authority": False,
        "paper_trading_authority": False,
        "qualification_sha256": store.contract.qualification_sha256,
        "target_accessed": False,
        "target_row_count": store.target_row_count(),
    }


def main(
    *,
    repository: Path = ROOT,
    default_database: Path = DEFAULT_DATABASE,
) -> int:
    parser = argparse.ArgumentParser(
        description="Run bounded target-blind Polymarket Round 22 L2 ingestion."
    )
    parser.add_argument("phase", choices=("status", "ingest", "features"))
    parser.add_argument("--database", type=Path, default=default_database)
    parser.add_argument(
        "--maximum-conditions",
        type=int,
        choices=range(1, POLYMARKET_ROUND22_MAXIMUM_CONDITIONS_PER_RUN + 1),
        default=1,
    )
    parser.add_argument(
        "--role",
        choices=("all_development", "train", "tune_calibration", "tune_selection"),
        default="all_development",
    )
    args = parser.parse_args()
    contract = load_round22_pilot_contract(repository)
    with Round22PilotStore(
        args.database,
        contract=contract,
        read_only=args.phase == "status",
    ) as store:
        _emit("round22_status", _status(store))
        if args.phase == "status":
            return 0
        if args.phase == "ingest":
            result = ingest_round22_development_conditions(
                store=store,
                contract=contract,
                identity_client=Round22GammaIdentityClient(),
                l2_client=PolymarketHistoricalL2Client(),
                maximum_conditions=args.maximum_conditions,
                role=args.role,
                progress=_emit,
            )
            _emit("round22_ingestion_complete", asdict(result))
        else:
            result = materialize_round22_development_features(
                pilot_store=store,
                feature_store=Round22FeatureStore(store),
                maximum_conditions=args.maximum_conditions,
                role=args.role,
                progress=_emit,
            )
            _emit("round22_feature_materialization_complete", asdict(result))
        _emit("round22_status", _status(store))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
