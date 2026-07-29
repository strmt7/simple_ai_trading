from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping


ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.polymarket_historical_screen import (  # noqa: E402
    HistoricalScreenStore,
)
from simple_ai_trading.polymarket_round16 import (  # noqa: E402
    Round16HistoricalPublicClient,
    collect_round16_market_identities,
    load_round16_historical_contract,
)
from simple_ai_trading.polymarket_round16_dataset import (  # noqa: E402
    materialize_round16_causal_features,
)
from simple_ai_trading.polymarket_round16_evaluation import (  # noqa: E402
    evaluate_round16_test_once,
)
from simple_ai_trading.polymarket_round16_model import (  # noqa: E402
    build_round16_pretest_artifact,
    fit_round16_pretest_candidates,
    load_round16_model_panel,
    record_round16_pretest_artifact,
)
from simple_ai_trading.polymarket_round16_targets import (  # noqa: E402
    collect_round16_development_targets,
    collect_round16_test_targets_once,
)


DEFAULT_CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-016-btc-15m-horizon-comparison-v1.json"
)
DEFAULT_DATABASE = (
    ROOT / "data" / "polymarket-round16-btc-15m-screen-v1.duckdb"
)
DEFAULT_FLOW_DATABASE = (
    ROOT / "data" / "polymarket-btc-flow-history-v1.duckdb"
)
DEFAULT_ARTIFACT_DIRECTORY = ROOT / "data" / "round16-shadow-artifacts"
_SOURCE_PATHS = (
    "docs/model-research/polymarket/"
    "round-016-btc-15m-horizon-comparison-v1.json",
    "src/simple_ai_trading/lightgbm_backend.py",
    "src/simple_ai_trading/polymarket_historical_model.py",
    "src/simple_ai_trading/polymarket_round16.py",
    "src/simple_ai_trading/polymarket_round16_dataset.py",
    "src/simple_ai_trading/polymarket_round16_evaluation.py",
    "src/simple_ai_trading/polymarket_round16_model.py",
    "src/simple_ai_trading/polymarket_round16_shadow.py",
    "src/simple_ai_trading/polymarket_round16_targets.py",
    "tools/run_polymarket_round16_screen.py",
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


def _emit(event: str, values: Mapping[str, object]) -> None:
    print(
        _canonical_json({"event": event, **dict(values)}),
        flush=True,
    )


def _status(store: HistoricalScreenStore) -> dict[str, object]:
    connection = store.connect()
    counts = {
        "market_count": int(
            connection.execute(
                "SELECT count(*) FROM feature.market_identity"
            ).fetchone()[0]
        ),
        "feature_row_count": int(
            connection.execute(
                "SELECT count(*) FROM feature.causal_row"
            ).fetchone()[0]
        ),
        "development_target_count": int(
            connection.execute(
                """
                SELECT count(*) FROM target.official_resolution
                WHERE role IN ('train', 'tune')
                """
            ).fetchone()[0]
        ),
        "test_target_count": int(
            connection.execute(
                """
                SELECT count(*) FROM target.official_resolution
                WHERE role = 'test'
                """
            ).fetchone()[0]
        ),
        "pretest_manifest_count": int(
            connection.execute(
                "SELECT count(*) FROM feature.pretest_manifest"
            ).fetchone()[0]
        ),
        "evaluation_manifest_count": int(
            connection.execute(
                "SELECT count(*) FROM target.evaluation_manifest"
            ).fetchone()[0]
        ),
    }
    return {
        "state": store.state,
        "contract_sha256": store.contract.contract_sha256,
        "database_bytes": (
            store.path.stat().st_size if store.path.exists() else 0
        ),
        **counts,
    }


def _committed_source_head() -> str:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *_SOURCE_PATHS],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if tracked.returncode != 0:
        raise ValueError("Round 16 model source must be tracked before fitting")
    for cached in (False, True):
        command = ["git", "diff", "--quiet"]
        if cached:
            command.append("--cached")
        command.extend(["HEAD", "--", *_SOURCE_PATHS])
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise ValueError(
                "Round 16 model source must be committed before fitting"
            )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    commit = result.stdout.strip().lower()
    if len(commit) != 40:
        raise ValueError("Round 16 source commit is invalid")
    return commit


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _export_artifacts(
    store: HistoricalScreenStore,
    *,
    directory: Path,
) -> Mapping[str, object]:
    if store.state != "evaluated":
        raise ValueError("Round 16 export requires a completed evaluation")
    pretest, pretest_envelope_sha = store.pretest_artifact()
    row = store.connect().execute(
        """
        SELECT artifact_json, artifact_sha256
        FROM target.evaluation_manifest
        WHERE singleton
        """
    ).fetchone()
    if row is None:
        raise ValueError("Round 16 evaluation artifact is missing")
    try:
        evaluation = json.loads(str(row[0]))
    except json.JSONDecodeError as exc:
        raise ValueError("Round 16 evaluation artifact is invalid") from exc
    if (
        not isinstance(evaluation, Mapping)
        or _canonical_json(evaluation) != str(row[0])
        or _canonical_sha256(evaluation) != str(row[1])
        or evaluation.get("pretest_artifact_sha256")
        != pretest_envelope_sha
    ):
        raise ValueError("Round 16 evaluation artifact integrity differs")
    pins_body: dict[str, object] = {
        "schema_version": "polymarket-round16-shadow-pins-v1",
        "contract_sha256": store.contract.contract_sha256,
        "dataset_sha256": str(pretest["dataset_sha256"]),
        "pretest_envelope_sha256": pretest_envelope_sha,
        "evaluation_envelope_sha256": str(row[1]),
        "accepted_predictive_edge": (
            evaluation.get("accepted_predictive_edge") is True
        ),
        "trading_authority": False,
    }
    pins = {**pins_body, "artifact_sha256": _canonical_sha256(pins_body)}
    _atomic_json(directory / "round16-pretest.json", pretest)
    _atomic_json(directory / "round16-evaluation.json", dict(evaluation))
    _atomic_json(directory / "round16-shadow-pins.json", pins)
    return {
        "directory": str(directory.resolve()),
        "pretest_envelope_sha256": pretest_envelope_sha,
        "evaluation_envelope_sha256": str(row[1]),
        "accepted_predictive_edge": pins_body["accepted_predictive_edge"],
        "trading_authority": False,
    }


def main(
    *,
    default_contract: Path = DEFAULT_CONTRACT,
    default_database: Path = DEFAULT_DATABASE,
    default_flow_database: Path = DEFAULT_FLOW_DATABASE,
    default_artifact_directory: Path = DEFAULT_ARTIFACT_DIRECTORY,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the phase-gated BTC Polymarket fifteen-minute screen."
        )
    )
    parser.add_argument(
        "phase",
        choices=(
            "status",
            "identities",
            "features",
            "unlabeled",
            "development-targets",
            "fit",
            "test-targets",
            "evaluate",
            "export",
        ),
    )
    parser.add_argument("--contract", type=Path, default=default_contract)
    parser.add_argument("--database", type=Path, default=default_database)
    parser.add_argument(
        "--flow-database",
        type=Path,
        default=default_flow_database,
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=default_artifact_directory,
    )
    parser.add_argument(
        "--compute-backend",
        choices=("auto", "cpu", "directml", "cuda", "rocm"),
        default="auto",
    )
    parser.add_argument(
        "--acknowledge-one-use-test-access",
        action="store_true",
        help="Required only for the irreversible test-target phase.",
    )
    args = parser.parse_args()
    if (
        args.phase == "test-targets"
        and not args.acknowledge_one_use_test_access
    ):
        parser.error(
            "test-targets requires --acknowledge-one-use-test-access"
        )
    contract = load_round16_historical_contract(args.contract)
    with HistoricalScreenStore(
        args.database,
        contract=contract.historical,
    ) as store:
        _emit("round16_status", _status(store))
        if args.phase == "status":
            return 0
        client = Round16HistoricalPublicClient()
        if args.phase in {"identities", "unlabeled"}:
            if store.state == "initialized":
                result = collect_round16_market_identities(
                    store,
                    contract,
                    client,
                    progress=_emit,
                )
                _emit(
                    "round16_identities_complete",
                    {
                        "day_count": len(result),
                        "market_count": sum(result.values()),
                    },
                )
        if args.phase in {"features", "unlabeled"}:
            if store.state == "identities_complete":
                result = materialize_round16_causal_features(
                    store,
                    contract,
                    flow_database_path=args.flow_database,
                    progress=_emit,
                )
                _emit(
                    "round16_features_complete",
                    {
                        "dataset_sha256": result.dataset_sha256,
                        "row_count": result.row_count,
                        "condition_count": result.condition_count,
                        "role_counts": dict(result.role_counts),
                    },
                )
        if args.phase == "development-targets":
            result = collect_round16_development_targets(
                store,
                contract,
                client,
                progress=_emit,
            )
            _emit("round16_development_targets_complete", result)
        if args.phase == "fit":
            train = load_round16_model_panel(
                store,
                contract,
                roles=("train",),
            )
            tune = load_round16_model_panel(
                store,
                contract,
                roles=("tune",),
            )
            candidates = fit_round16_pretest_candidates(
                train,
                tune,
                compute_backend=args.compute_backend,
                progress=_emit,
            )
            artifact = build_round16_pretest_artifact(
                train,
                tune,
                candidates,
                contract=contract,
                source_commit=_committed_source_head(),
            )
            envelope_sha = record_round16_pretest_artifact(
                store,
                contract,
                artifact,
            )
            _emit(
                "round16_pretest_complete",
                {
                    "pretest_envelope_sha256": envelope_sha,
                    "best_control_id": artifact["selected_best_control"],
                    "best_challenger_id": (
                        artifact["selected_best_challenger"]
                    ),
                    "test_targets_accessed": False,
                    "trading_authority": False,
                },
            )
        if args.phase == "test-targets":
            result = collect_round16_test_targets_once(
                store,
                contract,
                client,
                progress=_emit,
            )
            _emit("round16_test_targets_complete", result)
        if args.phase == "evaluate":
            artifact, envelope_sha = evaluate_round16_test_once(
                store,
                contract,
            )
            _emit(
                "round16_evaluation_complete",
                {
                    "evaluation_envelope_sha256": envelope_sha,
                    "accepted_predictive_edge": (
                        artifact["accepted_predictive_edge"]
                    ),
                    "best_control_id": artifact["best_control_id"],
                    "best_challenger_id": artifact["best_challenger_id"],
                    "gates": artifact["gates"],
                    "trading_authority": False,
                },
            )
        if args.phase == "export":
            _emit(
                "round16_artifacts_exported",
                _export_artifacts(
                    store,
                    directory=args.artifact_directory,
                ),
            )
        _emit("round16_status", _status(store))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
