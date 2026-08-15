"""Shared CLI/Windows controls for the independent Polymarket Round 21 workflow."""

from __future__ import annotations

import argparse
import asyncio
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import duckdb

from .compute import SUPPORTED_COMPUTE_BACKENDS
from .polymarket_ai_veto import PolymarketAIVetoConfig
from .polymarket_round21_corpus_store import publish_round21_core_corpus
from .polymarket import PolymarketPublicClient
from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_round21_ai_operator import run_round21_development_ai_program
from .polymarket_round21_ai_selection import POLYMARKET_ROUND21_AI_CANDIDATES
from .polymarket_round21_ablation import (
    load_round21_probability_basis_ablation_result,
)
from .polymarket_round21_one_use import (
    build_round21_pretest_manifest,
    create_round21_one_use_claim,
    load_round21_completed_sealed_bundle,
)
from .polymarket_round21_operator import (
    assemble_round21_core_development,
    assemble_round21_matched_development,
    evaluate_round21_core_probability_basis,
    fit_round21_core_baseline,
    fit_round21_matched_optional_candidate,
)
from .polymarket_round21_economic_operator import (
    replay_round21_development_economics,
)
from .polymarket_round21_model import load_round21_development_artifact
from .polymarket_round21_sealed import build_round21_sealed_result_bundle
from .polymarket_round21_sealed_operator import (
    evaluate_round21_terminal_sealed_once,
)
from .polymarket_round21_shadow_runtime import (
    build_polymarket_round21_shadow_runtime_stack,
)
from .polymarket_round21_shadow_store import Round21ProspectiveShadowStore
from .polymarket_round21_sidecar_terminal import (
    build_round21_sidecar_terminal_manifest,
    load_round21_sidecar_terminal_manifest,
    write_round21_sidecar_terminal_manifest,
)
from .polymarket_round21_terminal import (
    build_round21_terminal_transport_manifest,
    load_round21_terminal_transport_manifest,
    write_round21_terminal_transport_manifest,
)
from .storage import write_json_atomic


def register_polymarket_round21_commands(
    subparsers: argparse._SubParsersAction,  # noqa: SLF001 - argparse has no public type
) -> None:
    terminal = subparsers.add_parser(
        "polymarket-round21-terminal",
        help="seal the completed Round 21 transport evidence",
        description=(
            "After the independent Polymarket capture reaches its scheduled end, "
            "bind every completed, degraded, interrupted, and failed segment into "
            "one hash-valid terminal transport manifest. This command opens no "
            "capture database, outcome, model, credential, account, or order."
        ),
    )
    terminal.add_argument(
        "--campaign-plan",
        required=True,
        help="frozen Round 20/21 continuous-campaign plan JSON",
    )
    terminal.add_argument(
        "--state-root",
        required=True,
        help="terminal campaign state directory containing segment evidence",
    )
    terminal.add_argument(
        "--output",
        required=True,
        help="new or replaceable terminal transport manifest JSON path",
    )
    terminal.add_argument(
        "--repository",
        default=".",
        help="repository root containing the frozen Round 21 design",
    )
    terminal.add_argument(
        "--observed-at-ms",
        type=int,
        default=None,
        help="optional fixed terminal timestamp for reproducible controlled runs",
    )
    terminal.add_argument(
        "--json",
        action="store_true",
        help="emit the complete canonical terminal transport manifest",
    )
    terminal.set_defaults(func=command_polymarket_round21_terminal)

    sidecar_terminal = subparsers.add_parser(
        "polymarket-round21-sidecar-terminal",
        help="seal the independent public Binance predictor evidence",
        description=(
            "After the independent public Binance sidecar reaches its scheduled "
            "end, bind every segment result into a terminal manifest. This "
            "command opens no capture payload, outcome, model, credential, "
            "account, Polymarket execution state, or order."
        ),
    )
    sidecar_terminal.add_argument(
        "--campaign-plan",
        required=True,
        help="frozen Round 21 Binance-sidecar campaign plan JSON",
    )
    sidecar_terminal.add_argument(
        "--state-root",
        required=True,
        help="terminal sidecar state directory containing segment evidence",
    )
    sidecar_terminal.add_argument(
        "--output",
        required=True,
        help="new or replaceable sidecar terminal manifest JSON path",
    )
    sidecar_terminal.add_argument(
        "--observed-at-ms",
        type=int,
        default=None,
        help="optional fixed terminal timestamp for reproducible controlled runs",
    )
    sidecar_terminal.add_argument(
        "--json",
        action="store_true",
        help="emit the complete canonical sidecar terminal manifest",
    )
    sidecar_terminal.set_defaults(func=command_polymarket_round21_sidecar_terminal)

    parser = subparsers.add_parser(
        "polymarket-round21-corpus",
        help="publish the terminal target-blind Round 21 core corpus",
        description=(
            "After the independent Polymarket capture is terminal, reconcile its "
            "exact receipts and atomically publish physically separate development "
            "and sealed-test core feature stores. This command reads no outcomes, "
            "models, Binance data, credentials, accounts, or orders."
        ),
    )
    parser.add_argument(
        "--source-database",
        required=True,
        help="closed Polymarket Round 21 evidence DuckDB",
    )
    parser.add_argument(
        "--terminal-transport-manifest",
        required=True,
        help="hash-valid terminal Round 21 transport manifest JSON",
    )
    parser.add_argument(
        "--publication-directory",
        required=True,
        help="new directory for one atomic development/sealed-test publication",
    )
    parser.add_argument(
        "--repository",
        default=".",
        help="repository root containing the frozen Round 21 design",
    )
    parser.add_argument(
        "--observed-at-ms",
        type=int,
        default=None,
        help="optional fixed audit timestamp for reproducible controlled runs",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the complete canonical publication manifest",
    )
    parser.set_defaults(func=command_polymarket_round21_corpus)

    basis_ablation = subparsers.add_parser(
        "polymarket-round21-ablate-basis",
        help="gate the Round 21 market-prior probability basis",
        description=(
            "Run the preregistered paired six-fit screen on audited Polymarket "
            "development rows before any full model fit. Both log-loss and Brier "
            "improvement intervals must clear zero. This opens no Binance account, "
            "credential, position, order, sealed target, or execution authority."
        ),
    )
    basis_ablation.add_argument(
        "--source-database",
        required=True,
        help="closed Polymarket evidence DuckDB containing official resolutions",
    )
    basis_ablation.add_argument(
        "--terminal-transport-manifest",
        required=True,
        help="exact terminal transport manifest used for corpus publication",
    )
    basis_ablation.add_argument(
        "--publication-directory",
        required=True,
        help="validated Round 21 core corpus publication",
    )
    basis_ablation.add_argument(
        "--repository",
        default=".",
        help="repository root containing the preregistered ablation design",
    )
    basis_ablation.add_argument(
        "--output",
        required=True,
        help="new immutable probability-basis ablation result JSON path",
    )
    basis_ablation.add_argument(
        "--json",
        action="store_true",
        help="emit the compact source-bound operator result",
    )
    basis_ablation.set_defaults(func=command_polymarket_round21_ablate_basis)

    fit_core = subparsers.add_parser(
        "polymarket-round21-fit-core",
        help="fit the frozen Round 21 core development baseline",
        description=(
            "Load one audited terminal publication, attach independently "
            "cross-checked official outcomes, and fit the frozen core baseline "
            "across train, calibration, and selection roles. This is not the "
            "optional Binance-layer comparison or a trading-authority decision."
        ),
    )
    fit_core.add_argument(
        "--source-database",
        required=True,
        help="closed Polymarket evidence DuckDB containing official resolutions",
    )
    fit_core.add_argument(
        "--terminal-transport-manifest",
        required=True,
        help="exact terminal transport manifest used for corpus publication",
    )
    fit_core.add_argument(
        "--publication-directory",
        required=True,
        help="validated Round 21 development/sealed-test corpus publication",
    )
    fit_core.add_argument(
        "--basis-ablation-result",
        required=True,
        help="accepted source-bound probability-basis ablation result",
    )
    fit_core.add_argument(
        "--output",
        required=True,
        help="atomic core-baseline model artifact JSON path",
    )
    fit_core.add_argument(
        "--compute-backend",
        choices=SUPPORTED_COMPUTE_BACKENDS,
        default="auto",
        help="host-neutral training backend; explicit accelerators fail if unavailable",
    )
    fit_core.add_argument(
        "--json",
        action="store_true",
        help="emit the compact source-bound operator result",
    )
    fit_core.set_defaults(func=command_polymarket_round21_fit_core)

    fit_matched = subparsers.add_parser(
        "polymarket-round21-fit-matched",
        help="fit the matched core and optional predictor candidates",
        description=(
            "Replay the independent public Binance sidecar once at exact "
            "Polymarket decision times, attach the same official outcomes and "
            "frozen roles, and fit the core plus optional predictor candidates. "
            "This is predictive development evidence, not profitability or "
            "trading authority."
        ),
    )
    fit_matched.add_argument(
        "--source-database",
        required=True,
        help="closed Polymarket evidence DuckDB containing official resolutions",
    )
    fit_matched.add_argument(
        "--terminal-transport-manifest",
        required=True,
        help="exact Polymarket terminal manifest used for corpus publication",
    )
    fit_matched.add_argument(
        "--publication-directory",
        required=True,
        help="validated Round 21 core corpus publication",
    )
    fit_matched.add_argument(
        "--sidecar-database",
        required=True,
        help="closed WAL-free public Binance sidecar DuckDB",
    )
    fit_matched.add_argument(
        "--sidecar-terminal-manifest",
        required=True,
        help="exact independent Binance sidecar terminal manifest",
    )
    fit_matched.add_argument(
        "--basis-ablation-result",
        required=True,
        help="accepted source-bound probability-basis ablation result",
    )
    fit_matched.add_argument(
        "--output",
        required=True,
        help="new matched development model artifact JSON path",
    )
    fit_matched.add_argument(
        "--compute-backend",
        choices=SUPPORTED_COMPUTE_BACKENDS,
        default="auto",
        help="host-neutral training backend; explicit accelerators fail if unavailable",
    )
    fit_matched.add_argument(
        "--json",
        action="store_true",
        help="emit the compact source-bound operator result",
    )
    fit_matched.set_defaults(func=command_polymarket_round21_fit_matched)

    economics = subparsers.add_parser(
        "polymarket-round21-evaluate-development",
        help="replay all 81 development economic ledgers once",
        description=(
            "Rebuild exact Polymarket books from the terminal receipt audit and "
            "stream the frozen development population through every captured "
            "fee, latency, depth, adverse-price, and risk-profile scenario. "
            "Binance remains an optional public predictor only. This command "
            "has no credential, account, order, paper, or live authority."
        ),
    )
    economics.add_argument("--source-database", required=True)
    economics.add_argument("--terminal-transport-manifest", required=True)
    economics.add_argument("--publication-directory", required=True)
    economics.add_argument("--model-artifact", required=True)
    economics.add_argument(
        "--selected-layer",
        choices=("core", "core_spot", "core_spot_usdm"),
        default="core",
    )
    economics.add_argument(
        "--sidecar-database",
        default=None,
        help="closed public Binance sidecar DuckDB required by matched artifacts",
    )
    economics.add_argument(
        "--sidecar-terminal-manifest",
        default=None,
        help="independent sidecar terminal manifest required by matched artifacts",
    )
    economics.add_argument("--initial-capital-quote", type=Decimal, default="10000")
    economics.add_argument("--minimum-edge-per-share", type=Decimal, default="0.02")
    economics.add_argument("--builder-taker-fee-bps", type=Decimal, default="0")
    economics.add_argument(
        "--output",
        required=True,
        help="new compact hash-bound development economic report JSON",
    )
    economics.add_argument("--json", action="store_true")
    economics.set_defaults(func=command_polymarket_round21_evaluate_development)

    ai_development = subparsers.add_parser(
        "polymarket-round21-ai-development",
        help="benchmark and ablate the finite Round 21 local-AI candidates",
        description=(
            "Build one immutable historical AI case per development condition, "
            "benchmark the three preregistered local models, replay all candidates "
            "through the same 81 after-cost ledgers in one second source pass, and "
            "nominate at most one sealed-test challenger. This independent "
            "Polymarket workflow uses no Binance account or execution state and "
            "grants no paper or live authority."
        ),
    )
    ai_development.add_argument("--source-database", required=True)
    ai_development.add_argument("--terminal-transport-manifest", required=True)
    ai_development.add_argument("--publication-directory", required=True)
    ai_development.add_argument("--model-artifact", required=True)
    ai_development.add_argument(
        "--selected-layer",
        choices=("core", "core_spot", "core_spot_usdm"),
        default="core",
    )
    ai_development.add_argument("--sidecar-database", default=None)
    ai_development.add_argument("--sidecar-terminal-manifest", default=None)
    ai_development.add_argument(
        "--risk-benchmark-evidence",
        required=True,
        help="immutable local-AI adversarial risk benchmark file",
    )
    ai_development.add_argument(
        "--qwen3-5-9b-digest",
        required=True,
        help="expected exact Ollama manifest digest for qwen3.5:9b",
    )
    ai_development.add_argument(
        "--fin-r1-8b-digest",
        required=True,
        help="expected exact Ollama manifest digest for fin-r1:8b",
    )
    ai_development.add_argument(
        "--fino1-8b-digest",
        required=True,
        help="expected exact Ollama manifest digest for fino1:8b",
    )
    ai_development.add_argument(
        "--ai-cache-database",
        required=True,
        help="separate resumable AI-response cache; never the source database",
    )
    ai_development.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="loopback Ollama endpoint",
    )
    ai_development.add_argument("--ai-timeout-seconds", type=float, default=30.0)
    ai_development.add_argument(
        "--ai-minimum-confidence",
        type=float,
        default=0.65,
    )
    ai_development.add_argument(
        "--ai-maximum-latency-seconds",
        type=float,
        default=15.0,
    )
    ai_development.add_argument(
        "--initial-capital-quote",
        type=Decimal,
        default="10000",
    )
    ai_development.add_argument(
        "--minimum-edge-per-share",
        type=Decimal,
        default="0.02",
    )
    ai_development.add_argument(
        "--builder-taker-fee-bps",
        type=Decimal,
        default="0",
    )
    ai_development.add_argument(
        "--output",
        required=True,
        help="new hash-bound development AI program report JSON",
    )
    ai_development.add_argument(
        "--acknowledge-one-use-test-access",
        action="store_true",
        help="consume the frozen sealed test exactly once after development completes",
    )
    ai_development.add_argument(
        "--repository",
        default=".",
        help="clean repository root bound into the optional pretest seal",
    )
    ai_development.add_argument("--one-use-store", default=None)
    ai_development.add_argument("--pretest-output", default=None)
    ai_development.add_argument("--claim-output", default=None)
    ai_development.add_argument("--sealed-output", default=None)
    ai_development.add_argument(
        "--json",
        action="store_true",
        help="emit a compact completion summary",
    )
    ai_development.set_defaults(func=command_polymarket_round21_ai_development)

    recovery = subparsers.add_parser(
        "polymarket-round21-recover-sealed",
        help="recover a completed Round 21 sealed bundle",
        description=(
            "Export the complete validated sealed result retained in a completed "
            "one-use ledger. This recovery path does not reopen test access, "
            "rerun a model, access an account, or grant trading authority."
        ),
    )
    recovery.add_argument(
        "--one-use-store",
        required=True,
        help="completed durable Round 21 one-use SQLite ledger",
    )
    recovery.add_argument(
        "--output",
        required=True,
        help="new sealed-result bundle JSON path",
    )
    recovery.add_argument("--json", action="store_true")
    recovery.set_defaults(func=command_polymarket_round21_recover_sealed)

    shadow = subparsers.add_parser(
        "polymarket-round21-shadow",
        help="run or audit the no-order BTC five-minute Round 21 shadow",
        description=(
            "Run target-free BTC five-minute scoring from exact accepted model and "
            "sealed-evaluation files, or audit a terminal shadow ledger. This "
            "workflow has no credentials, account, promotion, or order authority."
        ),
    )
    shadow.add_argument("--action", choices=("run", "audit"), default="audit")
    shadow.add_argument(
        "--shadow-database",
        required=True,
        help="dedicated append-only Round 21 shadow SQLite database",
    )
    shadow.add_argument(
        "--run-id",
        default=None,
        help=(
            "exact 32-hex run ID; optional for a new run and for audit only "
            "when the database contains exactly one verified run"
        ),
    )
    shadow.add_argument(
        "--model-artifact",
        default=None,
        help="accepted Round 21 development artifact required for run",
    )
    shadow.add_argument(
        "--model-file-sha256",
        default=None,
        help="expected exact model-file SHA-256 required for run",
    )
    shadow.add_argument(
        "--evaluation-report",
        default=None,
        help="accepted sealed Round 21 evaluation bundle required for run",
    )
    shadow.add_argument(
        "--evaluation-file-sha256",
        default=None,
        help="expected exact evaluation-file SHA-256 required for run",
    )
    shadow.add_argument(
        "--duration-seconds",
        type=float,
        default=3_600.0,
        help="fixed prospective run duration in [5, 2592000] seconds",
    )
    shadow.add_argument(
        "--discovery-seconds",
        type=float,
        default=1.0,
        help="public Gamma discovery interval in [0.25, 30] seconds",
    )
    shadow.add_argument(
        "--poll-seconds",
        type=float,
        default=0.05,
        help="in-memory score polling interval in [0.01, 0.25] seconds",
    )
    shadow.add_argument(
        "--queue-capacity",
        type=int,
        default=20_000,
        help="bounded public-feed queue capacity in [1000, 100000]",
    )
    shadow.add_argument("--json", action="store_true")
    shadow.set_defaults(func=command_polymarket_round21_shadow)


def command_polymarket_round21_terminal(args: argparse.Namespace) -> int:
    output = Path(args.output)
    try:
        manifest = build_round21_terminal_transport_manifest(
            repository=Path(args.repository),
            plan_path=Path(args.campaign_plan),
            state_root=Path(args.state_root),
            observed_at_ms=args.observed_at_ms,
        )
        write_round21_terminal_transport_manifest(output, manifest)
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            f"polymarket-round21-terminal failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(
            "Round 21 terminal transport sealed: "
            f"manifest={manifest['manifest_sha256']} "
            f"eligible_runs={len(manifest['eligible_run_ids'])} "
            f"output={output} authority=false"
        )
    return 0


def command_polymarket_round21_sidecar_terminal(args: argparse.Namespace) -> int:
    output = Path(args.output)
    try:
        manifest = build_round21_sidecar_terminal_manifest(
            plan_path=Path(args.campaign_plan),
            state_root=Path(args.state_root),
            observed_at_ms=args.observed_at_ms,
        )
        write_round21_sidecar_terminal_manifest(output, manifest)
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            "polymarket-round21-sidecar-terminal failed: "
            f"{exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(
            "Round 21 public Binance sidecar sealed: "
            f"manifest={manifest['manifest_sha256']} "
            f"eligible_runs={len(manifest['eligible_run_ids'])} "
            f"output={output} authority=false"
        )
    return 0


def command_polymarket_round21_corpus(args: argparse.Namespace) -> int:
    try:
        transport = load_round21_terminal_transport_manifest(
            Path(args.terminal_transport_manifest)
        )
        manifest = publish_round21_core_corpus(
            repository=Path(args.repository),
            source_database=Path(args.source_database),
            terminal_transport_manifest=transport,
            publication_directory=Path(args.publication_directory),
            observed_at_ms=args.observed_at_ms,
        )
    except (duckdb.Error, OSError, RuntimeError, ValueError) as exc:
        print(
            f"polymarket-round21-corpus failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(
            "Round 21 core corpus published: "
            f"manifest={manifest['manifest_sha256']} "
            "authority=false"
        )
    return 0


def command_polymarket_round21_ablate_basis(args: argparse.Namespace) -> int:
    output = Path(args.output)
    try:
        if output.exists() or output.is_symlink():
            raise ValueError("probability-basis ablation output already exists")
        transport = load_round21_terminal_transport_manifest(
            Path(args.terminal_transport_manifest)
        )
        ablation = evaluate_round21_core_probability_basis(
            repository=Path(args.repository),
            publication_directory=Path(args.publication_directory),
            source_database=Path(args.source_database),
            terminal_transport_manifest=transport,
        )
        write_json_atomic(output, ablation, indent=2, sort_keys=True)
        file_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    except (duckdb.Error, OSError, RuntimeError, ValueError) as exc:
        print(
            f"polymarket-round21-ablate-basis failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    result = {
        "schema_version": "polymarket-round21-basis-ablation-operator-result-v1",
        "result_path": str(output.resolve()),
        "result_file_sha256": file_sha256,
        "result_sha256": ablation["result_sha256"],
        "basis_accepted": ablation["basis_accepted"],
        "next_action": ablation["next_action"],
        "development_targets_accessed": True,
        "sealed_test_accessed": False,
        "economic_evaluation_completed": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "Round 21 probability basis evaluated: "
            f"accepted={str(ablation['basis_accepted']).lower()} "
            f"result={ablation['result_sha256']} file={file_sha256} "
            "profitability=false authority=false"
        )
    return 0


def command_polymarket_round21_fit_core(args: argparse.Namespace) -> int:
    output = Path(args.output)
    try:
        if output.exists() or output.is_symlink():
            raise ValueError("core-baseline output already exists")
        transport = load_round21_terminal_transport_manifest(
            Path(args.terminal_transport_manifest)
        )
        basis_ablation = load_round21_probability_basis_ablation_result(
            Path(args.basis_ablation_result)
        )
        artifact = fit_round21_core_baseline(
            publication_directory=Path(args.publication_directory),
            source_database=Path(args.source_database),
            terminal_transport_manifest=transport,
            basis_ablation_result=basis_ablation,
            compute_backend=str(args.compute_backend),
        )
        write_json_atomic(output, artifact, indent=2, sort_keys=True)
        file_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    except (duckdb.Error, OSError, RuntimeError, ValueError) as exc:
        print(
            f"polymarket-round21-fit-core failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    result = {
        "schema_version": "polymarket-round21-core-fit-operator-result-v1",
        "artifact_path": str(output.resolve()),
        "artifact_file_sha256": file_sha256,
        "artifact_sha256": artifact["artifact_sha256"],
        "population_layer": "core",
        "optional_binance_comparison_completed": False,
        "sealed_test_accessed": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "Round 21 core baseline fitted: "
            f"artifact={result['artifact_sha256']} "
            f"file={file_sha256} optional_comparison=false authority=false"
        )
    return 0


def command_polymarket_round21_fit_matched(args: argparse.Namespace) -> int:
    output = Path(args.output)
    try:
        if output.exists() or output.is_symlink():
            raise ValueError("matched-model output already exists")
        transport = load_round21_terminal_transport_manifest(
            Path(args.terminal_transport_manifest)
        )
        sidecar_terminal = load_round21_sidecar_terminal_manifest(
            Path(args.sidecar_terminal_manifest)
        )
        basis_ablation = load_round21_probability_basis_ablation_result(
            Path(args.basis_ablation_result)
        )
        artifact = fit_round21_matched_optional_candidate(
            publication_directory=Path(args.publication_directory),
            source_database=Path(args.source_database),
            terminal_transport_manifest=transport,
            sidecar_database=Path(args.sidecar_database),
            sidecar_terminal_manifest=sidecar_terminal,
            basis_ablation_result=basis_ablation,
            compute_backend=str(args.compute_backend),
        )
        write_json_atomic(output, artifact, indent=2, sort_keys=True)
        file_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    except (duckdb.Error, OSError, RuntimeError, ValueError) as exc:
        print(
            f"polymarket-round21-fit-matched failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    result = {
        "schema_version": "polymarket-round21-matched-fit-operator-result-v1",
        "artifact_path": str(output.resolve()),
        "artifact_file_sha256": file_sha256,
        "artifact_sha256": artifact["artifact_sha256"],
        "trained_layers": artifact["trained_layers"],
        "matched_predictive_comparison_completed": True,
        "economic_evaluation_completed": False,
        "sealed_test_accessed": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "Round 21 matched predictor candidates fitted: "
            f"artifact={result['artifact_sha256']} file={file_sha256} "
            "economic_evaluation=false authority=false"
        )
    return 0


def _load_round21_development_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object], object, str]:
    transport = load_round21_terminal_transport_manifest(
        Path(args.terminal_transport_manifest)
    )
    artifact = load_round21_development_artifact(Path(args.model_artifact))
    selected_layer = str(args.selected_layer)
    trained_layers = tuple(artifact["trained_layers"])
    if selected_layer not in trained_layers:
        raise ValueError("selected layer is unavailable in the model artifact")
    if trained_layers == ("core",):
        assembly = assemble_round21_core_development(
            publication_directory=Path(args.publication_directory),
            source_database=Path(args.source_database),
            terminal_transport_manifest=transport,
        )
    else:
        sidecar_database = str(args.sidecar_database or "").strip()
        sidecar_manifest_path = str(args.sidecar_terminal_manifest or "").strip()
        if not sidecar_database or not sidecar_manifest_path:
            raise ValueError(
                "matched model artifacts require the exact sidecar database "
                "and terminal manifest"
            )
        sidecar_terminal = load_round21_sidecar_terminal_manifest(
            Path(sidecar_manifest_path)
        )
        assembly = assemble_round21_matched_development(
            publication_directory=Path(args.publication_directory),
            source_database=Path(args.source_database),
            terminal_transport_manifest=transport,
            sidecar_database=Path(sidecar_database),
            sidecar_terminal_manifest=sidecar_terminal,
        )
    return transport, artifact, assembly, selected_layer


def command_polymarket_round21_evaluate_development(args: argparse.Namespace) -> int:
    output = Path(args.output)

    def progress(stage: str, payload: dict[str, object]) -> None:
        count = int(payload.get("condition_count", 0))
        if stage == "condition_replayed" and count != 1 and count % 100:
            return
        details = " ".join(f"{key}={value}" for key, value in payload.items())
        print(f"Round 21 development economics: {stage} {details}", file=sys.stderr)

    try:
        if output.exists() or output.is_symlink():
            raise ValueError("development-economic output already exists")
        transport, artifact, assembly, selected_layer = (
            _load_round21_development_inputs(args)
        )
        result = replay_round21_development_economics(
            source_database=Path(args.source_database),
            terminal_transport_manifest=transport,
            partition_policy=assembly.partition_policy,
            development_panels=(
                assembly.train,
                assembly.tune_calibration,
                assembly.tune_selection,
            ),
            development_model_artifact=artifact,
            core_publication_manifest_sha256=assembly.publication_manifest_sha256,
            selected_population_layer=selected_layer,
            initial_capital_quote=args.initial_capital_quote,
            minimum_edge_per_share=args.minimum_edge_per_share,
            builder_taker_fee_bps=args.builder_taker_fee_bps,
            progress=progress,
        )
        payload = result.asdict()
        write_json_atomic(output, payload, indent=2, sort_keys=True)
        file_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    except (duckdb.Error, OSError, RuntimeError, ValueError) as exc:
        print(
            "polymarket-round21-evaluate-development failed: "
            f"{exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "Round 21 development economics complete: "
            f"result={result.result_sha256} file={file_sha256} "
            f"conditions={result.source_condition_count} "
            f"all_selected_gates={str(result.development_gate_passed).lower()} "
            "profitability_claim=false authority=false"
        )
    return 0


def command_polymarket_round21_ai_development(args: argparse.Namespace) -> int:
    output = Path(args.output)
    source_database = Path(args.source_database)
    cache_database = Path(args.ai_cache_database)
    sealed_requested = bool(args.acknowledge_one_use_test_access)
    sealed_paths = {
        "one-use store": args.one_use_store,
        "pretest output": args.pretest_output,
        "claim output": args.claim_output,
        "sealed output": args.sealed_output,
    }

    def progress(stage: str, payload: dict[str, object]) -> None:
        condition_count = int(payload.get("condition_count", 0))
        case = int(payload.get("case", 0))
        if (
            stage == "condition_replayed"
            and condition_count != 1
            and condition_count % 100
        ):
            return
        if stage == "polymarket_ai_veto" and case != 1 and case % 10:
            return
        details = " ".join(f"{key}={value}" for key, value in payload.items())
        print(f"Round 21 development AI: {stage} {details}", file=sys.stderr)

    try:
        if output.exists() or output.is_symlink():
            raise ValueError("development-AI output already exists")
        if sealed_requested != all(
            value is not None for value in sealed_paths.values()
        ):
            raise ValueError(
                "one-use sealed evaluation requires the acknowledgement, store, "
                "pretest, claim, and sealed output together"
            )
        if not sealed_requested and any(
            value is not None for value in sealed_paths.values()
        ):
            raise ValueError(
                "sealed evaluation paths require --acknowledge-one-use-test-access"
            )
        if sealed_requested:
            for name, value in sealed_paths.items():
                path = Path(str(value))
                if path.exists() or path.is_symlink():
                    raise ValueError(f"{name} already exists")
        if source_database.resolve() == cache_database.resolve():
            raise ValueError("AI cache database must be separate from source evidence")
        risk_path = Path(args.risk_benchmark_evidence)
        if not risk_path.is_file():
            raise ValueError("AI risk benchmark evidence is unavailable")
        risk_sha256 = hashlib.sha256(risk_path.read_bytes()).hexdigest()
        transport, artifact, assembly, selected_layer = (
            _load_round21_development_inputs(args)
        )
        expected_digests = {
            "qwen3.5:9b": str(args.qwen3_5_9b_digest).strip().lower(),
            "fin-r1:8b": str(args.fin_r1_8b_digest).strip().lower(),
            "fino1:8b": str(args.fino1_8b_digest).strip().lower(),
        }
        configs = tuple(
            PolymarketAIVetoConfig(
                model=model,
                base_url=str(args.ollama_url),
                timeout_seconds=float(args.ai_timeout_seconds),
                minimum_approval_confidence=float(args.ai_minimum_confidence),
                maximum_advisory_latency_seconds=float(args.ai_maximum_latency_seconds),
            )
            for model in POLYMARKET_ROUND21_AI_CANDIDATES
        )
        with PolymarketEvidenceStore(
            cache_database,
            memory_limit="512MB",
            threads=1,
        ) as cache_store:
            result = run_round21_development_ai_program(
                source_database=source_database,
                terminal_transport_manifest=transport,
                partition_policy=assembly.partition_policy,
                development_panels=(
                    assembly.train,
                    assembly.tune_calibration,
                    assembly.tune_selection,
                ),
                development_model_artifact=artifact,
                core_publication_manifest_sha256=(assembly.publication_manifest_sha256),
                selected_population_layer=selected_layer,
                risk_benchmark_evidence_sha256=risk_sha256,
                configs=configs,
                expected_model_digests=expected_digests,
                cache_store=cache_store,
                initial_capital_quote=args.initial_capital_quote,
                minimum_edge_per_share=args.minimum_edge_per_share,
                builder_taker_fee_bps=args.builder_taker_fee_bps,
                progress=progress,
            )
        pretest = None
        claim = None
        nominated_report = None
        nominated_comparison = None
        if sealed_requested:
            optional_terminal_sha256 = None
            if selected_layer != "core":
                sidecar_terminal = load_round21_sidecar_terminal_manifest(
                    args.sidecar_terminal_manifest
                )
                optional_terminal_sha256 = str(sidecar_terminal["manifest_sha256"])
            pretest = build_round21_pretest_manifest(
                args.repository,
                selected_population_layer=selected_layer,
                core_corpus_publication_directory=args.publication_directory,
                optional_campaign_terminal_sha256=optional_terminal_sha256,
                development_model_artifact=artifact,
                development_economic_matrix=result.economic_result.selected_matrix,
                development_optional_comparison=(
                    result.economic_result.optional_comparison
                ),
                development_ai_selection=result.candidate_selection,
            )
            claim = create_round21_one_use_claim(pretest)
            nominee = result.candidate_selection.nominated_model
            if nominee is not None:
                nominated_report = next(
                    value
                    for value in result.benchmark_result.reports
                    if value.config.model == nominee
                )
                nominated_comparison = next(
                    value
                    for value in result.replay_result.comparisons
                    if value.model == nominee
                )
        write_json_atomic(output, result.asdict(), indent=2, sort_keys=True)
        file_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
        sealed_outcome = None
        sealed_file_sha256 = None
        if sealed_requested:
            if pretest is None or claim is None:
                raise AssertionError("Round 21 sealed pretest is unavailable")
            write_json_atomic(
                Path(str(args.pretest_output)),
                pretest.asdict(),
                indent=2,
                sort_keys=True,
            )
            write_json_atomic(
                Path(str(args.claim_output)),
                claim.asdict(),
                indent=2,
                sort_keys=True,
            )
            sidecar_terminal = (
                None
                if selected_layer == "core"
                else load_round21_sidecar_terminal_manifest(
                    args.sidecar_terminal_manifest
                )
            )
            with PolymarketEvidenceStore(
                cache_database,
                memory_limit="512MB",
                threads=1,
            ) as cache_store:
                sealed_outcome = evaluate_round21_terminal_sealed_once(
                    store_path=Path(str(args.one_use_store)),
                    claim=claim,
                    pretest=pretest,
                    publication_directory=args.publication_directory,
                    source_database=source_database,
                    terminal_transport_manifest=transport,
                    development_model_artifact=artifact,
                    development_ai_report=nominated_report,
                    development_ai_comparison=nominated_comparison,
                    sidecar_database=(
                        None if selected_layer == "core" else args.sidecar_database
                    ),
                    sidecar_terminal_manifest=sidecar_terminal,
                    initial_capital_quote=args.initial_capital_quote,
                    minimum_edge_per_share=args.minimum_edge_per_share,
                    builder_taker_fee_bps=args.builder_taker_fee_bps,
                    cache_store=cache_store,
                    progress=progress,
                )
            sealed_payload = build_round21_sealed_result_bundle(sealed_outcome.result)
            sealed_path = Path(str(args.sealed_output))
            write_json_atomic(
                sealed_path,
                sealed_payload,
                indent=2,
                sort_keys=True,
            )
            sealed_file_sha256 = hashlib.sha256(sealed_path.read_bytes()).hexdigest()
    except (duckdb.Error, OSError, RuntimeError, ValueError) as exc:
        print(
            "polymarket-round21-ai-development failed: "
            f"{exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    summary = {
        "schema_version": "polymarket-round21-ai-development-operator-v1",
        "result_sha256": result.result_sha256,
        "artifact_file_sha256": file_sha256,
        "condition_count": result.economic_result.source_condition_count,
        "candidate_count": len(result.replay_result.comparisons),
        "qualified_candidate_count": (
            result.candidate_selection.qualified_candidate_count
        ),
        "nominated_model": result.candidate_selection.nominated_model,
        "target_accessed": sealed_requested,
        "sealed_candidate_accepted": (
            None if sealed_outcome is None else sealed_outcome.result.candidate_accepted
        ),
        "sealed_ai_enabled_candidate": (
            None
            if sealed_outcome is None
            else sealed_outcome.result.ai_enabled_candidate
        ),
        "sealed_artifact_file_sha256": sealed_file_sha256,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "Round 21 development AI complete: "
            f"result={result.result_sha256} file={file_sha256} "
            f"conditions={result.economic_result.source_condition_count} "
            f"qualified={result.candidate_selection.qualified_candidate_count} "
            f"nominee={result.candidate_selection.nominated_model or 'none'} "
            f"sealed={str(sealed_requested).lower()} "
            f"accepted={str(None if sealed_outcome is None else sealed_outcome.result.candidate_accepted).lower()} "
            "profitability_claim=false authority=false"
        )
    return 0


def command_polymarket_round21_recover_sealed(args: argparse.Namespace) -> int:
    store = Path(args.one_use_store)
    output = Path(args.output)
    try:
        if store.is_symlink() or not store.is_file():
            raise ValueError("completed one-use store is unavailable")
        if output.exists() or output.is_symlink():
            raise ValueError("sealed recovery output already exists")
        bundle = load_round21_completed_sealed_bundle(store)
        write_json_atomic(output, bundle, indent=2, sort_keys=True)
        file_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
        print(
            "polymarket-round21-recover-sealed failed: "
            f"{exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    result = bundle["result"]
    summary = {
        "schema_version": "polymarket-round21-sealed-recovery-v1",
        "result_sha256": result["result_sha256"],
        "candidate_accepted": result["candidate_accepted"],
        "ai_enabled_candidate": result["ai_enabled_candidate"],
        "artifact_file_sha256": file_sha256,
        "test_access_reopened": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "Round 21 sealed result recovered: "
            f"result={summary['result_sha256']} file={file_sha256} "
            "test_access_reopened=false authority=false"
        )
    return 0


def _required_shadow_argument(value: object, *, name: str) -> str:
    selected = str(value or "").strip()
    if not selected:
        raise ValueError(f"{name} is required for shadow run")
    return selected


def _shadow_payload(audit: object, *, action: str) -> dict[str, object]:
    run = audit.run
    terminal = audit.terminal
    return {
        "schema_version": "polymarket-round21-shadow-operator-result-v1",
        "action": action,
        "venue": "polymarket",
        "asset": "BTC",
        "market_variant": "fiveminute",
        "run_id": run.run_id,
        "source_model_artifact_sha256": run.source_model_artifact_sha256,
        "sealed_result_sha256": run.sealed_result_sha256,
        "population_layer": run.population_layer,
        "started_at_ms": run.started_at_ms,
        "run_sha256": run.run_sha256,
        "prediction_count": audit.prediction_count,
        "observed_count": audit.observed_count,
        "abstention_count": audit.abstention_count,
        "last_record_sha256": audit.last_record_sha256,
        "terminal": {
            "status": terminal.status,
            "reason": terminal.reason,
            "finished_at_ms": terminal.finished_at_ms,
            "terminal_sha256": terminal.terminal_sha256,
        },
        "integrity_passed": audit.integrity_passed,
        "target_accessed": False,
        "credentials_used": False,
        "account_connected": False,
        "binance_execution_connected": False,
        "grants_execution_authority": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }


def _render_shadow(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    terminal = payload["terminal"]
    print(
        "Round 21 no-order shadow: "
        f"action={payload['action']} run={payload['run_id']} "
        f"status={terminal['status']} predictions={payload['prediction_count']} "
        "authority=false"
    )


def command_polymarket_round21_shadow(args: argparse.Namespace) -> int:
    action = str(args.action)
    database = Path(args.shadow_database)
    public_client: PolymarketPublicClient | None = None
    stack = None
    try:
        if action == "audit":
            if not database.is_file():
                raise ValueError("shadow database is unavailable for audit")
            with Round21ProspectiveShadowStore(database) as store:
                run_id = str(args.run_id or "").strip()
                if not run_id:
                    run_ids = store.run_ids()
                    if len(run_ids) != 1:
                        raise ValueError(
                            "run-id is required unless exactly one shadow run exists"
                        )
                    run_id = run_ids[0]
                audit = store.audit_run(run_id)
        else:
            duration = float(args.duration_seconds)
            if not 5 <= duration <= 2_592_000:
                raise ValueError("duration-seconds must lie in [5, 2592000]")
            public_client = PolymarketPublicClient()
            stack = build_polymarket_round21_shadow_runtime_stack(
                public_client=public_client,
                model_artifact_path=_required_shadow_argument(
                    args.model_artifact,
                    name="model-artifact",
                ),
                expected_model_file_sha256=_required_shadow_argument(
                    args.model_file_sha256,
                    name="model-file-sha256",
                ),
                evaluation_report_path=_required_shadow_argument(
                    args.evaluation_report,
                    name="evaluation-report",
                ),
                expected_evaluation_file_sha256=_required_shadow_argument(
                    args.evaluation_file_sha256,
                    name="evaluation-file-sha256",
                ),
                shadow_database_path=database,
                run_id=args.run_id,
                discovery_interval_seconds=float(args.discovery_seconds),
                poll_interval_seconds=float(args.poll_seconds),
                queue_capacity=int(args.queue_capacity),
            )
            scheduled_end_ms = int(time.time_ns() // 1_000_000 + duration * 1_000)
            audit = asyncio.run(
                stack.runner.run(
                    asyncio.Event(),
                    scheduled_end_ms=scheduled_end_ms,
                )
            )
        payload = _shadow_payload(audit, action=action)
        _render_shadow(payload, as_json=bool(args.json))
        return 0
    except KeyboardInterrupt:
        print("polymarket-round21-shadow interrupted", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError) as exc:
        print(
            f"polymarket-round21-shadow failed: {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    finally:
        if stack is not None:
            stack.close()
        if public_client is not None:
            public_client.session.close()


__all__ = [
    "command_polymarket_round21_ablate_basis",
    "command_polymarket_round21_ai_development",
    "command_polymarket_round21_corpus",
    "command_polymarket_round21_fit_core",
    "command_polymarket_round21_recover_sealed",
    "command_polymarket_round21_shadow",
    "command_polymarket_round21_terminal",
    "register_polymarket_round21_commands",
]
