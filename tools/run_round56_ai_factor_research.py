"""Generate Round 56 action-conditioned factor programs with local AI models."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Mapping
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.ai_factor_programs import (  # noqa: E402
    ALLOWED_FUNCTIONS,
    ActionConditionedFactorProgram,
    parse_action_conditioned_factor_response_ledger,
)
from simple_ai_trading.paired_action_lightgbm import (  # noqa: E402
    action_conditioned_feature_names,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 56
DESIGN_SCHEMA = "round-056-paired-action-distributional-design-v1"
REPORT_SCHEMA = "round-056-ai-factor-research-report-v1"
LEDGER_SCHEMA = "round-056-action-conditioned-factor-program-ledger-v1"
EXPECTED_DATASET_SHA256 = (
    "13086282510f69862552dfc7d85839d6910bb5cfd3e67b69f6c879ccd1c8837f"
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Round 56 AI Git identity command failed") from exc


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root is not an object")
    return value


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 56 design")
    canonical = dict(design)
    claimed = str(canonical.pop("design_sha256", ""))
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or design.get("status") != "frozen_development_only"
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 56 design identity is invalid")
    contract = design.get("ai_factor_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Round 56 AI contract is absent")
    if (
        contract.get("models") != ["qwen3.5:9b", "fin-r1:8b", "fino1:8b"]
        or contract.get("programs_requested_per_model") != 3
        or contract.get("temperature") != 0.0
        or contract.get("seed") != 5600
        or contract.get(
            "market_values_timestamps_or_outcomes_visible_to_language_model"
        )
        is not False
        or contract.get("allowed_functions") != list(ALLOWED_FUNCTIONS)
        or contract.get("order_authority") is not False
        or contract.get("position_sizing_authority") is not False
        or contract.get("risk_gate_override") is not False
    ):
        raise ValueError("Round 56 AI implementation and frozen contract differ")
    return design, claimed


def _load_feature_metadata(
    path: Path,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, object]]:
    metadata_path = path / "metadata.json"
    metadata = _read_object(metadata_path, "Round 45 derived metadata")
    names = metadata.get("feature_names")
    if (
        metadata.get("schema_version") != "round-045-derived-dataset-cache-v1"
        or metadata.get("dataset_sha256") != EXPECTED_DATASET_SHA256
        or not isinstance(names, list)
        or len(names) != 71
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("Round 56 AI feature metadata is invalid")
    paired = action_conditioned_feature_names(tuple(names))
    visible = tuple(name for name in paired if name != "action_sign")
    if len(paired) != 72 or len(visible) != 71:
        raise RuntimeError("Round 56 action-conditioned feature count drifted")
    return tuple(names), visible, {
        "path": str(metadata_path.resolve()),
        "bytes": metadata_path.stat().st_size,
        "file_sha256": _file_sha256(metadata_path),
        "dataset_sha256": metadata["dataset_sha256"],
        "source_feature_count": len(names),
        "numerical_model_feature_count": len(paired),
        "language_model_visible_feature_count": len(visible),
        "language_model_visible_feature_names_sha256": _canonical_sha256(
            list(visible)
        ),
        "action_sign_visible_to_language_model": False,
    }


def _post_json(
    url: str,
    payload: Mapping[str, object],
    timeout: float,
) -> dict[str, object]:
    body = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
    message = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(message, timeout=timeout) as response:  # noqa: S310
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, error.URLError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"local Ollama request failed: {url}") from exc
    if not isinstance(result, dict):
        raise ValueError("local Ollama response root is not an object")
    return result


def _get_json(url: str, timeout: float) -> dict[str, object]:
    message = request.Request(url, headers={"Accept": "application/json"})
    try:
        with request.urlopen(message, timeout=timeout) as response:  # noqa: S310
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, error.URLError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"local Ollama request failed: {url}") from exc
    if not isinstance(result, dict):
        raise ValueError("local Ollama response root is not an object")
    return result


def _response_text(payload: Mapping[str, object]) -> str:
    message = payload.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("Ollama factor response has no message content")
    return str(message["content"])


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def _parameter_billions(inventory_row: Mapping[str, object]) -> float:
    details = inventory_row.get("details")
    if not isinstance(details, Mapping):
        raise ValueError("Ollama model inventory has no details")
    raw = str(details.get("parameter_size", "")).strip().upper()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)B", raw)
    if match is None:
        raise ValueError("Ollama model parameter size is not auditable")
    value = float(match.group(1))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Ollama model parameter size is invalid")
    return value


def _schema(factors: int) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["factors"],
        "properties": {
            "factors": {
                "type": "array",
                "minItems": factors,
                "maxItems": factors,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "name",
                        "expression",
                        "mechanism",
                        "failure_mode",
                        "expected_horizon",
                        "action_symmetry",
                    ],
                    "properties": {
                        "name": {
                            "type": "string",
                            "pattern": "^[a-z][a-z0-9_]{2,63}$",
                        },
                        "expression": {"type": "string"},
                        "mechanism": {"type": "string"},
                        "failure_mode": {"type": "string"},
                        "expected_horizon": {"type": "string"},
                        "action_symmetry": {"type": "string"},
                    },
                },
            }
        },
    }


def _prompt(feature_names: tuple[str, ...], factors: int) -> tuple[str, str]:
    system = (
        "You are a quantitative factor research assistant, not a trader. Produce only "
        "the requested strict JSON. You have no market values, timestamps, labels, "
        "future returns, backtest results, portfolio state, or order authority. Every "
        "factor must be causal, interpretable, dimensionally defensible, and invariant "
        "to whether the candidate row represents a long or a short after the supplied "
        "action alignment. Do not propose leverage, orders, sizes, thresholds, or "
        "performance claims."
    )
    user = (
        f"Propose exactly {factors} distinct factor expressions for a shared paired-"
        "action model of BTCUSDT, ETHUSDT, and SOLUSDT one-hour after-cost payoff. "
        "Every action_aligned_ feature is positive when its original signed quantity "
        "supports the candidate action: a long row uses the original sign and a short "
        "row uses the reflected sign. action_favorable_semivolatility measures motion "
        "with the candidate action; action_adverse_semivolatility measures motion "
        "against it. action_sign is intentionally unavailable. A valid expression must "
        "therefore mean the same thing for candidate long and candidate short rows. "
        "Expressions may use only the names and operators below. Binary operators are "
        "+, -, and *. Raw /, **, comparisons, booleans, indexing, attributes, methods, "
        "and unknown names are forbidden. Allowed functions and arity: abs(x), "
        "clip(x, lower_constant, upper_constant), maximum(x, y), minimum(x, y), "
        "safe_divide(x, y), sign(x), and tanh(x). Constants must be finite and at most "
        "10000 in absolute value. mechanism and failure_mode must each be substantive "
        "sentences of at least 20 characters. action_symmetry must explicitly explain "
        "how the expression has the same interpretation for both long and short rows, "
        "using the words long and short. expected_horizon must be one_hour. Addition, "
        "subtraction, maximum, and minimum may combine only like units. Features ending "
        "in _bps are basis points; flow, efficiency, relative-volume, liquidity, ratio, "
        "calendar, and symbol fields are dimensionless. Prefer economically distinct "
        "mechanisms involving action-aligned trend quality, order-flow confirmation, "
        "cross-asset residual behavior, volatility asymmetry, and liquidity-conditioned "
        "regimes. Avoid a naked directional prior or merely renaming one feature. "
        "Output JSON only.\n\nAvailable features:\n- "
        + "\n- ".join(feature_names)
    )
    return system, user


def _progress(path: Path, stage: str, details: Mapping[str, object]) -> None:
    write_json_atomic(
        path,
        {
            "round": ROUND,
            "stage": stage,
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "details": dict(details),
        },
    )


def _deduplicate(
    programs: tuple[ActionConditionedFactorProgram, ...],
    seen_expressions: set[str],
) -> tuple[
    tuple[ActionConditionedFactorProgram, ...],
    tuple[dict[str, object], ...],
]:
    accepted: list[ActionConditionedFactorProgram] = []
    rejected: list[dict[str, object]] = []
    for index, program in enumerate(programs):
        if program.canonical_expression in seen_expressions:
            rejected.append(
                {
                    "index": index,
                    "name": program.name,
                    "reason": "duplicate_canonical_expression_across_models",
                }
            )
            continue
        accepted.append(program)
        seen_expressions.add(program.canonical_expression)
    return tuple(accepted), tuple(rejected)


def run(arguments: argparse.Namespace) -> int:
    started = time.perf_counter()
    evidence_root = arguments.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=False)
    request_root = evidence_root / "requests"
    response_root = evidence_root / "responses"
    request_root.mkdir()
    response_root.mkdir()
    status_path = evidence_root / "status.json"
    design, design_sha = _validate_design(arguments.design.resolve())
    implementation_commit = _git("rev-parse", "HEAD")
    implementation_blobs = {
        path: _git("rev-parse", f"{implementation_commit}:{path}")
        for path in (
            "src/simple_ai_trading/ai_factor_programs.py",
            "src/simple_ai_trading/paired_action_lightgbm.py",
            "tools/run_round56_ai_factor_research.py",
        )
    }
    source_names, feature_names, metadata_evidence = _load_feature_metadata(
        arguments.derived_cache.resolve()
    )
    contract = design["ai_factor_contract"]
    assert isinstance(contract, Mapping)
    models = tuple(str(model) for model in contract["models"])
    factors = int(contract["programs_requested_per_model"])
    system_prompt, user_prompt = _prompt(feature_names, factors)
    base_url = arguments.ollama_url.rstrip("/")
    tags = _get_json(f"{base_url}/api/tags", arguments.timeout)
    inventory_rows = tags.get("models")
    if not isinstance(inventory_rows, list):
        raise ValueError("Ollama inventory has no model list")
    inventory = {
        str(row.get("name")): row
        for row in inventory_rows
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    }
    missing = [model for model in models if model not in inventory]
    if missing:
        raise ValueError(f"required local AI models are absent: {', '.join(missing)}")
    _progress(
        status_path,
        "contracts_validated",
        {
            "design_sha256": design_sha,
            "models": list(models),
            "visible_features": len(feature_names),
        },
    )

    accepted: list[ActionConditionedFactorProgram] = []
    rejected: list[dict[str, object]] = []
    model_evidence: dict[str, object] = {}
    output_schema = _schema(factors)
    seen_expressions: set[str] = set()
    for model_index, model in enumerate(models, start=1):
        _progress(
            status_path,
            "model_request",
            {"model": model, "index": model_index, "total": len(models)},
        )
        show = _post_json(
            f"{base_url}/api/show",
            {"model": model, "verbose": False},
            arguments.timeout,
        )
        payload: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "format": output_schema,
            "options": {
                "temperature": float(contract["temperature"]),
                "seed": int(contract["seed"]),
                "num_ctx": 8192,
            },
            "keep_alive": "5m",
        }
        request_path = request_root / f"{model_index:02d}-{_slug(model)}.json"
        write_json_atomic(request_path, payload)
        model_started = time.perf_counter()
        response = _post_json(
            f"{base_url}/api/chat",
            payload,
            arguments.timeout,
        )
        elapsed = time.perf_counter() - model_started
        response_path = response_root / f"{model_index:02d}-{_slug(model)}.json"
        write_json_atomic(response_path, response)
        text = _response_text(response)
        try:
            programs, program_rejections = (
                parse_action_conditioned_factor_response_ledger(
                    text,
                    model=model,
                    feature_names=feature_names,
                    maximum_factors=factors,
                )
            )
        except ValueError as exc:
            programs = ()
            program_rejections = (
                {"index": None, "name": None, "reason": str(exc)},
            )
        inventory_row = inventory[model]
        parameter_billions = _parameter_billions(inventory_row)
        parameter_eligible = parameter_billions >= float(
            contract["minimum_parameter_scale_billions"]
        )
        if not parameter_eligible:
            program_rejections = tuple(program_rejections) + tuple(
                {
                    "index": index,
                    "name": program.name,
                    "reason": "model_parameter_scale_below_frozen_minimum",
                }
                for index, program in enumerate(programs)
            )
            programs = ()
        programs, duplicate_rejections = _deduplicate(programs, seen_expressions)
        program_rejections = tuple(program_rejections) + duplicate_rejections
        accepted.extend(programs)
        rejected.extend(
            {"model": model, **dict(item)} for item in program_rejections
        )
        model_evidence[model] = {
            "inventory": dict(inventory_row),
            "parameter_billions": parameter_billions,
            "parameter_scale_eligible": parameter_eligible,
            "show_canonical_sha256": _canonical_sha256(show),
            "request_path": str(request_path.resolve()),
            "request_file_sha256": _file_sha256(request_path),
            "response_path": str(response_path.resolve()),
            "response_file_sha256": _file_sha256(response_path),
            "response_content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "elapsed_seconds": elapsed,
            "accepted_programs": len(programs),
            "rejected_programs": len(program_rejections),
            "done": response.get("done"),
            "done_reason": response.get("done_reason"),
            "load_duration_ns": response.get("load_duration"),
            "prompt_eval_count": response.get("prompt_eval_count"),
            "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
            "eval_count": response.get("eval_count"),
            "eval_duration_ns": response.get("eval_duration"),
        }
        _progress(
            status_path,
            "model_complete",
            {
                "model": model,
                "index": model_index,
                "total": len(models),
                "elapsed_seconds": elapsed,
                "accepted_programs": len(programs),
                "rejected_programs": len(program_rejections),
            },
        )

    ledger: dict[str, object] = {
        "schema_version": LEDGER_SCHEMA,
        "round": ROUND,
        "design_sha256": design_sha,
        "source_dataset_sha256": EXPECTED_DATASET_SHA256,
        "source_feature_names": list(source_names),
        "language_model_visible_feature_names": list(feature_names),
        "programs": [program.asdict() for program in accepted],
        "rejections": rejected,
        "market_values_timestamps_or_outcomes_read": False,
        "action_sign_visible_to_language_model": False,
        "order_authority": False,
        "position_sizing_authority": False,
        "risk_gate_override": False,
    }
    ledger["ledger_sha256"] = _canonical_sha256(ledger)
    ledger_path = evidence_root / "factor-program-ledger.json"
    write_json_atomic(ledger_path, ledger)
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "round": ROUND,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete" if accepted else "rejected_no_valid_programs",
        "design_sha256": design_sha,
        "implementation_commit": implementation_commit,
        "implementation_blobs": implementation_blobs,
        "feature_metadata": metadata_evidence,
        "models": model_evidence,
        "ledger_path": str(ledger_path.resolve()),
        "ledger_file_sha256": _file_sha256(ledger_path),
        "ledger_canonical_sha256": ledger["ledger_sha256"],
        "accepted_programs": len(accepted),
        "rejected_programs": len(rejected),
        "market_values_read": False,
        "timestamps_read": False,
        "outcomes_read": False,
        "trading_authority": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["report_sha256"] = _canonical_sha256(report)
    report_path = evidence_root / "report.json"
    write_json_atomic(report_path, report)
    _progress(
        status_path,
        "complete",
        {
            "accepted_programs": len(accepted),
            "rejected_programs": len(rejected),
            "report_sha256": report["report_sha256"],
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if accepted else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=ROOT
        / "docs/model-research/action-value/round-056-paired-action-distributional-design.json",
    )
    parser.add_argument("--derived-cache", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))
