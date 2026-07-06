"""Benchmark local AI models for finance-risk review suitability."""

from __future__ import annotations

import json
import math
import shutil
import subprocess  # nosec B404
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .ai_runtime import estimate_model_parameters_b
from .storage import write_json_atomic

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
PostJson = Callable[[str, Mapping[str, object], float], object]

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["approve", "veto", "cooldown", "human_review"]},
        "risk_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string"},
        "concerns": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "required_actions": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
    },
    "required": ["action", "risk_score", "confidence", "rationale", "concerns", "required_actions"],
}


@dataclass(frozen=True)
class FinanceAIModelCandidate:
    model: str
    family: str
    local_priority: int
    finance_specialized: bool
    reasoning_or_risk_review: bool
    notes: str

    @property
    def model_parameters_b(self) -> float | None:
        return estimate_model_parameters_b(self.model)

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["model_parameters_b"] = self.model_parameters_b
        return payload


@dataclass(frozen=True)
class FinanceAITestCase:
    name: str
    prompt_payload: dict[str, object]
    expected_action: str
    min_risk_score: float
    max_risk_score: float
    must_mention: tuple[str, ...] = ()

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AIModelBenchmarkResult:
    model: str
    installed: bool
    model_parameters_b: float | None
    score: float
    passed: bool
    valid_json_cases: int
    action_match_cases: int
    average_latency_seconds: float
    failures: tuple[str, ...]
    case_results: tuple[dict[str, object], ...]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AIModelBenchmarkReport:
    generated_at_ms: int
    base_url: str
    selected_model: str | None
    minimum_score: float
    candidates: tuple[dict[str, object], ...]
    tests: tuple[dict[str, object], ...]
    results: tuple[AIModelBenchmarkResult, ...]

    @property
    def passed(self) -> bool:
        return self.selected_model is not None

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


def finance_ai_candidates() -> tuple[FinanceAIModelCandidate, ...]:
    """Return the curated small/local finance AI candidate order."""

    return (
        FinanceAIModelCandidate(
            model="qwen3:8b",
            family="qwen",
            local_priority=100,
            finance_specialized=False,
            reasoning_or_risk_review=True,
            notes="strong local structured-output/risk-review baseline; installed on this host",
        ),
        FinanceAIModelCandidate(
            model="deepseek-r1:8b",
            family="deepseek-r1",
            local_priority=90,
            finance_specialized=False,
            reasoning_or_risk_review=True,
            notes="reasoning-oriented local second opinion; may be slower",
        ),
        FinanceAIModelCandidate(
            model="gemma4:e4b",
            family="gemma",
            local_priority=70,
            finance_specialized=False,
            reasoning_or_risk_review=True,
            notes="smaller local fallback for latency-sensitive risk review",
        ),
        FinanceAIModelCandidate(
            model="falcon3:latest",
            family="falcon",
            local_priority=60,
            finance_specialized=False,
            reasoning_or_risk_review=True,
            notes="general local fallback; must prove structured risk decisions",
        ),
        FinanceAIModelCandidate(
            model="qwen3:4b",
            family="qwen",
            local_priority=50,
            finance_specialized=False,
            reasoning_or_risk_review=True,
            notes="lower-latency fallback; below preferred 7B/8B class",
        ),
        FinanceAIModelCandidate(
            model="DragonLLM/Qwen-Open-Finance-R-8B",
            family="qwen-open-finance",
            local_priority=45,
            finance_specialized=True,
            reasoning_or_risk_review=True,
            notes="finance-specialized 8B research candidate; must be locally served and benchmarked before use",
        ),
        FinanceAIModelCandidate(
            model="FinGPT/fingpt-mt_llama2-7b_lora",
            family="fingpt",
            local_priority=40,
            finance_specialized=True,
            reasoning_or_risk_review=False,
            notes="finance-tuned sentiment/classification candidate when served locally",
        ),
        FinanceAIModelCandidate(
            model="TheFinAI/finma-7b-full",
            family="finma",
            local_priority=35,
            finance_specialized=True,
            reasoning_or_risk_review=False,
            notes="finance-tuned candidate; best treated as a specialist signal, not sole risk reviewer",
        ),
    )


def default_finance_ai_test_cases() -> tuple[FinanceAITestCase, ...]:
    return (
        FinanceAITestCase(
            name="veto_failed_ai_uplift",
            expected_action="veto",
            min_risk_score=0.75,
            max_risk_score=1.0,
            must_mention=("uplift", "drawdown"),
            prompt_payload={
                "scenario": "accepted deterministic model but AI overlay underperformed",
                "accepted_symbols": ["BTCUSDT", "ETHUSDT"],
                "data_coverage": {"interval": "1s", "coverage_ratio": 1.0, "gap_count": 0},
                "portfolio": {"accepted": True, "cvar95": -0.012, "max_drawdown": 0.04},
                "baseline_ml": {"realized_pnl": 120.0, "max_drawdown": 0.03, "closed_trades": 48},
                "ai_overlay": {"realized_pnl": 80.0, "max_drawdown": 0.07, "closed_trades": 48},
                "ai_uplift": {"accepted": False, "reasons": ["ai_pnl_not_above_baseline", "ai_drawdown_worse_than_baseline"]},
            },
        ),
        FinanceAITestCase(
            name="cooldown_unpredictable_low_liquidity",
            expected_action="cooldown",
            min_risk_score=0.55,
            max_risk_score=0.90,
            must_mention=("liquidity", "cooldown"),
            prompt_payload={
                "scenario": "model is profitable but current market microstructure is unsafe",
                "accepted_symbols": ["BTCUSDT"],
                "regime": {"dominant": "random_walk", "confidence": 0.41, "unpredictability_score": 0.82},
                "liquidity": {"spread_bps": 28.0, "trade_count_24h": 4200, "depth_usdt": 12000},
                "risk": {"max_drawdown_limit": 0.10, "current_drawdown": 0.01},
                "data_coverage": {"interval": "1s", "coverage_ratio": 1.0, "gap_count": 0},
            },
        ),
        FinanceAITestCase(
            name="approve_clean_positive_uplift",
            expected_action="approve",
            min_risk_score=0.0,
            max_risk_score=0.35,
            must_mention=("risk",),
            prompt_payload={
                "scenario": "all deterministic gates passed and AI overlay improves holdout",
                "accepted_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                "data_coverage": {"interval": "1s", "coverage_ratio": 1.0, "gap_count": 0, "years": 3.0},
                "portfolio": {"accepted": True, "cvar95": -0.006, "max_drawdown": 0.025, "correlation_adjusted_symbols": 2.4},
                "baseline_ml": {"realized_pnl": 220.0, "max_drawdown": 0.035, "closed_trades": 160, "profit_factor": 1.42},
                "ai_overlay": {"realized_pnl": 260.0, "max_drawdown": 0.032, "closed_trades": 164, "profit_factor": 1.58},
                "ai_uplift": {"accepted": True, "positive_delta_rate": 0.68, "sign_test_p_value": 0.021},
            },
        ),
        FinanceAITestCase(
            name="veto_missing_data_integrity",
            expected_action="veto",
            min_risk_score=0.80,
            max_risk_score=1.0,
            must_mention=("data", "gap"),
            prompt_payload={
                "scenario": "profitable looking model has missing second-level data",
                "accepted_symbols": ["BTCUSDT"],
                "data_coverage": {"interval": "1s", "coverage_ratio": 0.91, "gap_count": 3810, "years": 0.02},
                "baseline_ml": {"realized_pnl": 500.0, "max_drawdown": 0.01, "closed_trades": 3},
                "ai_overlay": {"realized_pnl": 540.0, "max_drawdown": 0.01, "closed_trades": 3},
            },
        ),
    )


def installed_ollama_models() -> tuple[str, ...]:
    exe = shutil.which("ollama")
    if not exe:
        return ()
    try:
        completed = subprocess.run(  # nosec B603
            [exe, "list"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    names: list[str] = []
    for index, line in enumerate(completed.stdout.splitlines()):
        if index == 0 and line.upper().startswith("NAME"):
            continue
        parts = line.split()
        if parts:
            names.append(parts[0])
    return tuple(dict.fromkeys(names))


def _post_json(url: str, payload: Mapping[str, object], timeout: float) -> object:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "simple-ai-trading-ai-benchmark/0.1"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc)) from exc


def _response_text(payload: object) -> str:
    if not isinstance(payload, Mapping):
        return ""
    message = payload.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str):
            return content
    response = payload.get("response")
    return response if isinstance(response, str) else ""


def _json_mapping_from_text(text: str) -> Mapping[str, object] | None:
    clean = str(text or "").strip()
    if not clean:
        return None
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(clean[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, Mapping) else None


def _bounded_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(0.0, min(1.0, parsed))


def _prompt(case: FinanceAITestCase) -> str:
    payload = json.dumps(
        {"case_name": case.name, "evidence": case.prompt_payload},
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "You are an institutional autonomous day-trading risk reviewer. "
        "Use only the supplied case. Return JSON only. "
        "Approve only if data integrity, deterministic risk gates, and AI-vs-ML uplift are favorable. "
        "Use cooldown when the model may be valid but current market liquidity/regime is unsafe. "
        "Use veto when evidence is missing, AI underperforms ML, drawdown worsens, liquidation risk appears, "
        "or data coverage is not trustworthy. "
        f"CASE={payload}"
    )


def _case_score(case: FinanceAITestCase, parsed: Mapping[str, object] | None, latency_seconds: float) -> dict[str, object]:
    if parsed is None:
        return {
            "name": case.name,
            "score": 0.0,
            "valid_json": False,
            "action_match": False,
            "latency_seconds": float(latency_seconds),
            "failure": "invalid_json",
        }
    action = str(parsed.get("action") or "").strip().lower()
    risk_score = _bounded_float(parsed.get("risk_score"))
    confidence = _bounded_float(parsed.get("confidence"))
    rationale = str(parsed.get("rationale") or "")
    concerns = parsed.get("concerns") if isinstance(parsed.get("concerns"), list) else []
    required_actions = parsed.get("required_actions") if isinstance(parsed.get("required_actions"), list) else []
    searchable = " ".join(
        [rationale, *(str(item) for item in concerns), *(str(item) for item in required_actions)]
    ).lower()
    action_match = action == case.expected_action
    risk_match = case.min_risk_score <= risk_score <= case.max_risk_score
    mention_hits = sum(1 for term in case.must_mention if term.lower() in searchable)
    mention_score = mention_hits / max(1, len(case.must_mention))
    score = (
        0.35
        + (0.30 if action_match else 0.0)
        + (0.15 if risk_match else 0.0)
        + 0.10 * mention_score
        + 0.05 * confidence
        + (0.05 if latency_seconds <= 30.0 else 0.0)
    )
    failures: list[str] = []
    if not action_match:
        failures.append(f"action={action or 'missing'}!={case.expected_action}")
    if not risk_match:
        failures.append(f"risk_score={risk_score:.3f} not in [{case.min_risk_score:.3f},{case.max_risk_score:.3f}]")
    missing_terms = [term for term in case.must_mention if term.lower() not in searchable]
    if missing_terms:
        failures.append("missing_terms=" + ",".join(missing_terms))
    return {
        "name": case.name,
        "score": float(max(0.0, min(1.0, score))),
        "valid_json": True,
        "action": action,
        "expected_action": case.expected_action,
        "action_match": action_match,
        "risk_score": risk_score,
        "confidence": confidence,
        "latency_seconds": float(latency_seconds),
        "failure": "; ".join(failures),
    }


def _resolve_benchmark_models(models: Sequence[str] | None, installed: Sequence[str]) -> tuple[str, ...]:
    explicit = [str(model).strip() for model in (models or ()) if str(model).strip()]
    if explicit:
        return tuple(dict.fromkeys(explicit))
    installed_set = {name.lower() for name in installed}
    candidates = [candidate.model for candidate in finance_ai_candidates() if candidate.model.lower() in installed_set]
    return tuple(candidates or [candidate.model for candidate in finance_ai_candidates()[:4]])


def benchmark_finance_ai_models(
    *,
    models: Sequence[str] | None = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: float = 20.0,
    minimum_score: float = 0.78,
    post_json: PostJson = _post_json,
    installed_models: Sequence[str] | None = None,
) -> AIModelBenchmarkReport:
    installed = tuple(installed_models) if installed_models is not None else installed_ollama_models()
    selected_models = _resolve_benchmark_models(models, installed)
    installed_set = {name.lower() for name in installed}
    endpoint = f"{str(base_url or DEFAULT_OLLAMA_URL).rstrip('/')}/api/chat"
    tests = default_finance_ai_test_cases()
    results: list[AIModelBenchmarkResult] = []
    for model in selected_models:
        case_results: list[dict[str, object]] = []
        failures: list[str] = []
        for case in tests:
            request = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Do not think. Return only valid JSON matching the supplied schema.",
                    },
                    {"role": "user", "content": _prompt(case)},
                ],
                "stream": False,
                "think": False,
                "format": _RESPONSE_SCHEMA,
                "options": {"temperature": 0.0, "num_predict": 512},
            }
            started = time.monotonic()
            parsed: Mapping[str, object] | None = None
            try:
                response = post_json(endpoint, request, timeout_seconds)
                parsed = _json_mapping_from_text(_response_text(response))
            except Exception as exc:  # noqa: BLE001 - benchmark records provider failures
                failures.append(f"{case.name}: provider_error={exc}")
            latency = max(0.0, time.monotonic() - started)
            result = _case_score(case, parsed, latency)
            if result.get("failure"):
                failures.append(f"{case.name}: {result['failure']}")
            case_results.append(result)
        valid_json_cases = sum(1 for item in case_results if item.get("valid_json") is True)
        action_match_cases = sum(1 for item in case_results if item.get("action_match") is True)
        score = sum(float(item.get("score") or 0.0) for item in case_results) / max(1, len(case_results))
        average_latency = sum(float(item.get("latency_seconds") or 0.0) for item in case_results) / max(1, len(case_results))
        parameters_b = estimate_model_parameters_b(model)
        if parameters_b is not None and parameters_b < 2.0:
            failures.append(f"model_parameters_b={parameters_b:.2f}<2.00")
        passed = score >= float(minimum_score) and action_match_cases == len(tests) and valid_json_cases == len(tests)
        if parameters_b is not None and parameters_b < 2.0:
            passed = False
        results.append(
            AIModelBenchmarkResult(
                model=model,
                installed=model.lower() in installed_set,
                model_parameters_b=parameters_b,
                score=float(score),
                passed=bool(passed),
                valid_json_cases=int(valid_json_cases),
                action_match_cases=int(action_match_cases),
                average_latency_seconds=float(average_latency),
                failures=tuple(dict.fromkeys(failures)),
                case_results=tuple(case_results),
            )
        )
    ranked = tuple(
        sorted(
            results,
            key=lambda item: (
                item.passed,
                item.score,
                -item.average_latency_seconds,
                item.model_parameters_b or 0.0,
            ),
            reverse=True,
        )
    )
    selected = next((item.model for item in ranked if item.passed), None)
    return AIModelBenchmarkReport(
        generated_at_ms=int(time.time() * 1000),
        base_url=str(base_url or DEFAULT_OLLAMA_URL),
        selected_model=selected,
        minimum_score=float(minimum_score),
        candidates=tuple(candidate.asdict() for candidate in finance_ai_candidates()),
        tests=tuple(test.asdict() for test in tests),
        results=ranked,
    )


def write_benchmark_report(report: AIModelBenchmarkReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    write_json_atomic(path, report.asdict(), indent=2, sort_keys=True)
    return path


__all__ = [
    "AIModelBenchmarkReport",
    "AIModelBenchmarkResult",
    "FinanceAIModelCandidate",
    "FinanceAITestCase",
    "benchmark_finance_ai_models",
    "default_finance_ai_test_cases",
    "finance_ai_candidates",
    "installed_ollama_models",
    "write_benchmark_report",
]
