"""Periodic source-quality grading over replayable telemetry."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .external_signals import DEFAULT_OLLAMA_NEWS_MODEL, DEFAULT_OLLAMA_URL, PostJson, _post_json
from .telemetry_store import SourceGrade, TradingTelemetryStore


@dataclass(frozen=True)
class SourceGradeRun:
    status: str
    db_path: str
    model: str
    window_start_ms: int
    window_end_ms: int
    graded_sources: int
    ai_status: str
    ai_latency_ms: int
    warnings: list[str]
    grades: list[SourceGrade]

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["grades"] = [grade.asdict() for grade in self.grades]
        return payload


def _now_ms() -> int:
    return int(time.time() * 1000)


def _clamp_int(value: object, low: int, high: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(low, min(high, parsed))


def _heuristic_grade(rollup: Mapping[str, object]) -> tuple[int, str]:
    samples = max(0, int(float(rollup.get("sample_count") or 0)))
    avg_confidence = max(0.0, min(1.0, float(rollup.get("avg_confidence") or 0.0)))
    avg_abs_score = max(0.0, min(1.0, float(rollup.get("avg_abs_score") or 0.0)))
    raw_records = max(0, int(float(rollup.get("raw_records") or 0)))
    component_records = max(0, int(float(rollup.get("component_records") or 0)))
    sample_bonus = min(3.0, math.log(samples + 1.0) * 1.15)
    evidence_bonus = 1.0 if raw_records and component_records else (0.4 if raw_records or component_records else 0.0)
    grade = round(1.5 + sample_bonus + avg_confidence * 3.0 + avg_abs_score * 2.0 + evidence_bonus)
    bounded = max(0, min(10, int(grade)))
    reason = (
        f"samples={samples} confidence={avg_confidence:.2f} "
        f"actionability={avg_abs_score:.2f} raw={raw_records}"
    )
    return bounded, reason


def _grade_prompt(rollups: list[dict[str, object]]) -> str:
    compact = [
        (
            f"{row['source']}|{row['horizon']}|s={row['sample_count']}|"
            f"score={float(row['avg_score']):+.2f}|abs={float(row['avg_abs_score']):.2f}|"
            f"conf={float(row['avg_confidence']):.2f}|raw={row['raw_records']}|"
            f"comp={row['component_records']}"
        )
        for row in rollups[:120]
    ]
    return (
        "Grade BTCUSDC data sources 0-10. Higher=timely, replayable, actionable, consistent. "
        "Return JSON only: {\"grades\":{\"source|horizon\":grade,...}}.\n"
        + "\n".join(compact)
    )


def _json_mapping_from_text(text: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, Mapping):
        raise ValueError("grade response was not a JSON object")
    return parsed


def _ai_grade_batch(
    rollups: list[dict[str, object]],
    *,
    model: str,
    base_url: str,
    timeout_seconds: float,
    post_json: PostJson,
) -> tuple[dict[tuple[str, str], tuple[int, str]], int]:
    started = time.perf_counter()
    endpoint = f"{str(base_url or DEFAULT_OLLAMA_URL).rstrip('/')}/api/generate"
    payload = post_json(
        endpoint,
        {
            "model": model,
            "prompt": _grade_prompt(rollups),
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 768},
        },
        timeout_seconds,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    if not isinstance(payload, Mapping):
        raise ValueError("unexpected Ollama grading payload")
    parsed = _json_mapping_from_text(str(payload.get("response") or ""))
    raw_grades = parsed.get("grades")
    if not isinstance(raw_grades, (Mapping, list)):
        raise ValueError("Ollama grading response missed grades")
    output: dict[tuple[str, str], tuple[int, str]] = {}
    if isinstance(raw_grades, Mapping):
        for key, grade_value in raw_grades.items():
            source, sep, horizon = str(key).partition("|")
            if not source or not sep:
                continue
            output[(source, horizon or "medium")] = (_clamp_int(grade_value, 0, 10, 5), "AI grade")
        return output, latency_ms
    for item in raw_grades:
        if isinstance(item, Mapping):
            source = str(item.get("source") or "")
            horizon = str(item.get("horizon") or "medium")
            grade_value = item.get("grade")
            reason_value = item.get("reason")
        elif isinstance(item, list):
            if len(item) < 3:
                continue
            source = str(item[0] or "")
            horizon = str(item[1] or "medium")
            grade_value = item[2]
            reason_value = item[3] if len(item) > 3 else "AI grade"
        else:
            continue
        if not source:
            continue
        grade = _clamp_int(grade_value, 0, 10, 5)
        reason = " ".join(str(reason_value or "AI grade").split())[:220]
        output[(source, horizon)] = (grade, reason)
    return output, latency_ms


def _ai_grades(
    rollups: list[dict[str, object]],
    *,
    model: str,
    base_url: str,
    timeout_seconds: float,
    post_json: PostJson,
) -> tuple[dict[tuple[str, str], tuple[int, str]], int]:
    output: dict[tuple[str, str], tuple[int, str]] = {}
    total_latency_ms = 0
    batch_size = 10
    for start in range(0, len(rollups), batch_size):
        batch_output, latency_ms = _ai_grade_batch(
            rollups[start:start + batch_size],
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            post_json=post_json,
        )
        output.update(batch_output)
        total_latency_ms += latency_ms
    return output, total_latency_ms


def grade_sources(
    *,
    db_path: str | Path = "data/trading_telemetry.sqlite",
    window_hours: float = 24.0,
    model: str = DEFAULT_OLLAMA_NEWS_MODEL,
    ollama_enabled: bool = True,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    ollama_timeout_seconds: float = 15.0,
    post_json: PostJson = _post_json,
    now_ms: int | None = None,
) -> SourceGradeRun:
    now = _now_ms() if now_ms is None else int(now_ms)
    window_ms = max(60_000, int(float(window_hours) * 3_600_000))
    start = now - window_ms
    warnings: list[str] = []
    ai_status = "disabled"
    ai_latency_ms = 0
    with TradingTelemetryStore(db_path) as store:
        rollups = store.source_rollups(since_ms=start, until_ms=now)
        if not rollups:
            return SourceGradeRun(
                status="empty",
                db_path=str(db_path),
                model=model,
                window_start_ms=start,
                window_end_ms=now,
                graded_sources=0,
                ai_status=ai_status,
                ai_latency_ms=0,
                warnings=["no telemetry observations in grading window"],
                grades=[],
            )
        ai_output: dict[tuple[str, str], tuple[int, str]] = {}
        if ollama_enabled:
            try:
                ai_output, ai_latency_ms = _ai_grades(
                    rollups,
                    model=model,
                    base_url=ollama_url,
                    timeout_seconds=ollama_timeout_seconds,
                    post_json=post_json,
                )
                ai_status = "ok"
            except Exception as exc:
                ai_status = "error"
                warnings.append(f"ollama grading unavailable: {exc}")
        grades: list[SourceGrade] = []
        for rollup in rollups:
            source = str(rollup["source"])
            horizon = str(rollup["horizon"] or "medium")
            heuristic_grade, heuristic_reason = _heuristic_grade(rollup)
            grade, reason = ai_output.get((source, horizon), (heuristic_grade, heuristic_reason))
            store.record_source_grade(
                source=source,
                horizon=horizon,
                window_start_ms=start,
                window_end_ms=now,
                grade=grade,
                sample_count=int(rollup["sample_count"]),
                model=model if ai_status == "ok" else "heuristic",
                reason=reason,
                evidence=rollup,
            )
        grades = store.recent_grades(limit=len(rollups))
    return SourceGradeRun(
        status="ok" if not warnings else "warn",
        db_path=str(db_path),
        model=model if ai_status == "ok" else "heuristic",
        window_start_ms=start,
        window_end_ms=now,
        graded_sources=len(grades),
        ai_status=ai_status,
        ai_latency_ms=ai_latency_ms,
        warnings=warnings,
        grades=grades,
    )


def render_source_grade_run(run: SourceGradeRun) -> str:
    lines = [
        "Source grade run",
        (
            f"status={run.status} graded_sources={run.graded_sources} "
            f"ai={run.ai_status} model={run.model} latency_ms={run.ai_latency_ms}"
        ),
        f"db={run.db_path}",
    ]
    for grade in run.grades[:30]:
        lines.append(
            f"- {grade.source} horizon={grade.horizon} grade={grade.grade}/10 "
            f"samples={grade.sample_count} reason={grade.reason}"
        )
    for warning in run.warnings:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)
