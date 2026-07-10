"""Process-isolated JSON-lines worker for foundation-model inference."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


MAX_REQUEST_BYTES = 8 * 1024 * 1024


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, separators=(",", ":"), allow_nan=False), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-size", choices=("small", "base"), required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--source-cache", default=None)
    parser.add_argument("--require-accelerator", action="store_true")
    return parser


def _request_frames(payload: dict[str, Any]) -> tuple[list[Any], list[Any], list[Any]]:
    import pandas as pd

    contexts = payload.get("contexts")
    if not isinstance(contexts, list) or not contexts or len(contexts) > 3:
        raise ValueError("worker contexts must contain between one and three items")
    frames: list[Any] = []
    histories: list[Any] = []
    futures: list[Any] = []
    for context in contexts:
        if not isinstance(context, dict):
            raise ValueError("worker context must be an object")
        columns = context.get("columns")
        values = context.get("values")
        history_ms = context.get("history_ms")
        future_ms = context.get("future_ms")
        if columns not in (
            ["open", "high", "low", "close"],
            ["open", "high", "low", "close", "volume", "amount"],
        ):
            raise ValueError("worker context columns violate the Kronos input contract")
        if not isinstance(values, list) or not isinstance(history_ms, list) or not isinstance(future_ms, list):
            raise ValueError("worker context arrays are missing")
        if len(values) != len(history_ms) or not values or not future_ms:
            raise ValueError("worker context array lengths are inconsistent")
        frames.append(pd.DataFrame(values, columns=columns, dtype="float32"))
        histories.append(pd.Series(pd.to_datetime(history_ms, unit="ms", utc=True)))
        futures.append(pd.Series(pd.to_datetime(future_ms, unit="ms", utc=True)))
    return frames, histories, futures


def worker_main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        from .foundation_forecast import KronosForecastEngine

        with contextlib.redirect_stdout(sys.stderr):
            engine = KronosForecastEngine.load(
                model_size=args.model_size,
                backend=args.backend,
                source_cache_root=(Path(args.source_cache) if args.source_cache else None),
                bootstrap_source=False,
                require_accelerator=bool(args.require_accelerator),
            )
    except Exception as exc:
        _emit(
            {
                "type": "startup_error",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
        )
        traceback.print_exc(file=sys.stderr)
        return 2
    _emit({"type": "ready", "worker_pid": os.getpid(), "report": engine.report.asdict()})
    for raw_line in sys.stdin:
        if len(raw_line.encode("utf-8")) > MAX_REQUEST_BYTES:
            _emit({"type": "error", "message": "worker request exceeded byte limit"})
            return 2
        payload: object = None
        try:
            payload = json.loads(raw_line)
            if not isinstance(payload, dict) or payload.get("type") != "predict":
                raise ValueError("worker request type must be predict")
            request_id = int(payload["id"])
            frames, histories, futures = _request_frames(payload)
            started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                predictions = engine.predict_batch(
                    frames,
                    histories,
                    futures,
                    prediction_length=int(payload["prediction_length"]),
                    temperature=float(payload["temperature"]),
                    top_k=int(payload["top_k"]),
                    top_p=float(payload["top_p"]),
                    sample_count=int(payload["sample_count"]),
                    seed=int(payload["seed"]),
                )
            _emit(
                {
                    "type": "result",
                    "id": request_id,
                    "worker_pid": os.getpid(),
                    "seconds": time.perf_counter() - started,
                    "predicted_closes": [
                        [float(value) for value in prediction["close"]]
                        for prediction in predictions
                    ],
                }
            )
        except Exception as exc:
            _emit(
                {
                    "type": "error",
                    "id": payload.get("id") if isinstance(payload, dict) else None,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            traceback.print_exc(file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(worker_main())
