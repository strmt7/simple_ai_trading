"""One-use, synthetic-only CPU/OpenCL comparison in an isolated runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import lightgbm as lgb
import numpy as np


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sample(rng, rows: int, features: int):
    x = rng.standard_normal((rows, features)).astype(np.float32)
    y = (
        x[:, 0] + 0.5 * x[:, 1] - 0.2 * x[:, 2] + rng.standard_normal(rows) * 0.5 > 0
    ).astype(np.float32)
    return x, y


def fit(spec, backend, x, y, vx, vy, rounds):
    parameters = {
        "objective": "binary",
        "verbosity": -1,
        "seed": spec["seed"],
        "num_threads": spec["threads"],
        "max_bin": spec["max_bin"],
        "num_leaves": spec["num_leaves"],
        "device_type": backend,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
    }
    if backend == "gpu":
        parameters.update(
            {k: spec[k] for k in ("gpu_platform_id", "gpu_device_id", "gpu_use_dp")}
        )
    start = time.perf_counter()
    model = lgb.train(parameters, lgb.Dataset(x, label=y), num_boost_round=rounds)
    prediction = model.predict(vx)
    elapsed = time.perf_counter() - start
    if not np.isfinite(prediction).all():
        raise RuntimeError("Nonfinite prediction")
    serialized = model.model_to_string()
    restored = lgb.Booster(model_str=serialized).predict(
        vx, num_threads=spec["threads"]
    )
    if not np.array_equal(prediction, restored):
        raise RuntimeError("Train/save/reload prediction mismatch")
    clipped = np.clip(prediction, 1e-15, 1 - 1e-15)
    loss = float(-np.mean(vy * np.log(clipped) + (1 - vy) * np.log1p(-clipped)))
    return {
        "backend": backend,
        "seconds": elapsed,
        "logloss": loss,
        "model_sha256": digest(serialized.encode()),
        "prediction_sha256": digest(prediction.tobytes()),
        "reload_equal": True,
    }, prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Benchmark output already exists")
    spec_bytes = args.spec.read_bytes()
    spec = json.loads(spec_bytes)
    if (lgb.__version__, np.__version__) != (
        spec["lightgbm_version"],
        spec["numpy_version"],
    ):
        raise RuntimeError("Runtime versions differ from the benchmark specification")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently replacing an observed benchmark.
    journal_path = args.output.with_suffix(".journal.jsonl")
    with journal_path.open("x", encoding="utf-8") as journal:

        def emit(event):
            journal.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            journal.flush()

        result = {
            "spec_sha256": digest(spec_bytes),
            "implementation_sha256": digest(Path(__file__).read_bytes()),
            "python": sys.version,
            "platform": platform.platform(),
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "library_sha256": digest(Path(lgb.basic._LIB._name).read_bytes()),
            "warmups": [],
            "workloads": [],
            "financial_validation": False,
        }
        emit({"phase": "started", "spec_sha256": result["spec_sha256"]})
        try:
            rng = np.random.Generator(np.random.PCG64(spec["seed"]))
            vx, vy = sample(rng, spec["validation_rows"], spec["features"])
            wx, wy = sample(rng, spec["warmup_rows"], spec["features"])
            for backend in ("cpu", "gpu"):
                emit({"phase": "warmup_started", "backend": backend})
                summary, _ = fit(spec, backend, wx, wy, vx, vy, spec["warmup_rounds"])
                result["warmups"].append(summary)
                emit({"phase": "warmup_completed", **summary})
            for rows in spec["training_rows"]:
                x, y = sample(rng, rows, spec["features"])
                trials, pairs = [], []
                for repeat in range(spec["repetitions"]):
                    predictions, losses = {}, {}
                    order = ("cpu", "gpu") if repeat % 2 == 0 else ("gpu", "cpu")
                    for backend in order:
                        emit(
                            {
                                "phase": "fit_started",
                                "rows": rows,
                                "repeat": repeat,
                                "backend": backend,
                            }
                        )
                        summary, predictions[backend] = fit(
                            spec, backend, x, y, vx, vy, spec["boosting_rounds"]
                        )
                        losses[backend] = summary["logloss"]
                        trials.append({"repeat": repeat, **summary})
                        emit({"phase": "fit_completed", "rows": rows, **trials[-1]})
                    pairs.append(
                        {
                            "max_prediction_difference": float(
                                np.max(np.abs(predictions["cpu"] - predictions["gpu"]))
                            ),
                            "logloss_difference": abs(losses["cpu"] - losses["gpu"]),
                        }
                    )
                medians = {
                    b: statistics.median(
                        t["seconds"] for t in trials if t["backend"] == b
                    )
                    for b in ("cpu", "gpu")
                }
                valid = all(
                    p["max_prediction_difference"]
                    <= spec["max_absolute_prediction_difference"]
                    and p["logloss_difference"]
                    <= spec["max_absolute_logloss_difference"]
                    for p in pairs
                )
                speedup = medians["cpu"] / medians["gpu"]
                result["workloads"].append(
                    {
                        "rows": rows,
                        "input_sha256": digest(x.tobytes() + y.tobytes()),
                        "trials": trials,
                        "paired_checks": pairs,
                        "median_seconds": medians,
                        "cpu_over_gpu_speedup": speedup,
                        "paired_tolerances_passed": valid,
                        "prefer_gpu_for_this_workload": valid
                        and speedup >= spec["minimum_median_speedup_to_prefer_gpu"],
                    }
                )
                emit(
                    {
                        "phase": "workload_completed",
                        "rows": rows,
                        "speedup": speedup,
                        "paired_tolerances_passed": valid,
                    }
                )
            result["status"] = "completed"
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            emit({"phase": "failed", "error": result["error"]})
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
        emit({"phase": result["status"], "output": str(args.output)})
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
