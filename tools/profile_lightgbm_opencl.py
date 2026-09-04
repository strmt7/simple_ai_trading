"""Synthetic throughput/telemetry profile; never a market-accuracy experiment."""

from __future__ import annotations

import argparse
import ctypes as ct
from ctypes import wintypes as wt
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import lightgbm as lgb
import numpy as np


class CounterValue(ct.Structure):
    _fields_ = [("status", wt.DWORD), ("value", ct.c_double)]


class CounterItem(ct.Structure):
    _fields_ = [("name", wt.LPWSTR), ("formatted", CounterValue)]


class GpuCounters:
    """Windows PDH busiest process engine, not summed engines or ALU occupancy."""

    def __init__(self):
        self.api = ct.WinDLL("pdh.dll")
        self.query = ct.c_void_p()
        self.counters = {}
        self.api.PdhOpenQueryW.argtypes = [
            wt.LPCWSTR,
            ct.c_size_t,
            ct.POINTER(ct.c_void_p),
        ]
        self.api.PdhAddEnglishCounterW.argtypes = [
            ct.c_void_p,
            wt.LPCWSTR,
            ct.c_size_t,
            ct.POINTER(ct.c_void_p),
        ]
        self.api.PdhCollectQueryData.argtypes = [ct.c_void_p]
        self.api.PdhCloseQuery.argtypes = [ct.c_void_p]
        self.api.PdhGetFormattedCounterArrayW.argtypes = [
            ct.c_void_p,
            wt.DWORD,
            ct.POINTER(wt.DWORD),
            ct.POINTER(wt.DWORD),
            ct.c_void_p,
        ]
        self.api.PdhGetFormattedCounterArrayW.restype = wt.DWORD
        if self.api.PdhOpenQueryW(None, 0, ct.byref(self.query)):
            raise RuntimeError("PDH query unavailable")
        try:
            for key, path in {
                "engine_percent": r"\GPU Engine(*)\Utilization Percentage",
                "dedicated_bytes": r"\GPU Process Memory(*)\Dedicated Usage",
                "shared_bytes": r"\GPU Process Memory(*)\Shared Usage",
            }.items():
                counter = ct.c_void_p()
                if self.api.PdhAddEnglishCounterW(
                    self.query, path, 0, ct.byref(counter)
                ):
                    raise RuntimeError(f"PDH counter unavailable: {key}")
                self.counters[key] = counter
            self.api.PdhCollectQueryData(self.query)
        except Exception:
            self.close()
            raise

    def close(self):
        if self.query:
            self.api.PdhCloseQuery(self.query)
            self.query = ct.c_void_p()

    def sample(self):
        collection_status = self.api.PdhCollectQueryData(self.query)
        output = {
            "collection_status": collection_status,
            "seconds": time.perf_counter(),
        }
        prefix = f"pid_{os.getpid()}_"
        for key, counter in self.counters.items():
            size, count = wt.DWORD(0), wt.DWORD(0)
            status = self.api.PdhGetFormattedCounterArrayW(
                counter, 0x200, ct.byref(size), ct.byref(count), None
            )
            if status != 0x800007D2 or size.value > 8_000_000:
                output[key] = {"status": status, "values": {}}
                continue
            buffer = ct.create_string_buffer(size.value)
            status = self.api.PdhGetFormattedCounterArrayW(
                counter, 0x200, ct.byref(size), ct.byref(count), buffer
            )
            values = {}
            if status == 0:
                entries = ct.cast(buffer, ct.POINTER(CounterItem))
                for i in range(count.value):
                    entry = entries[i]
                    if entry.name.startswith(prefix) and entry.formatted.status in (
                        0,
                        1,
                    ):
                        value = entry.formatted.value
                        if np.isfinite(value) and value >= 0:
                            values[entry.name] = value
            output[key] = {"status": status, "values": values}
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--telemetry-preflight", action="store_true")
    args = parser.parse_args()
    if args.telemetry_preflight:
        counters = GpuCounters()
        try:
            sample = counters.sample()
            print(
                json.dumps(
                    {k: v["status"] for k, v in sample.items() if isinstance(v, dict)}
                )
            )
        finally:
            counters.close()
        return
    if args.output.exists():
        raise RuntimeError("Profile output already exists")
    spec_raw = args.spec.read_bytes()
    spec = json.loads(spec_raw)
    library_hash = hashlib.sha256(Path(lgb.basic._LIB._name).read_bytes()).hexdigest()
    if (
        library_hash != spec["library_sha256"]
        or np.__version__ != spec["numpy_version"]
    ):
        raise RuntimeError("Profile runtime identity mismatch")
    if spec["rows"] * spec["features"] * 4 > spec["maximum_input_bytes"]:
        raise RuntimeError("Input budget exceeded")
    result = {
        "spec_sha256": hashlib.sha256(spec_raw).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "library_sha256": library_hash,
        "trials": [],
        "financial_validation": False,
    }
    with args.output.with_suffix(".journal.jsonl").open(
        "x", encoding="utf-8"
    ) as journal:

        def emit(value):
            journal.write(json.dumps(value, allow_nan=False) + "\n")
            journal.flush()

        emit({"phase": "started", "spec_sha256": result["spec_sha256"]})
        try:
            rng = np.random.Generator(np.random.PCG64(spec["seed"]))
            started = time.perf_counter()
            x = rng.standard_normal((spec["rows"], spec["features"]), dtype=np.float32)
            y = (
                x[:, 0]
                + 0.5 * x[:, 1]
                - 0.2 * x[:, 2]
                + 0.5 * rng.standard_normal(spec["rows"], dtype=np.float32)
                > 0
            ).astype(np.float32)
            vx = rng.standard_normal((4096, spec["features"]), dtype=np.float32)
            vy = (
                vx[:, 0]
                + 0.5 * vx[:, 1]
                - 0.2 * vx[:, 2]
                + 0.5 * rng.standard_normal(4096, dtype=np.float32)
                > 0
            ).astype(np.float32)
            result["generation_seconds"] = time.perf_counter() - started
            result["input_bytes"] = x.nbytes + y.nbytes
            h = hashlib.sha256()
            h.update(memoryview(x))
            h.update(memoryview(y))
            result["input_sha256"] = h.hexdigest()
            # Initialize the real GPU context before priming Windows process counters.
            warmup = lgb.train(
                {
                    "objective": "binary",
                    "verbosity": -1,
                    "device_type": "gpu",
                    "gpu_platform_id": 0,
                    "gpu_device_id": 0,
                    "gpu_use_dp": True,
                    "num_threads": 8,
                    "max_bin": 63,
                },
                lgb.Dataset(x[:512], label=y[:512]),
                num_boost_round=1,
            )
            del warmup
            for index, (backend, threads) in enumerate(spec["trial_order"]):
                emit(
                    {
                        "phase": "trial_started",
                        "index": index,
                        "backend": backend,
                        "threads": threads,
                    }
                )
                parameters = {
                    "objective": "binary",
                    "verbosity": -1,
                    "seed": spec["seed"],
                    "max_bin": 63,
                    "num_leaves": 31,
                    "device_type": backend,
                    "num_threads": threads,
                }
                if backend == "gpu":
                    parameters.update(
                        gpu_platform_id=0, gpu_device_id=0, gpu_use_dp=True
                    )
                samples, telemetry_errors = [], []
                stop = threading.Event()
                counters = GpuCounters()

                def monitor():
                    while not stop.wait(spec["sample_interval_seconds"]):
                        try:
                            samples.append(counters.sample())
                        except Exception as exc:
                            telemetry_errors.append(type(exc).__name__)
                            return

                worker = threading.Thread(target=monitor, daemon=True)
                worker.start()
                try:
                    start, cpu_start = time.perf_counter(), time.process_time()
                    dataset = lgb.Dataset(x, label=y, params=parameters).construct()
                    constructed = time.perf_counter()
                    model = lgb.train(
                        parameters, dataset, num_boost_round=spec["rounds"]
                    )
                    trained = time.perf_counter()
                    prediction = model.predict(vx, num_threads=threads)
                    finished = time.perf_counter()
                    cpu_seconds = time.process_time() - cpu_start
                finally:
                    stop.set()
                    worker.join()
                    counters.close()
                restored = lgb.Booster(model_str=model.model_to_string()).predict(
                    vx, num_threads=threads
                )
                if not np.isfinite(prediction).all() or not np.array_equal(
                    prediction, restored
                ):
                    raise RuntimeError("Prediction/reload check failed")
                clipped = np.clip(prediction, 1e-15, 1 - 1e-15)
                loss = float(
                    -np.mean(vy * np.log(clipped) + (1 - vy) * np.log1p(-clipped))
                )
                trial = {
                    "index": index,
                    "backend": backend,
                    "threads": threads,
                    "dataset_seconds": constructed - start,
                    "train_seconds": trained - constructed,
                    "prediction_seconds": finished - trained,
                    "total_seconds": finished - start,
                    "cpu_seconds": cpu_seconds,
                    "cpu_busy_core_equivalents": cpu_seconds / (finished - start),
                    "logloss": loss,
                    "reload_equal": True,
                    "samples": samples,
                    "telemetry_errors": telemetry_errors,
                    "train_window_counter_summary": {},
                }
                for key in ("engine_percent", "dedicated_bytes", "shared_bytes"):
                    selected = [
                        s[key]["values"]
                        for s in samples
                        if constructed <= s["seconds"] <= trained and s[key]["values"]
                    ]
                    values = [
                        (
                            max(v.values())
                            if key == "engine_percent"
                            else sum(v.values())
                        )
                        for v in selected
                    ]
                    trial["train_window_counter_summary"][key] = {
                        "valid_samples": len(values),
                        "max": max(values) if values else None,
                        "mean": float(np.mean(values)) if values else None,
                    }
                result["trials"].append(trial)
                emit(
                    {
                        "phase": "trial_completed",
                        **{k: v for k, v in trial.items() if k != "samples"},
                    }
                )
                del model, dataset
            result["status"] = "completed"
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        emit({"phase": result["status"]})
    print(json.dumps({"status": result["status"], "trials": len(result["trials"])}))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
