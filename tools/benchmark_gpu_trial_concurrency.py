"""Bounded synthetic worker scheduling comparison; no market data or orders."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    raw = args.spec.read_bytes()
    spec = json.loads(raw)
    args.output_dir.mkdir(exist_ok=False)
    result = {
        "spec_sha256": hashlib.sha256(raw).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "financial_validation": False,
        "configurations": [],
    }
    base = json.loads(Path(spec["worker_template"]).read_bytes())
    worker_script = Path("tools/profile_lightgbm_opencl.py")
    result["worker_implementation_sha256"] = hashlib.sha256(
        worker_script.read_bytes()
    ).hexdigest()
    result["worker_template_sha256"] = hashlib.sha256(
        Path(spec["worker_template"]).read_bytes()
    ).hexdigest()
    # Freeze every worker configuration before observing any concurrency result.
    for config in spec["configurations"]:
        worker_spec = {
            **base,
            "trial_order": [["gpu", config["threads_per_worker"]]],
            "purpose": "same_synthetic_dataset_concurrency_profile_not_market_training",
        }
        path = args.output_dir / f"workers-{config['workers']}-spec.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(worker_spec, stream, indent=2)
    with (args.output_dir / "journal.jsonl").open("x", encoding="utf-8") as journal:

        def emit(record):
            journal.write(json.dumps(record) + "\n")
            journal.flush()

        try:
            for config in spec["configurations"]:
                workers = config["workers"]
                if workers > 4 or workers * config["threads_per_worker"] > 16:
                    raise RuntimeError("worker/CPU budget exceeded")
                begin = time.perf_counter()
                records = []
                for first in range(0, spec["total_fits_per_configuration"], workers):
                    active = []
                    try:
                        for index in range(
                            first,
                            min(first + workers, spec["total_fits_per_configuration"]),
                        ):
                            output = (
                                args.output_dir / f"workers-{workers}-fit-{index}.json"
                            )
                            log = output.with_suffix(".log").open("x", encoding="utf-8")
                            command = [
                                sys.executable,
                                str(worker_script),
                                "--spec",
                                str(args.output_dir / f"workers-{workers}-spec.json"),
                                "--output",
                                str(output),
                            ]
                            try:
                                child = subprocess.Popen(
                                    command,
                                    stdout=log,
                                    stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW,
                                )
                            except Exception:
                                log.close()
                                raise
                            active.append((child, log, output))
                            emit(
                                {
                                    "phase": "worker_started",
                                    "workers": workers,
                                    "index": index,
                                    "pid": child.pid,
                                    "output": str(output),
                                }
                            )
                        for child, log, output in active:
                            code = child.wait(timeout=spec["worker_timeout_seconds"])
                            log.close()
                            if code != 0:
                                raise RuntimeError(
                                    f"worker {child.pid} failed; inspect retained log"
                                )
                            value = json.loads(output.read_bytes())
                            if (
                                value["status"] != "completed"
                                or len(value["trials"]) != 1
                            ):
                                raise RuntimeError("worker result incomplete")
                            records.append(
                                {
                                    "path": str(output),
                                    "sha256": hashlib.sha256(
                                        output.read_bytes()
                                    ).hexdigest(),
                                    "input_sha256": value["input_sha256"],
                                    "trial": {
                                        k: v
                                        for k, v in value["trials"][0].items()
                                        if k != "samples"
                                    },
                                }
                            )
                    finally:
                        for child, log, _ in active:
                            if child.poll() is None:
                                child.terminate()
                                child.wait(timeout=15)
                            log.close()
                if len({row["input_sha256"] for row in records}) != 1:
                    raise RuntimeError("workers did not use identical inputs")
                elapsed = time.perf_counter() - begin
                summary = {
                    **config,
                    "total_wall_seconds": elapsed,
                    "records": records,
                    "fits_per_minute": len(records) * 60 / elapsed,
                }
                result["configurations"].append(summary)
                emit(
                    {
                        "phase": "configuration_completed",
                        "workers": workers,
                        "total_wall_seconds": elapsed,
                        "fits_per_minute": summary["fits_per_minute"],
                    }
                )
            result["status"] = "completed"
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        with (args.output_dir / "result.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        emit({"phase": result["status"]})
    print(
        json.dumps(
            {
                "status": result["status"],
                "configurations": len(result["configurations"]),
            }
        )
    )
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
