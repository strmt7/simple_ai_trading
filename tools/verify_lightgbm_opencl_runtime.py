"""Fail closed on the wrong local wheel or an unavailable actual OpenCL learner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm

from simple_ai_trading.lightgbm_backend import (
    discover_opencl_gpu_devices,
    lightgbm_backend_parameters,
)


def verify(manifest: dict) -> dict:
    library = Path(lightgbm.basic._LIB._name)
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    if lightgbm.__version__ != manifest["lightgbm_version"]:
        raise RuntimeError("Unexpected LightGBM version")
    if digest != manifest["library_sha256"]:
        raise RuntimeError(
            "GPU library identity changed; do not substitute a CPU wheel"
        )
    parameters, backend, device = lightgbm_backend_parameters(
        "auto", 20260904, pin_opencl_device=True, reproducible=True
    )
    if backend != "opencl":
        raise RuntimeError("Actual OpenCL training probe failed; GPU launch stopped")
    devices = [
        entry.as_dict()
        for entry in discover_opencl_gpu_devices()
        if entry.platform_id == parameters["gpu_platform_id"]
        and entry.device_id == parameters["gpu_device_id"]
    ]
    if len(devices) != 1 or devices[0]["display_name"] != manifest["device_name"]:
        raise RuntimeError("GPU device differs from the reviewed runtime")
    return {
        "backend": backend,
        "device": device,
        "library_sha256": digest,
        "driver_version": devices[0]["driver_version"],
        "probe_gpu_use_dp": parameters["gpu_use_dp"],
        "scope": "capability_only_training_arguments_and_frozen_contracts_unchanged",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(json.dumps(verify(manifest), sort_keys=True))


if __name__ == "__main__":
    main()
