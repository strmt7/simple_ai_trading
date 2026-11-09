"""Run an explicitly configured, bounded local provider without replacing others."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from urllib.parse import urlsplit


def session_environment(rocm_root: Path, host: str) -> dict[str, str]:
    """Validate explicit capabilities; requesting ROCm is not proof of GPU residency."""
    library = rocm_root.resolve() / "bin" / "rocblas" / "library"
    if not library.is_dir():
        raise ValueError("Configured ROCm root has no rocBLAS library directory")
    endpoint = urlsplit(host if "://" in host else "//" + host)
    if (
        endpoint.scheme not in ("", "http")
        or endpoint.username
        or endpoint.password
        or endpoint.path
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError(
            "Provider endpoint must be an unauthenticated loopback host and port"
        )
    if (
        not endpoint.hostname
        or not ipaddress.ip_address(endpoint.hostname).is_loopback
        or not endpoint.port
    ):
        raise ValueError("An explicit loopback address and port are required")
    family = socket.AF_INET6 if ":" in endpoint.hostname else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        # Refuse occupied endpoints; never terminate or reconfigure their owner.
        probe.bind((endpoint.hostname, endpoint.port))
    env = os.environ.copy()
    env.update(
        {
            "OLLAMA_LLM_LIBRARY": "rocm",
            "ROCBLAS_TENSILE_LIBPATH": str(library),
            "HIP_PATH": str(rocm_root.resolve()),
            "ROCM_PATH": str(rocm_root.resolve()),
            "OLLAMA_HOST": "http://" + endpoint.netloc,
        }
    )
    env.pop("OLLAMA_VULKAN", None)
    return env


def run_session(
    executable: Path, env: dict[str, str], seconds: int
) -> dict[str, object]:
    """Contain the original provider and runners until the session ends or expires."""
    if os.name != "nt" or type(seconds) is not int or not 1 <= seconds <= 3600:
        raise ValueError("Require Windows and a bounded one-to-3600-second session")
    if not executable.is_file():
        raise ValueError("Configured provider executable does not exist")
    from simple_ai_trading.windows_owned_job import CREATE_SUSPENDED, WindowsOwnedJob

    job, process = WindowsOwnedJob(), None
    try:
        process = subprocess.Popen(
            [str(executable), "serve"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=CREATE_SUSPENDED | subprocess.CREATE_NO_WINDOW,
        )
        job.enroll_and_resume(process)
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            pass
        else:
            if process.returncode:
                raise RuntimeError("Owned provider exited before session completion")
    finally:
        try:
            job.close()
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    return {
        "session_finished": True,
        "gpu_residency_verified": False,
        "persistent_installation": False,
    }


def main() -> None:
    """Require explicit storage/endpoint settings and leave no detached server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm-root", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--executable", type=Path)
    args = parser.parse_args()
    executable = args.executable or shutil.which("ollama")
    if not executable:
        parser.error(
            "An installed provider executable is required; no installation is performed"
        )
    print(
        json.dumps(
            run_session(
                Path(executable),
                session_environment(args.rocm_root, args.host),
                args.seconds,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
