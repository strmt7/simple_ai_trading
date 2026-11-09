#!/usr/bin/env python3
"""Provision an owned Windows environment without modifying host installations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import stat
import sys
import sysconfig
import tomllib
import venv

# The bootstrap uses only stdlib and the existing owned-child implementation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tools.cocoindex_agent_search import FileLock, run_captured_process  # noqa: E402


def install_plan(repository: Path, extras: list[str]) -> dict[str, object]:
    """Bind an installation to exact declarations, lock bytes and chosen extras."""
    project_bytes = (repository / "pyproject.toml").read_bytes()
    lock_bytes = (repository / "uv.lock").read_bytes()
    project = tomllib.loads(project_bytes.decode("utf-8"))
    lock = tomllib.loads(lock_bytes.decode("utf-8"))
    selected = sorted(set(extras))
    declared = project["project"]["optional-dependencies"]
    if any(extra not in declared for extra in selected):
        raise ValueError("Select only extras declared by this checkout.")
    for conflicts in project["tool"]["uv"].get("conflicts", []):
        if sum(entry.get("extra") in selected for entry in conflicts) > 1:
            raise ValueError("The selected extras conflict; use separate environments.")
    package = next(item for item in lock["package"] if item["name"] == "uv")
    hashes = sorted({wheel["hash"] for wheel in package["wheels"]})
    if not hashes or any(
        not value.startswith("sha256:") or len(value) != 71 for value in hashes
    ):
        raise ValueError("The uv bootstrap requires SHA-256 wheel hashes.")
    wheel_suffix = "-" + sysconfig.get_platform().replace("-", "_") + ".whl"
    wheels = [item for item in package["wheels"] if item["url"].endswith(wheel_suffix)]
    if os.name == "nt" and len(wheels) != 1:
        raise ValueError(
            "No unique locked uv wheel supports this bootstrap architecture."
        )
    return {
        "schema": "owned-environment-v1",
        "repository": str(repository.resolve()),
        "project_sha256": hashlib.sha256(project_bytes).hexdigest(),
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "extras": selected,
        "python": project["tool"]["simple-ai-trading"]["installer"]["python"],
        "platform": sysconfig.get_platform(),
        "uv_version": package["version"],
        "uv_hashes": hashes,
        "uv_wheel": wheels[0] if len(wheels) == 1 else None,
    }


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _claim(path: Path, identity: dict[str, object]) -> None:
    """Refuse to reuse another application, checkout or unowned directory."""
    marker = path / "ownership.json"
    if path.exists():
        if _reparse_point(path):
            raise RuntimeError("Installation directories must not be reparse points.")
        if not marker.is_file() or _reparse_point(marker):
            raise RuntimeError("Refusing an existing unowned installation directory.")
        if json.loads(marker.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("Installation ownership does not match this checkout.")
    else:
        path.mkdir(parents=True)
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(_canonical(identity))


def _reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    if not path.exists():
        return False
    return bool(
        getattr(path.lstat(), "st_file_attributes", 0)
        & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def _run(args: list[str], repository: Path, env: dict[str, str]) -> str:
    result = run_captured_process(
        args,
        cwd=repository,
        env=env,
        input_text="",
        timeout=3600,
        inherit_environment=False,
    )
    if result.returncode:
        # Keep native package-manager diagnostics local; no account secrets are needed.
        raise RuntimeError(
            f"Installer command failed ({result.returncode}):\n{result.stderr}"
        )
    return result.stdout


def provision(repository: Path, storage: Path, extras: list[str]) -> Path:
    """Create once per lock/profile; never sync over a running environment."""
    if os.name != "nt":
        raise RuntimeError(
            "This owned-process bootstrap currently supports Windows only."
        )
    repository = repository.resolve()
    storage = storage.absolute()
    # Reject reparse points before resolving them, including existing ancestors.
    for component in (storage, *storage.parents):
        if _reparse_point(component):
            raise RuntimeError("Installation storage must not traverse reparse points.")
    plan = install_plan(repository, extras)
    identity = {"schema": "owned-installer-v1", "repository": str(repository)}
    _claim(storage, identity)
    key = hashlib.sha256(_canonical(plan).encode("utf-8")).hexdigest()[:24]
    target = storage / key
    with FileLock(storage / "install.lock"):
        _claim(target, plan)
        python = target / "runtime" / "Scripts" / "python.exe"
        complete = target / "ready.json"
        env = os.environ.copy()
        # Ambient package-manager preferences must not redirect or widen installs.
        for name in tuple(env):
            if name.startswith(("UV_", "PIP_")) or name in {
                "PYTHONHOME",
                "PYTHONPATH",
                "VIRTUAL_ENV",
                "CONDA_PREFIX",
            }:
                env.pop(name)
        env.update(
            {
                "UV_CACHE_DIR": str(storage / "cache"),
                "UV_PYTHON_INSTALL_DIR": str(storage / "python"),
                "UV_PROJECT_ENVIRONMENT": str(target / "runtime"),
                "UV_LINK_MODE": "copy",
                "UV_PYTHON_DOWNLOADS": "automatic",
                "UV_CONCURRENT_DOWNLOADS": "2",
                "UV_CONCURRENT_BUILDS": "1",
                "UV_CONCURRENT_INSTALLS": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )
        if complete.exists():
            if json.loads(complete.read_text(encoding="utf-8")) != plan:
                raise RuntimeError("Completed installation marker is inconsistent.")
            if not python.is_file():
                raise RuntimeError(
                    "Completed installation is damaged; retain it and select new storage."
                )
            # Read-only check: returning to a running installation must not repair it.
            _run(
                [
                    str(python),
                    str(repository / "tools/verify_runtime_entrypoint.py"),
                    "--repository",
                    str(repository),
                ],
                repository,
                env,
            )
            return python

        if (target / "runtime").exists():
            raise RuntimeError(
                "An incomplete runtime is retained; select new storage rather than overwrite it."
            )

        bootstrap = target / "bootstrap"
        _claim(bootstrap, {"schema": "owned-uv-bootstrap-v1", "plan": plan})
        bootstrap_python = bootstrap / "venv" / "Scripts" / "python.exe"
        if not bootstrap_python.is_file():
            # ensurepip affects this venv only; it never upgrades host pip.
            venv.EnvBuilder(with_pip=False).create(bootstrap / "venv")
            _run(
                [str(bootstrap_python), "-I", "-m", "ensurepip", "--default-pip"],
                repository,
                env,
            )
        requirements = bootstrap / "uv-requirements.txt"
        requirements.write_text(
            "uv @ "
            + str(plan["uv_wheel"]["url"])
            + " --hash="
            + str(plan["uv_wheel"]["hash"])
            + "\n",
            encoding="ascii",
        )
        _run(
            [
                str(bootstrap_python),
                "-I",
                "-m",
                "pip",
                "--isolated",
                "--disable-pip-version-check",
                "install",
                "--no-input",
                "--no-deps",
                "--only-binary=:all:",
                "--require-hashes",
                "--no-index",
                "--cache-dir",
                str(storage / "pip-cache"),
                "-r",
                str(requirements),
            ],
            repository,
            env,
        )
        uv = bootstrap / "venv" / "Scripts" / "uv.exe"
        args = [
            str(uv),
            "--no-config",
            "sync",
            "--locked",
            "--managed-python",
            "--python",
            str(plan["python"]),
            "--no-build-package",
            "numpy",
            "--group",
            "tools",
            "--project",
            str(repository),
        ]
        for extra in plan["extras"]:
            args.extend(["--extra", str(extra)])
        _run(args, repository, env)
        _run(
            [str(uv), "--no-config", "pip", "check", "--python", str(python)],
            repository,
            env,
        )
        _run(
            [
                str(python),
                str(repository / "tools/verify_runtime_entrypoint.py"),
                "--repository",
                str(repository),
            ],
            repository,
            env,
        )
        with complete.open("x", encoding="utf-8") as handle:
            handle.write(_canonical(plan))
        return python


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage", type=Path, default=ROOT / "data/tooling/application"
    )
    parser.add_argument("--extra", action="append", default=[])
    args = parser.parse_args()
    try:
        python = provision(ROOT, args.storage, args.extra)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"python": str(python), "ready": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
