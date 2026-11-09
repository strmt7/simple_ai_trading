from __future__ import annotations

import json
import os
import sys

import pytest

from tools import install_environment as installer
from tools.cocoindex_agent_search import run_captured_process


def test_install_plan_binds_lock_profile_platform_and_declared_conflicts():
    plan = installer.install_plan(installer.ROOT, ["reporting", "gpu", "gpu"])
    assert plan["extras"] == ["gpu", "reporting"]
    assert plan["uv_version"] == "0.13.7"
    assert plan["uv_hashes"]
    assert plan["platform"]
    assert len(plan["lock_sha256"]) == 64
    assert installer.install_plan(installer.ROOT, ["gpu", "reporting"]) == plan
    assert installer.install_plan(installer.ROOT, []) != plan
    with pytest.raises(ValueError, match="conflict"):
        installer.install_plan(installer.ROOT, ["gpu", "directml"])
    with pytest.raises(ValueError, match="declared"):
        installer.install_plan(installer.ROOT, ["unknown"])


def test_claim_refuses_foreign_directories_and_preserves_their_bytes(tmp_path):
    target = tmp_path / "foreign"
    target.mkdir()
    data = target / "important.txt"
    data.write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unowned"):
        installer._claim(target, {"repository": "this"})
    assert data.read_text(encoding="utf-8") == "keep"
    owned = tmp_path / "owned"
    installer._claim(owned, {"repository": "this"})
    installer._claim(owned, {"repository": "this"})
    with pytest.raises(RuntimeError, match="ownership"):
        installer._claim(owned, {"repository": "another"})


def test_owned_command_can_exclude_ambient_package_manager_configuration(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("UV_DEFAULT_INDEX", "must-not-be-inherited")
    env = {key: value for key, value in os.environ.items() if not key.startswith("UV_")}
    result = run_captured_process(
        [
            sys.executable,
            "-I",
            "-c",
            "import os,json; print(json.dumps(os.getenv('UV_DEFAULT_INDEX')))",
        ],
        cwd=tmp_path,
        env=env,
        input_text="",
        timeout=30,
        inherit_environment=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) is None


@pytest.mark.skipif(os.name != "nt", reason="Windows owned-job installer")
def test_completed_environment_is_checked_without_resync(monkeypatch, tmp_path):
    plan = installer.install_plan(installer.ROOT, [])
    storage = tmp_path / "install"
    installer._claim(
        storage, {"schema": "owned-installer-v1", "repository": str(installer.ROOT)}
    )
    key = installer.hashlib.sha256(installer._canonical(plan).encode()).hexdigest()[:24]
    target = storage / key
    installer._claim(target, plan)
    python = target / "runtime/Scripts/python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"not executed")
    (target / "ready.json").write_text(installer._canonical(plan), encoding="utf-8")
    commands = []
    monkeypatch.setattr(installer, "_run", lambda args, *_: commands.append(args) or "")
    assert installer.provision(installer.ROOT, storage, []) == python
    assert len(commands) == 1
    assert "verify_runtime_entrypoint.py" in commands[0][1]
    (target / "ready.json").unlink()
    with pytest.raises(RuntimeError, match="incomplete runtime"):
        installer.provision(installer.ROOT, storage, [])
    assert len(commands) == 1
