"""Native PDF imports must use fresh caches and an explicit environment."""

import json
from hashlib import sha256
from pathlib import Path
import sys
from types import SimpleNamespace

from tools import capture_crawl4ai_pdf as module
from tools.crawl4ai_pdf_source import canonical_digest, digest, write_json


def test_pdf_worker_does_not_reuse_parent_environment_or_installed_bytecode(
    tmp_path, monkeypatch
):
    provider_root = tmp_path / "provider"
    (provider_root / "tools/requirements").mkdir(parents=True)
    tool = provider_root / "tools/crawl4ai_tool.py"
    lock = provider_root / "tools/requirements/crawl4ai.txt"
    tool.write_text("# offline fixture\n", encoding="utf-8")
    lock.write_text("# offline fixture\n", encoding="utf-8")
    provider = tmp_path / "provider.json"
    write_json(
        provider,
        {
            "provider_root": str(provider_root),
            "provider_python": sys.executable,
            "tool_sha256": sha256(
                tool.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest(),
            "lock_sha256": sha256(
                lock.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest(),
        },
    )
    root = Path(module.__file__).resolve().parents[1]
    output = tmp_path / "capture"
    environment = tmp_path / "environment.json"
    write_json(environment, {"PDF_FIXTURE_EXPLICIT": "synthetic"})
    runtime = tmp_path / "runtime.json"
    verification = tmp_path / "verification.json"
    write_json(runtime, {})
    write_json(verification, {})
    plan = {
        "worker_sha256": digest(root / "tools/crawl4ai_pdf_source.py"),
        "controller_sha256": digest(Path(module.__file__)),
        "outer_timeout_seconds": 150,
        "output": str(output),
        "controller_log": str(tmp_path / "controller.log"),
        "controller_journal": str(tmp_path / "journal.jsonl"),
        "controller_result": str(tmp_path / "terminal.json"),
    }
    for key, path in (
        ("provider_record", provider),
        ("environment", environment),
        ("runtime_manifest", runtime),
        ("provider_verification", verification),
    ):
        plan[key] = str(path)
        plan[key + "_sha256"] = digest(path)
    plan["plan_sha256"] = canonical_digest(plan)
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    monkeypatch.setenv("PDF_FIXTURE_PARENT_ONLY", "must not be inherited")
    calls = []

    def run(command, **options):
        calls.append(command)
        assert command[1:4] == ["-B", "-I", "-X"]
        assert command[4] == "pycache_prefix=" + str(output / "import-cache")
        assert not output.exists()
        assert options["inherit_environment"] is False
        assert options["env"] == {"PDF_FIXTURE_EXPLICIT": "synthetic"}
        assert options["timeout"] == 150 and options["input_text"] == ""
        output.mkdir()
        write_json(output / "admission.json", {"admitted": True})
        return SimpleNamespace(returncode=0, stdout="offline fixture", stderr="")

    monkeypatch.setattr(module, "run_captured_process", run)
    assert module.capture(plan_path, preflight=True) is None and not calls
    terminal = module.capture(plan_path)
    assert len(calls) == 1 and terminal["transport_and_extraction_admitted"]
    assert json.loads((tmp_path / "terminal.json").read_bytes()) == terminal
