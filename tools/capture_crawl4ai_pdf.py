"""Supervise one frozen native PDF capture with durable one-use evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tools.cocoindex_agent_search import run_captured_process
from tools.crawl4ai_pdf_source import canonical_digest, digest, record, write_json


def capture(plan_path: Path, *, preflight: bool = False) -> dict | None:
    plan = json.loads(plan_path.read_bytes())
    body = dict(plan)
    frozen = body.pop("plan_sha256")
    if canonical_digest(body) != frozen:
        raise ValueError("PDF contract digest differs")
    root = Path(__file__).resolve().parents[1]
    worker = root / "tools/crawl4ai_pdf_source.py"
    if (
        digest(worker) != plan["worker_sha256"]
        or digest(Path(__file__)) != plan["controller_sha256"]
    ):
        raise ValueError("PDF capture code changed after freeze")
    for key in (
        "provider_record",
        "environment",
        "runtime_manifest",
        "provider_verification",
    ):
        if digest(Path(plan[key])) != plan[key + "_sha256"]:
            raise ValueError("PDF provider evidence changed after freeze")
    provider = json.loads(Path(plan["provider_record"]).read_bytes())
    checkout = Path(provider["provider_root"])
    for name, key in (
        ("tools/crawl4ai_tool.py", "tool_sha256"),
        ("tools/requirements/crawl4ai.txt", "lock_sha256"),
    ):
        import hashlib

        actual = hashlib.sha256(
            (checkout / name).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        if actual != provider[key]:
            raise ValueError("reviewed provider source changed")
    if (
        type(plan["outer_timeout_seconds"]) is not int
        or plan["outer_timeout_seconds"] != 150
    ):
        raise ValueError(
            "PDF capture requires its admitted 150-second owned-process deadline"
        )
    paths = {
        key: Path(plan[key]).resolve()
        for key in (
            "output",
            "controller_log",
            "controller_journal",
            "controller_result",
        )
    }
    if len(set(paths.values())) != 4:
        raise ValueError("PDF outputs must be distinct")
    for path in paths.values():
        if (
            path.exists()
            or not path.parent.is_dir()
            or not os.access(path.parent, os.W_OK)
        ):
            raise FileExistsError("PDF outputs must be new and their parents writable")
    if not Path(provider["provider_python"]).is_file():
        raise FileNotFoundError("admitted provider interpreter is unavailable")
    if preflight:
        return None
    # Exclusive reservation precedes process creation, including failed launches.
    with paths["controller_journal"].open("x", encoding="utf-8"):
        pass
    record(paths["controller_journal"], "intent", plan_sha256=frozen)
    terminal = {"plan_sha256": frozen, "error_type": None, "exit_code": None}
    try:
        completed = run_captured_process(
            [
                provider["provider_python"],
                "-B",
                "-I",
                "-X",
                "pycache_prefix=" + str(paths["output"] / "import-cache"),
                str(worker),
                "--plan",
                str(plan_path.resolve()),
            ],
            cwd=root,
            env=json.loads(Path(plan["environment"]).read_bytes()),
            input_text="",
            timeout=150,
            inherit_environment=False,
        )
        terminal["exit_code"] = completed.returncode
        with paths["controller_log"].open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(completed.stdout + "\n" + completed.stderr)
            stream.flush()
            os.fsync(stream.fileno())
        admission = paths["output"] / "admission.json"
        terminal["transport_and_extraction_admitted"] = (
            completed.returncode == 0
            and admission.is_file()
            and json.loads(admission.read_bytes())["admitted"] is True
        )
    except BaseException as failure:
        terminal["error_type"] = type(failure).__name__
        terminal["transport_and_extraction_admitted"] = False
        raise
    finally:
        write_json(paths["controller_result"], terminal)
        record(paths["controller_journal"], "terminal", **terminal)
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    print(json.dumps(capture(args.plan, preflight=args.preflight)))


if __name__ == "__main__":
    main()
