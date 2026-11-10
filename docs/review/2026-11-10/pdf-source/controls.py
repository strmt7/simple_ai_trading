"""Exercise the native PDF adapter against one owned loopback server only."""

from __future__ import annotations

import argparse
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from tools.capture_crawl4ai_pdf import capture
from tools.crawl4ai_pdf_source import canonical_digest, digest, write_json


def fixture_pdf(pages: int, *, blank: bool = False) -> bytes:
    text = b"Primary legal fixture: exercise cash flow and matching settlement. " * 5
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = b" ".join(f"{4 + 2 * i} 0 R".encode() for i in range(pages))
    objects.append(
        b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(pages).encode() + b" >>"
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for index in range(pages):
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents "
            + str(5 + 2 * index).encode()
            + b" 0 R >>"
        )
        content = b"" if blank else b"BT /F1 10 Tf 10 750 Td (" + text + b") Tj ET"
        objects.append(
            b"<< /Length "
            + str(len(content)).encode()
            + b" >>\nstream\n"
            + content
            + b"\nendstream"
        )
    data, offsets = bytearray(b"%PDF-1.4\n"), [0]
    for index, body in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(str(index).encode() + b" 0 obj\n" + body + b"\nendobj\n")
    start = len(data)
    data.extend(
        b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    )
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    )
    return bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    root = Path(__file__).resolve().parents[4]
    worker = root / "tools/crawl4ai_pdf_source.py"
    state = {"case": "", "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            state["requests"].append(self.path)
            case = state["case"]
            if self.path == "/robots.txt":
                status = 500 if case == "robots-error" else 200
                body = (
                    b"User-agent: *\nDisallow: /\n"
                    if case == "robots-deny"
                    else b"User-agent: *\nAllow: /\n"
                )
                content_type = "text/plain"
            else:
                status = {"http-202": 202, "redirect": 302}.get(case, 200)
                body = (
                    b"<html>challenge</html>"
                    if case == "not-pdf"
                    else fixture_pdf(
                        2 if case == "page-cap" else 1, blank=case == "blank"
                    )
                )
                content_type = "application/pdf"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if case == "redirect" and self.path != "/robots.txt":
                self.send_header("Location", "/forbidden-second.pdf")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.block_on_close = True
    thread = threading.Thread(target=server.serve_forever)
    rows = []
    thread.start()
    try:
        for case in (
            "valid",
            "robots-deny",
            "robots-error",
            "http-202",
            "redirect",
            "not-pdf",
            "page-cap",
            "blank",
        ):
            state.update(case=case, requests=[])
            output = args.output / case
            plan = {
                "url": f"http://127.0.0.1:{server.server_port}/source.pdf",
                "source_access": False,
                "maximum_pages": 1,
                "minimum_characters": 100,
                "output": str(output.resolve()),
                "extension_root": str(args.extension.resolve()),
                "runtime_manifest": str(args.runtime.resolve()),
                "runtime_sha256": digest(args.runtime),
                "worker_sha256": digest(worker),
            }
            plan.update(
                controller_sha256=digest(root / "tools/capture_crawl4ai_pdf.py"),
                outer_timeout_seconds=150,
                controller_log=str((args.output / (case + ".log")).resolve()),
                controller_journal=str(
                    (args.output / (case + "-journal.jsonl")).resolve()
                ),
                controller_result=str(
                    (args.output / (case + "-terminal.json")).resolve()
                ),
            )
            for key, path in (
                ("provider_record", args.provider),
                ("environment", args.environment),
                ("runtime_manifest", args.runtime),
                ("provider_verification", args.verification),
            ):
                plan[key] = str(path.resolve())
                plan[key + "_sha256"] = digest(path)
            plan["plan_sha256"] = canonical_digest(plan)
            plan_path = args.output / (case + "-plan.local.json")
            write_json(plan_path, plan)
            capture(plan_path, preflight=True)
            terminal = capture(plan_path)
            result = output / "admission.json"
            admission = json.loads(result.read_bytes()) if result.exists() else {}
            expected_requests = (
                ["/robots.txt"]
                if case.startswith("robots-")
                else ["/robots.txt", "/source.pdf"]
            )
            row = {
                "case": case,
                "exit_code": terminal["exit_code"],
                "admitted": admission.get("admitted", False),
                "requests": list(state["requests"]),
                "passed": admission.get("admitted", False) is (case == "valid")
                and state["requests"] == expected_requests
                and (terminal["exit_code"] == 0 if case == "valid" else True),
                "temporary_files": len(list((output / "temporary").iterdir())),
            }
            row["passed"] = row["passed"] and row["temporary_files"] == 0
            row["native_result_retained"] = (output / "native-result.json").is_file()
            row["passed"] = (
                row["passed"] and row["native_result_retained"] and result.is_file()
            )
            rows.append(row)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    summary = {
        "worker_sha256": digest(worker),
        "controls_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": rows,
        "server_stopped": not thread.is_alive(),
        "passed": all(row["passed"] for row in rows) and not thread.is_alive(),
        "public_source_requests": 0,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
