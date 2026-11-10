"""Extract the exact retained Clearing Procedures offline with native Crawl4AI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import socket
import sys
from unittest.mock import patch


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    root = Path(__file__).resolve().parents[4]
    api = runpy.run_path(str(root / "tools/crawl4ai_pdf_source.py"))
    body = dict(plan)
    frozen = body.pop("plan_sha256")
    if api["canonical_digest"](body) != frozen:
        raise ValueError("offline plan digest differs")
    source = root / plan["source"]
    if api["digest"](source) != plan["source_sha256"]:
        raise ValueError("retained clearing source differs")
    if api["digest"](Path(__file__)) != plan["script_sha256"]:
        raise ValueError("offline script changed after freeze")
    output = Path(plan["output"])
    output.mkdir(exist_ok=False)
    state = output / "state"
    state.mkdir()
    os.environ["CRAWL4_AI_BASE_DIRECTORY"] = str(state.resolve())
    os.environ["CRAWL4AI_BASE_DIRECTORY"] = str(state.resolve())
    current = api["admit_runtime"](Path(plan["extension_root"]))
    if current != json.loads(Path(plan["runtime_manifest"]).read_bytes()):
        raise ValueError("offline parser runtime differs")
    from bs4 import BeautifulSoup
    from crawl4ai.processors.pdf import PDFContentScrapingStrategy

    local = output / "owned-input.pdf"
    shutil.copyfile(source, local)
    try:
        with patch.object(
            socket.socket,
            "connect",
            side_effect=RuntimeError("offline extraction forbids network"),
        ):
            result = PDFContentScrapingStrategy(
                max_pdf_bytes=8 * 1024 * 1024,
                max_pdf_pages=64,
                batch_size=1,
                extract_images=False,
                save_images_locally=False,
            ).scrap(str(local.resolve()), "")
        api["write_json"](output / "native-scraping.json", api["native_json"](result))
        soup = BeautifulSoup(result.cleaned_html, "html.parser")
        pages = [
            {"pdf_page": int(page["data-page"]), "text": page.get_text(" ", strip=True)}
            for page in soup.select("div.pdf-page")
        ]
        api["write_json"](output / "pages.json", pages)
        if len(pages) != 52 or [page["pdf_page"] for page in pages] != list(
            range(1, 53)
        ):
            raise ValueError(
                "native extraction does not cover the complete retained 52-page document"
            )
        if api["digest"](source) != plan["source_sha256"]:
            raise ValueError("retained original changed during extraction")
        api["write_json"](
            output / "summary.json",
            {
                "source_sha256": plan["source_sha256"],
                "native_success": result.success,
                "pdf_pages": len(pages),
                "native_sha256": api["digest"](output / "native-scraping.json"),
                "pages_sha256": api["digest"](output / "pages.json"),
                "public_requests": 0,
                "original_unchanged": True,
                "new_source_capture": False,
            },
        )
    finally:
        local.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
