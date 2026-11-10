"""Extract a frozen public PDF through the admitted native Crawl4AI PDF API.

Run only under a bounded owned-process controller. The separate parser extension,
runtime inventory, worker bytes and one-use plan must be admitted before access.
This worker installs nothing, launches no browser and follows no document links.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from urllib.parse import urlsplit


MAXIMUM_BYTES = 8 * 1024 * 1024


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def record(path: Path, phase: str, **values) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                {
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                    "phase": phase,
                    **values,
                },
                sort_keys=True,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def extension_inventory(root: Path) -> dict[str, str]:
    root = root.resolve(strict=True)
    expected = {"pypdf", "pypdf-7.0.0.dist-info", ".lock"}
    if {path.name for path in root.iterdir()} - expected:
        raise ValueError("PDF extension contains an unexpected top-level entry")
    files, total = {}, 0
    for path in root.rglob("*"):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("PDF extension contains an alias")
        if path.is_file():
            total += path.stat().st_size
            files[path.relative_to(root).as_posix()] = digest(path)
    if not files or len(files) > 512 or total > MAXIMUM_BYTES:
        raise ValueError("PDF extension exceeds its declared inventory bounds")
    return files


def admit_runtime(extension: Path) -> dict:
    inventory = extension_inventory(extension)
    sys.path.insert(0, str(extension.resolve()))
    import crawl4ai
    import pypdf
    from importlib.metadata import version

    if version("pypdf") != "7.0.0" or not Path(pypdf.__file__).resolve().is_relative_to(
        extension.resolve()
    ):
        raise ValueError("PDF parser identity differs from the isolated extension")
    if version("crawl4ai") != "0.9.5+flitzzip.portable2":
        raise ValueError("PDF feature requires the reviewed Crawl4AI version")
    package = Path(crawl4ai.__file__).resolve().parent
    names = (
        "__init__.py",
        "async_configs.py",
        "async_webcrawler.py",
        "utils.py",
        "models.py",
        "markdown_generation_strategy.py",
        "processors/pdf/__init__.py",
        "processors/pdf/processor.py",
        "processors/pdf/utils.py",
    )
    return {
        "extension_files": inventory,
        "pypdf_version": version("pypdf"),
        "crawl4ai_version": version("crawl4ai"),
        "crawl4ai_package": str(package),
        "crawl4ai_files": {name: digest(package / name) for name in names},
    }


def validate_url(url: str, expected: str, *, loopback: bool = False) -> None:
    parsed = urlsplit(url)
    if (
        url != expected
        or parsed.username
        or parsed.password
        or parsed.fragment
        or not parsed.hostname
    ):
        raise ValueError("PDF request differs from the declared source")
    if loopback:
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
            raise ValueError("fixture request must stay on the owned loopback server")
        return
    if parsed.scheme != "https" or parsed.port not in {None, 443}:
        raise ValueError("PDF research requires ordinary public HTTPS")
    addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(
        not ipaddress.ip_address(item[4][0]).is_global for item in addresses
    ):
        raise ValueError("PDF source does not resolve exclusively to public addresses")


def native_json(value):
    if hasattr(value, "model_dump"):
        # Native failure results can contain a property object. Preserve the
        # failure fields and mark unsupported values instead of losing the result.
        return value.model_dump(
            mode="json",
            fallback=lambda item: {
                "unserializable_native_type": f"{type(item).__module__}.{type(item).__qualname__}"
            },
        )
    if is_dataclass(value):
        return asdict(value)
    raise TypeError("unrecognized native PDF result")


async def capture(plan: dict, output: Path) -> dict:
    from bs4 import BeautifulSoup
    import requests
    from pypdf import PdfReader
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
    from crawl4ai.processors.pdf import (
        PDFContentScrapingStrategy,
        PDFCrawlerStrategy,
    )
    from crawl4ai.utils import RobotsParser, get_home_folder

    url, loopback = plan["url"], not plan["source_access"]
    maximum_pages, minimum_characters = (
        plan["maximum_pages"],
        plan["minimum_characters"],
    )
    state, temporary = output / "state", output / "temporary"
    if not Path(get_home_folder()).resolve().is_relative_to(state.resolve()):
        raise RuntimeError(
            "native crawler cache is outside the owned research directory"
        )
    journal = output / "journal.jsonl"
    agent = requests.utils.default_user_agent()
    receipts, attempts, page_characters = [], set(), []
    raw_pages = None
    parsed = urlsplit(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    def fetch(target: str, kind: str, ceiling: int) -> tuple[dict, Path]:
        expected = robots_url if kind == "robots" else url
        validate_url(target, expected, loopback=loopback)
        if kind in attempts:
            raise RuntimeError("one-use PDF or robots request already consumed")
        attempts.add(kind)
        record(journal, "request_started", kind=kind, url=target)
        raw_path = output / (kind + ".raw")
        with raw_path.open("xb") as stream, requests.Session() as session:
            session.trust_env = False
            session.headers.update({"User-Agent": agent, "Accept-Encoding": "identity"})
            with session.get(
                target, stream=True, timeout=(10, 10), allow_redirects=False
            ) as response:
                peer = PDFContentScrapingStrategy._peer_ip(response)
                receipt = {
                    "kind": kind,
                    "url": response.url,
                    "status_code": response.status_code,
                    "headers": {
                        key.lower(): value for key, value in response.headers.items()
                    },
                    "peer_ip": peer,
                }
                receipts.append(receipt)
                write_json(output / (kind + "-http.json"), receipt)
                valid_peer = peer is not None and (
                    peer == "127.0.0.1"
                    if loopback
                    else ipaddress.ip_address(peer).is_global
                )
                if response.url != target or not valid_peer:
                    raise ValueError(
                        "PDF response URL or connected peer is outside source scope"
                    )
                if receipt["headers"].get(
                    "content-encoding", "identity"
                ).lower() not in {"identity", ""}:
                    raise ValueError(
                        "encoded source body is outside this capture profile"
                    )
                deadline, count = time.monotonic() + 30, 0
                try:
                    for chunk in response.iter_content(8192):
                        # Preserve a bounded prefix, including one excess byte on overflow.
                        kept = chunk[: ceiling + 1 - count]
                        stream.write(kept)
                        count += len(kept)
                        if count > ceiling or time.monotonic() > deadline:
                            raise ValueError(
                                "PDF source body exceeds its byte or time budget"
                            )
                    # EOF can arrive after the last full chunk without another
                    # loop iteration. It is part of the body observation budget.
                    if time.monotonic() > deadline:
                        raise TimeoutError("PDF source EOF exceeded its body budget")
                finally:
                    stream.flush()
                    os.fsync(stream.fileno())
                    record(journal, "body_terminal", kind=kind, bytes=count)
        record(journal, "response_retained", kind=kind, sha256=digest(raw_path))
        return receipt, raw_path

    class StrictRobots(RobotsParser):
        async def can_fetch(self, target, user_agent="*"):
            if target != url or user_agent != agent:
                raise ValueError("robots policy differs from the actual PDF request")
            receipt, raw_path = await asyncio.to_thread(
                fetch, robots_url, "robots", 512 * 1024
            )
            body = raw_path.read_bytes()
            if receipt["status_code"] in {404, 410}:
                rules = ""
            elif receipt["status_code"] == 200:
                rules = body.decode("utf-8-sig")
                if rules.lstrip().lower().startswith(("<html", "<!doctype")):
                    raise ValueError("robots response is HTML instead of policy text")
            else:
                raise ValueError("robots policy could not be qualified")
            # A fresh owned cache prevents the native parser's fail-open network path.
            self._cache_rules(parsed.netloc, rules)
            allowed = await super().can_fetch(target, user_agent)
            write_json(
                output / "robots-decision.json",
                {
                    "allowed": allowed,
                    "user_agent": user_agent,
                    "raw_sha256": digest(raw_path),
                },
            )
            return allowed

    class SourcePDF(PDFContentScrapingStrategy):
        def _get_pdf_path(self, target):
            nonlocal raw_pages
            receipt, raw_path = fetch(target, "document", MAXIMUM_BYTES)
            if receipt["status_code"] != 200:
                raise ValueError("actual PDF download did not return HTTP 200")
            raw = raw_path.read_bytes()
            if not raw.startswith(b"%PDF-"):
                raise ValueError("download is not a PDF")
            with raw_path.open("rb") as stream:
                reader = PdfReader(stream)
                if reader.is_encrypted:
                    raise ValueError(
                        "encrypted documents are outside this source profile"
                    )
                raw_pages = len(reader.pages)
            if not 0 < raw_pages <= maximum_pages:
                raise ValueError("complete PDF exceeds its declared page budget")
            # The native scraper owns and deletes only this temporary copy.
            with tempfile.NamedTemporaryFile(
                dir=temporary, suffix=".pdf", delete=False
            ) as stream:
                stream.write(raw)
                return stream.name

        def scrap(self, target, html, **kwargs):
            result = super().scrap(target, html, **kwargs)
            write_json(output / "native-scraping.json", native_json(result))
            soup = BeautifulSoup(result.cleaned_html, "html.parser")
            for page in soup.select("div.pdf-page"):
                page_characters.append(
                    {
                        "page": int(page["data-page"]),
                        "characters": len(page.get_text(" ", strip=True)),
                    }
                )
            return result

    scraper = SourcePDF(
        max_pdf_bytes=MAXIMUM_BYTES,
        max_pdf_pages=maximum_pages,
        max_redirects=0,
        batch_size=1,
        extract_images=False,
        save_images_locally=False,
    )
    config = CrawlerRunConfig(
        scraping_strategy=scraper,
        cache_mode=CacheMode.DISABLED,
        check_robots_txt=True,
        max_retries=0,
        page_timeout=45000,
    )
    async with AsyncWebCrawler(
        crawler_strategy=PDFCrawlerStrategy(),
        config=BrowserConfig(headless=True, verbose=False, user_agent=agent),
        base_directory=str(state.resolve()),
    ) as crawler:
        crawler.robots_parser = StrictRobots(cache_dir=str(state / "robots"))
        result = await crawler.arun(url=url, config=config)
    write_json(output / "native-result.json", native_json(result))
    markdown = result.markdown.raw_markdown if result.markdown else ""
    document = next(
        (receipt for receipt in receipts if receipt["kind"] == "document"), {}
    )
    decision_path = output / "robots-decision.json"
    decision = json.loads(decision_path.read_bytes()) if decision_path.exists() else {}
    gates = {
        "native_success": result.success is True,
        "actual_http_200": document.get("status_code") == 200,
        "robots_allowed": decision.get("allowed") is True,
        "complete_pages": raw_pages is not None
        and [row["page"] for row in page_characters] == list(range(1, raw_pages + 1))
        and all(row["characters"] >= 20 for row in page_characters),
        "substantive_text": len(markdown) >= minimum_characters,
    }
    summary = {
        "admitted": all(gates.values()),
        "gates": gates,
        "native_status_is_not_transport_evidence": True,
        "raw_pages": raw_pages,
        "page_characters": page_characters,
        "markdown_characters": len(markdown),
        "document_requests": sum(r["kind"] == "document" for r in receipts),
    }
    write_json(output / "admission.json", summary)
    return summary


def main() -> None:
    # Isolated (-I) interpreters ignore PYTHONIOENCODING on Windows pipes.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runtime-info", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    if not args.runtime_info:
        contract = dict(plan)
        recorded = contract.pop("plan_sha256")
        if (
            canonical_digest(contract) != recorded
            or digest(Path(__file__)) != plan["worker_sha256"]
        ):
            raise ValueError("PDF plan or worker bytes changed after freeze")
        if type(plan["source_access"]) is not bool:
            raise ValueError("source_access must be explicit")
        if (
            type(plan["maximum_pages"]) is not int
            or not 1 <= plan["maximum_pages"] <= 64
        ):
            raise ValueError("PDF page budget is outside the admitted profile")
        if (
            type(plan["minimum_characters"]) is not int
            or plan["minimum_characters"] < 100
        ):
            raise ValueError("PDF text threshold is outside the admitted profile")
    output = Path(plan["output"]).resolve()
    output.mkdir(exist_ok=False)
    state, temporary = output / "state", output / "temporary"
    state.mkdir()
    temporary.mkdir()
    # This must precede Crawl4AI imports, including runtime qualification.
    os.environ["CRAWL4_AI_BASE_DIRECTORY"] = str(state)
    os.environ["CRAWL4AI_BASE_DIRECTORY"] = str(state)
    os.environ["TEMP"] = os.environ["TMP"] = str(temporary)
    tempfile.tempdir = str(temporary)
    journal = output / "journal.jsonl"
    record(journal, "worker_started", runtime_info=args.runtime_info)
    try:
        runtime = admit_runtime(Path(plan["extension_root"]))
        if args.runtime_info:
            write_json(Path(plan["runtime_output"]), runtime)
            record(journal, "runtime_qualified")
            return
        manifest = Path(plan["runtime_manifest"])
        if digest(manifest) != plan["runtime_sha256"] or runtime != json.loads(
            manifest.read_bytes()
        ):
            raise RuntimeError("native PDF runtime changed after admission")
        summary = asyncio.run(capture(plan, output))
        record(journal, "worker_terminal", admitted=summary["admitted"])
        print(json.dumps(summary))
    except BaseException as failure:
        record(
            journal,
            "worker_failed",
            error_type=type(failure).__name__,
            error=str(failure),
        )
        raise


if __name__ == "__main__":
    main()
