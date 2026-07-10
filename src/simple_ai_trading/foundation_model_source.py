"""Hash-pinned source provisioning for optional financial foundation models."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import requests

from .storage import write_json_atomic


KRONOS_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
KRONOS_REPOSITORY = "https://github.com/shiyu-coder/Kronos"
KRONOS_RAW_ROOT = (
    "https://raw.githubusercontent.com/shiyu-coder/Kronos/" f"{KRONOS_COMMIT}"
)


@dataclass(frozen=True)
class PinnedSourceFile:
    relative_path: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return f"{KRONOS_RAW_ROOT}/{self.relative_path}"

    def asdict(self) -> dict[str, object]:
        return {**asdict(self), "url": self.url}


KRONOS_SOURCE_FILES = (
    PinnedSourceFile(
        "model/__init__.py",
        412,
        "f8f856ca3fedadcaac97e196be23d1aeda1c3c9ffe8903d66d43ea3bcac6240c",
    ),
    PinnedSourceFile(
        "model/kronos.py",
        30_133,
        "0a5f90282e2039c2de0771473419715c845def154896dbd0f5747837e6241032",
    ),
    PinnedSourceFile(
        "model/module.py",
        23_426,
        "a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f",
    ),
    PinnedSourceFile(
        "LICENSE",
        1_062,
        "acb2d194d378204e5f2be4dcd24d39ecac437903620c790c3315a96dab388fdc",
    ),
)

FetchSource = Callable[[str, float, int], bytes]


@dataclass(frozen=True)
class FoundationSourceReport:
    provider: str
    repository: str
    commit: str
    source_root: str
    verified: bool
    downloaded: tuple[str, ...]
    files: tuple[dict[str, object], ...]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def default_foundation_cache_root() -> Path:
    """Return a per-user cache path without writing to the repository."""

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "SimpleAITrading" / "foundation-models"
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home) / "simple-ai-trading" / "foundation-models"
    return Path.home() / ".cache" / "simple-ai-trading" / "foundation-models"


def kronos_source_root(cache_root: str | Path | None = None) -> Path:
    base = Path(cache_root) if cache_root is not None else default_foundation_cache_root()
    return base / "kronos" / KRONOS_COMMIT


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_payload(spec: PinnedSourceFile, payload: bytes) -> None:
    if len(payload) != spec.size:
        raise RuntimeError(
            f"Kronos source size mismatch for {spec.relative_path}: "
            f"expected {spec.size}, received {len(payload)}"
        )
    digest = _sha256(payload)
    if digest != spec.sha256:
        raise RuntimeError(
            f"Kronos source SHA-256 mismatch for {spec.relative_path}: "
            f"expected {spec.sha256}, received {digest}"
        )


def _download_source(url: str, timeout_seconds: float, maximum_bytes: int) -> bytes:
    if not url.startswith("https://"):
        raise RuntimeError("foundation-model source downloads require HTTPS")
    with requests.get(
        url,
        timeout=max(0.1, float(timeout_seconds)),
        stream=True,
        headers={"User-Agent": "simple-ai-trading/0.1.0-beta.1"},
    ) as response:
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > maximum_bytes:
            raise RuntimeError(f"foundation-model source response exceeds {maximum_bytes} bytes")
        chunks: list[bytes] = []
        received = 0
        for chunk in response.iter_content(chunk_size=16_384):
            if not chunk:
                continue
            received += len(chunk)
            if received > maximum_bytes:
                raise RuntimeError(f"foundation-model source response exceeds {maximum_bytes} bytes")
            chunks.append(chunk)
        return b"".join(chunks)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _verified_report(root: Path, downloaded: tuple[str, ...]) -> FoundationSourceReport:
    files = tuple(spec.asdict() for spec in KRONOS_SOURCE_FILES)
    return FoundationSourceReport(
        provider="kronos",
        repository=KRONOS_REPOSITORY,
        commit=KRONOS_COMMIT,
        source_root=str(root),
        verified=True,
        downloaded=downloaded,
        files=files,
    )


def verify_kronos_source(cache_root: str | Path | None = None) -> FoundationSourceReport:
    """Verify every executable upstream file against the pinned source contract."""

    root = kronos_source_root(cache_root)
    for spec in KRONOS_SOURCE_FILES:
        target = root / Path(spec.relative_path)
        if not target.is_file():
            raise RuntimeError(f"Kronos source file is missing: {spec.relative_path}")
        _validate_payload(spec, target.read_bytes())
    return _verified_report(root, ())


def provision_kronos_source(
    cache_root: str | Path | None = None,
    *,
    timeout_seconds: float = 30.0,
    repair: bool = False,
    fetch_source: FetchSource | None = None,
) -> FoundationSourceReport:
    """Download missing pinned source files and fail closed on local modification.

    ``repair`` must be explicit before an existing mismatched file is replaced. This
    prevents an unnoticed cache mutation from becoming executable model code.
    """

    root = kronos_source_root(cache_root)
    fetch = fetch_source or _download_source
    downloaded: list[str] = []
    for spec in KRONOS_SOURCE_FILES:
        target = root / Path(spec.relative_path)
        if target.exists():
            if not target.is_file():
                raise RuntimeError(f"Kronos source path is not a regular file: {spec.relative_path}")
            try:
                _validate_payload(spec, target.read_bytes())
                continue
            except RuntimeError:
                if not repair:
                    raise
        payload = fetch(spec.url, max(0.1, float(timeout_seconds)), spec.size)
        _validate_payload(spec, payload)
        _write_bytes_atomic(target, payload)
        downloaded.append(spec.relative_path)

    verify_kronos_source(cache_root)
    report = _verified_report(root, tuple(downloaded))
    write_json_atomic(
        root / "source-manifest.json",
        report.asdict(),
        sort_keys=True,
    )
    return report


__all__ = [
    "FoundationSourceReport",
    "KRONOS_COMMIT",
    "KRONOS_REPOSITORY",
    "KRONOS_SOURCE_FILES",
    "default_foundation_cache_root",
    "kronos_source_root",
    "provision_kronos_source",
    "verify_kronos_source",
]
