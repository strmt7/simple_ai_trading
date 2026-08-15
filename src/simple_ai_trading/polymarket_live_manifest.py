"""Exact installed-runtime manifest for independent Polymarket live authority."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Sequence


POLYMARKET_LIVE_IMPLEMENTATION_MANIFEST_SCHEMA_VERSION = (
    "polymarket-live-implementation-manifest-v1"
)
_DISTRIBUTION_NAME = "simple-ai-trading"
_ENTRYPOINT = "simple_ai_trading.entrypoint:main"
_MAXIMUM_MANIFEST_BYTES = 512 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_DEPENDENCIES = (
    "polymarket-client",
    "py-clob-client-v2",
)
_REQUIRED_RUNTIME_FILES = frozenset(
    {
        "__init__.py",
        "entrypoint.py",
        "polymarket.py",
        "polymarket_autonomous.py",
        "polymarket_autonomous_runtime.py",
        "polymarket_fees.py",
        "polymarket_live.py",
        "polymarket_live_activation.py",
        "polymarket_live_cli.py",
        "polymarket_live_manifest.py",
        "polymarket_live_promotion.py",
        "polymarket_live_qualification.py",
        "polymarket_live_risk.py",
        "polymarket_live_runtime.py",
        "polymarket_live_settlement.py",
        "polymarket_live_stop.py",
        "polymarket_live_v2.py",
        "polymarket_runtime_control.py",
    }
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(
                "Polymarket implementation manifest JSON contains duplicate keys"
            )
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Polymarket implementation manifest JSON contains {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return selected


def _source_commit(value: object) -> str:
    selected = str(value or "").strip().lower()
    if _GIT_COMMIT.fullmatch(selected) is None:
        raise ValueError("Polymarket implementation source commit is invalid")
    return selected


def _safe_python_path(value: object) -> str:
    text = str(value or "").strip()
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.suffix != ".py"
    ):
        raise ValueError("Polymarket implementation file path is invalid")
    return path.as_posix()


def _package_root(value: str | Path | None) -> Path:
    selected = Path(__file__).resolve().parent if value is None else Path(value)
    if selected.is_symlink():
        raise ValueError("Polymarket implementation package root cannot be a symlink")
    resolved = selected.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("Polymarket implementation package root is not a directory")
    return resolved


def _python_sources(root: Path) -> tuple[tuple[str, Path], ...]:
    sources: list[tuple[str, Path]] = []
    for path in root.rglob("*.py"):
        if path.is_symlink() or any(
            parent.is_symlink() for parent in path.parents if parent != root
        ):
            raise ValueError("Polymarket implementation source cannot be a symlink")
        resolved = path.resolve(strict=True)
        if root not in resolved.parents:
            raise ValueError("Polymarket implementation source escaped its package")
        relative = resolved.relative_to(root).as_posix()
        sources.append((_safe_python_path(relative), resolved))
    sources.sort(key=lambda item: item[0])
    paths = tuple(path for path, _ in sources)
    if not sources or len(set(paths)) != len(paths):
        raise ValueError("Polymarket implementation source set is invalid")
    if not _REQUIRED_RUNTIME_FILES <= set(paths):
        raise ValueError("Polymarket implementation runtime files are incomplete")
    return tuple(sources)


def _distribution_version(distribution: str) -> str:
    try:
        selected = version(distribution)
    except PackageNotFoundError as exc:
        raise ValueError(
            f"required Polymarket live distribution is not installed: {distribution}"
        ) from exc
    if not selected or len(selected) > 128:
        raise ValueError("Polymarket implementation distribution version is invalid")
    return selected


def _installed_dependencies() -> dict[str, str]:
    return {
        dependency: _distribution_version(dependency)
        for dependency in _REQUIRED_DEPENDENCIES
    }


@dataclass(frozen=True, slots=True)
class VerifiedPolymarketLiveImplementationManifest:
    path: Path
    file_sha256: str
    manifest_sha256: str
    source_commit: str
    package_version: str
    dependency_versions: Mapping[str, str]
    source_files: tuple[str, ...]


def build_polymarket_live_implementation_manifest(
    *,
    source_commit: str,
    package_root: str | Path | None = None,
) -> dict[str, object]:
    """Build a canonical manifest body from the exact installed source bytes."""

    commit = _source_commit(source_commit)
    root = _package_root(package_root)
    files = [
        {"path": relative, "sha256": _sha256_file(path)}
        for relative, path in _python_sources(root)
    ]
    body: dict[str, object] = {
        "schema_version": POLYMARKET_LIVE_IMPLEMENTATION_MANIFEST_SCHEMA_VERSION,
        "source_commit": commit,
        "venue": "polymarket",
        "protocol_version": 2,
        "authority_scope": "live",
        "distribution": _DISTRIBUTION_NAME,
        "package_version": _distribution_version(_DISTRIBUTION_NAME),
        "entrypoint": _ENTRYPOINT,
        "dependencies": _installed_dependencies(),
        "files": files,
    }
    return {**body, "manifest_sha256": _canonical_sha256(body)}


def _validated_files(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("Polymarket implementation files must be an array")
    output: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise ValueError("Polymarket implementation file entry is invalid")
        output.append(
            (
                _safe_python_path(item["path"]),
                _sha(item["sha256"], name="implementation source hash"),
            )
        )
    paths = tuple(path for path, _ in output)
    if (
        not output
        or paths != tuple(sorted(paths))
        or len(set(paths)) != len(paths)
        or not _REQUIRED_RUNTIME_FILES <= set(paths)
    ):
        raise ValueError("Polymarket implementation file set is invalid")
    return tuple(output)


def _validated_dependencies(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(_REQUIRED_DEPENDENCIES):
        raise ValueError("Polymarket implementation dependency set is invalid")
    selected = {str(key): str(item or "").strip() for key, item in value.items()}
    if any(not item or len(item) > 128 for item in selected.values()):
        raise ValueError("Polymarket implementation dependency version is invalid")
    return dict(sorted(selected.items()))


def load_polymarket_live_implementation_manifest(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_source_commit: str,
    package_root: str | Path | None = None,
) -> VerifiedPolymarketLiveImplementationManifest:
    """Verify strict manifest bytes and the complete installed Python runtime."""

    selected = Path(path)
    if selected.is_symlink():
        raise ValueError("Polymarket implementation manifest cannot be a symlink")
    raw = selected.read_bytes()
    if not raw or len(raw) > _MAXIMUM_MANIFEST_BYTES:
        raise ValueError("Polymarket implementation manifest file size is invalid")
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if file_sha256 != _sha(
        expected_file_sha256,
        name="expected implementation manifest file hash",
    ):
        raise ValueError("Polymarket implementation manifest file hash differs")
    try:
        payload = json.loads(
            raw.decode("ascii", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Polymarket implementation manifest is not strict JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Polymarket implementation manifest is not an object")
    if _canonical_json(payload).encode("ascii") != raw:
        raise ValueError("Polymarket implementation manifest is not canonical")
    required_keys = {
        "schema_version",
        "manifest_sha256",
        "source_commit",
        "venue",
        "protocol_version",
        "authority_scope",
        "distribution",
        "package_version",
        "entrypoint",
        "dependencies",
        "files",
    }
    if set(payload) != required_keys:
        raise ValueError("Polymarket implementation manifest schema is invalid")
    if (
        payload["schema_version"]
        != POLYMARKET_LIVE_IMPLEMENTATION_MANIFEST_SCHEMA_VERSION
        or payload["venue"] != "polymarket"
        or payload["protocol_version"] != 2
        or payload["authority_scope"] != "live"
        or payload["distribution"] != _DISTRIBUTION_NAME
        or payload["entrypoint"] != _ENTRYPOINT
    ):
        raise ValueError("Polymarket implementation manifest scope is invalid")
    commit = _source_commit(payload["source_commit"])
    if commit != _source_commit(expected_source_commit):
        raise ValueError("Polymarket implementation source commit differs")
    claimed = _sha(payload["manifest_sha256"], name="implementation manifest hash")
    body = dict(payload)
    body.pop("manifest_sha256")
    if _canonical_sha256(body) != claimed:
        raise ValueError("Polymarket implementation manifest hash differs")
    expected_files = _validated_files(payload["files"])
    expected_dependencies = _validated_dependencies(payload["dependencies"])
    root = _package_root(package_root)
    actual_sources = _python_sources(root)
    actual_paths = tuple(path for path, _ in actual_sources)
    if actual_paths != tuple(path for path, _ in expected_files):
        raise ValueError("Polymarket installed implementation file set differs")
    for (relative, source), (expected_relative, expected_sha256) in zip(
        actual_sources,
        expected_files,
        strict=True,
    ):
        if relative != expected_relative or _sha256_file(source) != expected_sha256:
            raise ValueError(
                f"Polymarket installed implementation file differs: {relative}"
            )
    package_version = str(payload["package_version"] or "").strip()
    if (
        not package_version
        or len(package_version) > 128
        or package_version != _distribution_version(_DISTRIBUTION_NAME)
    ):
        raise ValueError("Polymarket installed package version differs")
    if expected_dependencies != _installed_dependencies():
        raise ValueError("Polymarket installed dependency versions differ")
    return VerifiedPolymarketLiveImplementationManifest(
        path=selected.resolve(strict=True),
        file_sha256=file_sha256,
        manifest_sha256=claimed,
        source_commit=commit,
        package_version=package_version,
        dependency_versions=expected_dependencies,
        source_files=actual_paths,
    )


__all__ = [
    "POLYMARKET_LIVE_IMPLEMENTATION_MANIFEST_SCHEMA_VERSION",
    "VerifiedPolymarketLiveImplementationManifest",
    "build_polymarket_live_implementation_manifest",
    "load_polymarket_live_implementation_manifest",
]
