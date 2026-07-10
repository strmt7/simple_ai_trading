from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from simple_ai_trading import foundation_model_source as source
from simple_ai_trading.foundation_model_source import (
    KRONOS_COMMIT,
    kronos_source_root,
    provision_kronos_source,
    verify_kronos_source,
)


@pytest.fixture
def pinned_payloads(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    payload_by_path = {
        "model/__init__.py": b"from .kronos import Kronos\n",
        "model/kronos.py": b"class Kronos:\n    pass\n",
        "model/module.py": b"VALUE = 1\n",
        "LICENSE": b"MIT test fixture\n",
    }
    specs = tuple(
        source.PinnedSourceFile(
            relative_path=relative_path,
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        for relative_path, payload in payload_by_path.items()
    )
    monkeypatch.setattr(source, "KRONOS_SOURCE_FILES", specs)
    return {spec.url: payload_by_path[spec.relative_path] for spec in specs}


def test_provision_and_verify_hash_pinned_kronos_source(
    tmp_path: Path,
    pinned_payloads: dict[str, bytes],
) -> None:
    fetched: list[str] = []

    def fetch(url: str, _timeout: float, maximum: int) -> bytes:
        fetched.append(url)
        payload = pinned_payloads[url]
        assert len(payload) <= maximum
        return payload

    report = provision_kronos_source(tmp_path, fetch_source=fetch)
    verified = verify_kronos_source(tmp_path)

    assert report.verified is True
    assert report.commit == KRONOS_COMMIT
    assert len(report.downloaded) == len(source.KRONOS_SOURCE_FILES)
    assert len(fetched) == len(source.KRONOS_SOURCE_FILES)
    assert verified.downloaded == ()
    assert Path(report.source_root) == kronos_source_root(tmp_path)
    assert (Path(report.source_root) / "source-manifest.json").is_file()


def test_existing_modified_source_fails_closed_without_repair(
    tmp_path: Path,
    pinned_payloads: dict[str, bytes],
) -> None:

    def fetch(url: str, _timeout: float, _maximum: int) -> bytes:
        return pinned_payloads[url]

    report = provision_kronos_source(tmp_path, fetch_source=fetch)
    modified = Path(report.source_root) / source.KRONOS_SOURCE_FILES[0].relative_path
    modified.write_bytes(modified.read_bytes() + b"\n# changed\n")

    with pytest.raises(RuntimeError, match="size mismatch"):
        provision_kronos_source(tmp_path, fetch_source=fetch)
    with pytest.raises(RuntimeError, match="size mismatch"):
        verify_kronos_source(tmp_path)


def test_explicit_repair_restores_modified_source(
    tmp_path: Path,
    pinned_payloads: dict[str, bytes],
) -> None:

    def fetch(url: str, _timeout: float, _maximum: int) -> bytes:
        return pinned_payloads[url]

    report = provision_kronos_source(tmp_path, fetch_source=fetch)
    modified = Path(report.source_root) / source.KRONOS_SOURCE_FILES[1].relative_path
    modified.write_bytes(b"modified")

    repaired = provision_kronos_source(tmp_path, repair=True, fetch_source=fetch)

    assert repaired.downloaded == (source.KRONOS_SOURCE_FILES[1].relative_path,)
    assert verify_kronos_source(tmp_path).verified is True


def test_download_with_wrong_digest_never_reaches_cache(
    tmp_path: Path,
    pinned_payloads: dict[str, bytes],
) -> None:
    def fetch(_url: str, _timeout: float, maximum: int) -> bytes:
        return b"x" * maximum

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        provision_kronos_source(tmp_path, fetch_source=fetch)

    assert not (
        kronos_source_root(tmp_path) / source.KRONOS_SOURCE_FILES[0].relative_path
    ).exists()
