from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_live_manifest import (
    build_polymarket_live_implementation_manifest,
    load_polymarket_live_implementation_manifest,
)


SOURCE_COMMIT = "b" * 40


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write(path: Path, payload: dict[str, object]) -> str:
    raw = _canonical(payload)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope="module")
def implementation_manifest() -> dict[str, object]:
    return build_polymarket_live_implementation_manifest(
        source_commit=SOURCE_COMMIT,
    )


def test_manifest_verifies_exact_installed_runtime(
    tmp_path: Path,
    implementation_manifest: dict[str, object],
) -> None:
    path = tmp_path / "implementation.json"
    file_sha256 = _write(path, implementation_manifest)

    verified = load_polymarket_live_implementation_manifest(
        path,
        expected_file_sha256=file_sha256,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert verified.source_commit == SOURCE_COMMIT
    assert verified.manifest_sha256 == implementation_manifest["manifest_sha256"]
    assert "polymarket_live.py" in verified.source_files
    assert verified.dependency_versions == {
        "polymarket-client": "0.2.0",
        "py-clob-client-v2": "1.1.0",
    }


def test_manifest_rejects_runtime_hash_claim_even_when_manifest_is_rehashed(
    tmp_path: Path,
    implementation_manifest: dict[str, object],
) -> None:
    payload = deepcopy(implementation_manifest)
    files = list(payload["files"])
    target = dict(files[0])
    target["sha256"] = "0" * 64
    files[0] = target
    payload["files"] = files
    body = dict(payload)
    body.pop("manifest_sha256")
    payload["manifest_sha256"] = _canonical_sha(body)
    path = tmp_path / "implementation.json"
    file_sha256 = _write(path, payload)

    with pytest.raises(ValueError, match="installed implementation file differs"):
        load_polymarket_live_implementation_manifest(
            path,
            expected_file_sha256=file_sha256,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_manifest_rejects_unlisted_or_extra_installed_source_contract(
    tmp_path: Path,
    implementation_manifest: dict[str, object],
) -> None:
    payload = deepcopy(implementation_manifest)
    files = list(payload["files"])
    files.append({"path": "zz_unlisted.py", "sha256": "0" * 64})
    payload["files"] = files
    body = dict(payload)
    body.pop("manifest_sha256")
    payload["manifest_sha256"] = _canonical_sha(body)
    path = tmp_path / "implementation.json"
    file_sha256 = _write(path, payload)

    with pytest.raises(ValueError, match="installed implementation file set differs"):
        load_polymarket_live_implementation_manifest(
            path,
            expected_file_sha256=file_sha256,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_manifest_rejects_source_commit_or_noncanonical_bytes(
    tmp_path: Path,
    implementation_manifest: dict[str, object],
) -> None:
    path = tmp_path / "implementation.json"
    file_sha256 = _write(path, implementation_manifest)
    with pytest.raises(ValueError, match="source commit differs"):
        load_polymarket_live_implementation_manifest(
            path,
            expected_file_sha256=file_sha256,
            expected_source_commit="c" * 40,
        )

    raw = path.read_bytes() + b"\n"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="not canonical"):
        load_polymarket_live_implementation_manifest(
            path,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            expected_source_commit=SOURCE_COMMIT,
        )
