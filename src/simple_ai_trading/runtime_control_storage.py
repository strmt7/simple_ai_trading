"""Independent enrollment evidence for an existing runtime-control store.

Records contain no path or credential. They detect missing/replaced enrolled
resources; they are not account ownership, rollback protection or authentication
against an actor able to rewrite every local authority file.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import BinaryIO

from .storage import write_bytes_atomic


ENROLLMENT_SCHEMA = "runtime-control-storage-v1"
MAX_ENROLLMENT_BYTES = 8192
_RESOURCE_NAMES = frozenset({"database", "submission_lock", "close_lock"})


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("runtime enrollment contains duplicate fields")
        result[key] = value
    return result


def validate_parent(path: Path) -> None:
    """Reject redirected directory components before opening an authority file."""
    for parent in reversed(path.absolute().parents):
        info = parent.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 0x400
        ):
            raise ValueError("runtime storage parent must be an ordinary directory")


@dataclass(frozen=True, slots=True)
class FileIdentity:
    device: int
    inode: int

    def __post_init__(self) -> None:
        if (
            type(self.device) is not int
            or self.device < 0
            or type(self.inode) is not int
            or self.inode <= 0
        ):
            raise ValueError("runtime filesystem must provide stable file identities")

    @classmethod
    def from_stat(cls, info: os.stat_result) -> FileIdentity:
        if (
            not stat.S_ISREG(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 0x400
        ):
            raise ValueError("runtime storage resource must be an ordinary file")
        return cls(info.st_dev, info.st_ino)

    @classmethod
    def at(cls, path: Path) -> FileIdentity:
        validate_parent(path)
        return cls.from_stat(path.lstat())

    def verify(self, path: Path, handle: BinaryIO | None = None) -> None:
        if self.at(path) != self or (
            handle is not None and self.from_stat(os.fstat(handle.fileno())) != self
        ):
            raise ValueError("runtime enrolled resource identity changed")

    def payload(self) -> list[int]:
        return [self.device, self.inode]


@dataclass(frozen=True, slots=True)
class RuntimeStorageEnrollment:
    namespace: str
    control_schema_version: str
    store_id: str
    state: str
    database: FileIdentity
    submission_lock: FileIdentity
    close_lock: FileIdentity

    def __post_init__(self) -> None:
        if (
            not isinstance(self.namespace, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,62}", self.namespace) is None
        ):
            raise ValueError("runtime enrollment namespace is invalid")
        if (
            not isinstance(self.control_schema_version, str)
            or not 1 <= len(self.control_schema_version) <= 128
        ):
            raise ValueError("runtime enrollment schema is invalid")
        if (
            not isinstance(self.store_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", self.store_id) is None
        ):
            raise ValueError("runtime enrollment store identity is invalid")
        if self.state not in {"prepared", "ready"}:
            raise ValueError("runtime enrollment state is invalid")
        if not all(
            isinstance(getattr(self, name), FileIdentity) for name in _RESOURCE_NAMES
        ):
            raise ValueError("runtime enrollment resources are invalid")
        if len({getattr(self, name) for name in _RESOURCE_NAMES}) != len(
            _RESOURCE_NAMES
        ):
            raise ValueError("runtime enrollment resources must be independent files")

    def body(self) -> dict[str, object]:
        return {
            "schema_version": ENROLLMENT_SCHEMA,
            "namespace": self.namespace,
            "control_schema_version": self.control_schema_version,
            "store_id": self.store_id,
            "state": self.state,
            "resources": {
                name: getattr(self, name).payload() for name in sorted(_RESOURCE_NAMES)
            },
        }

    def encoded(self) -> bytes:
        body = self.body()
        return (
            _canonical(
                {**body, "record_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
            )
            + b"\n"
        )

    @classmethod
    def read(
        cls, path: Path, *, allow_prepared: bool = False
    ) -> RuntimeStorageEnrollment:
        identity = FileIdentity.at(path)
        with path.open("rb") as handle:
            identity.verify(path, handle)
            raw = handle.read(MAX_ENROLLMENT_BYTES + 1)
            identity.verify(path, handle)
        if len(raw) > MAX_ENROLLMENT_BYTES:
            raise ValueError("runtime enrollment is oversized")
        value = json.loads(raw, object_pairs_hook=_unique_object)
        fields = {
            "schema_version",
            "namespace",
            "control_schema_version",
            "store_id",
            "state",
            "resources",
            "record_sha256",
        }
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or value["schema_version"] != ENROLLMENT_SCHEMA
        ):
            raise ValueError("runtime enrollment fields differ")
        recorded = value.pop("record_sha256")
        if recorded != hashlib.sha256(_canonical(value)).hexdigest():
            raise ValueError("runtime enrollment digest differs")
        resources = value["resources"]
        if not isinstance(resources, dict) or set(resources) != _RESOURCE_NAMES:
            raise ValueError("runtime enrollment resources differ")
        identities = {}
        for name, pair in resources.items():
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("runtime enrollment resource identity is invalid")
            identities[name] = FileIdentity(*pair)
        result = cls(
            value["namespace"],
            value["control_schema_version"],
            value["store_id"],
            value["state"],
            **identities,
        )
        if result.state != "ready" and not allow_prepared:
            raise ValueError(
                "runtime storage enrollment is unfinished; explicit recovery is required"
            )
        return result

    def prepare(self, path: Path) -> None:
        if self.state != "prepared":
            raise ValueError("runtime enrollment must begin prepared")
        validate_parent(path)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(self.encoded())
            handle.flush()
            os.fsync(handle.fileno())

    def finish(self, path: Path) -> None:
        if self.state != "prepared" or self.read(path, allow_prepared=True) != self:
            raise ValueError("runtime prepared enrollment changed")
        write_bytes_atomic(path, replace(self, state="ready").encoded(), mode=0o600)
