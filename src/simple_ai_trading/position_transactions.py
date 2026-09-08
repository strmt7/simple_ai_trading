"""Recoverable paired JSON replacements, serialized by the existing intent DB.

The committed redo record is authoritative during interruption. Readers and
writers using this boundary finish that record before observing either file.
External edits, missing enrolled files and unexpected bytes fail closed; they
are not an invitation to restore an arbitrary snapshot. This is crash recovery,
not protection against coordinated filesystem/database rollback or power loss.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .binance_open_intents import BinanceOpenIntentJournal
from .storage import write_bytes_atomic

_FILES = ("open_positions.json", "ledger.json")
_COLUMNS = (
    "id",
    "state",
    "before_open",
    "before_closed",
    "after_open",
    "after_closed",
    "digest",
)


def _digest(bodies: tuple[bytes | None, ...]) -> str:
    digest = hashlib.sha256()
    for body in bodies:
        digest.update(
            b"-1:" if body is None else str(len(body)).encode("ascii") + b":" + body
        )
    return digest.hexdigest()


def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _snapshot(root: Path) -> tuple[bytes | None, bytes | None]:
    return _read(root / _FILES[0]), _read(root / _FILES[1])


def _recover(connection: sqlite3.Connection, root: Path) -> None:
    columns = tuple(
        row[1] for row in connection.execute("PRAGMA table_info(position_replacement)")
    )
    if not columns:
        return
    if columns != _COLUMNS:
        raise ValueError("position transaction schema is not recognized")
    rows = connection.execute("SELECT * FROM position_replacement LIMIT 2").fetchall()
    if len(rows) != 1:
        raise ValueError("position transaction record is missing or ambiguous")
    identity, state, *bodies, digest = rows[0]
    if (
        identity != 1
        or state not in {"PREPARED", "APPLIED"}
        or any(body is not None and not isinstance(body, bytes) for body in bodies)
    ):
        raise ValueError("position transaction record is invalid")
    if digest != _digest(tuple(bodies)):
        raise ValueError("position transaction checksum differs")
    before, after = tuple(bodies[:2]), tuple(bodies[2:])
    current = _snapshot(root)
    if state == "APPLIED":
        if current != after:
            raise ValueError("position ledger differs from committed transaction")
        return
    # Admit BOTH files before touching either. Never overwrite unknown bytes.
    if any(now not in (old, new) for now, old, new in zip(current, before, after)):
        raise ValueError("interrupted position transaction has conflicting files")
    if any(new is None and old is not None for old, new in zip(before, after)):
        raise ValueError("position transaction cannot delete a retained ledger")
    for name, now, new in zip(_FILES, current, after):
        if now != new:
            if new is None:
                raise ValueError("position transaction cannot remove a ledger")
            write_bytes_atomic(root / name, new)
    connection.execute("UPDATE position_replacement SET state='APPLIED' WHERE id=1")


@dataclass(frozen=True)
class PositionTransaction:
    root: Path
    connection: sqlite3.Connection | None
    before: tuple[bytes | None, bytes | None]

    def read(self, path: Path) -> bytes | None:
        for index, name in enumerate(_FILES):
            if path.resolve() == (self.root / name).resolve():
                return self.before[index]
        raise ValueError("position transaction path is outside its ledger pair")

    def replace(self, replacements: dict[str, bytes]) -> None:
        """Commit the exact pair before publishing either JSON projection."""
        if (
            self.connection is None
            or not replacements
            or any(
                name not in _FILES or not isinstance(body, bytes)
                for name, body in replacements.items()
            )
        ):
            raise ValueError("position transaction replacements are invalid")
        connection = self.connection
        after = tuple(
            replacements.get(name, old) for name, old in zip(_FILES, self.before)
        )
        if _snapshot(self.root) != self.before:
            raise ValueError("position files changed during the transaction")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS position_replacement ("
            "id INTEGER PRIMARY KEY CHECK(id=1), state TEXT NOT NULL "
            "CHECK(state IN ('PREPARED', 'APPLIED')), before_open BLOB, "
            "before_closed BLOB, after_open BLOB, after_closed BLOB, digest TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO position_replacement VALUES (1, 'PREPARED', ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET state='PREPARED', "
            "before_open=excluded.before_open, before_closed=excluded.before_closed, "
            "after_open=excluded.after_open, after_closed=excluded.after_closed, "
            "digest=excluded.digest",
            (*self.before, *after, _digest((*self.before, *after))),
        )
        connection.commit()
        # A competing participant may finish this redo first, then transact its
        # own change. Recover the CURRENT row, never replay our stale snapshot.
        connection.execute("BEGIN IMMEDIATE")
        _recover(connection, self.root)


@contextmanager
def position_transaction(
    journal: BinanceOpenIntentJournal, *, write: bool = False
) -> Iterator[PositionTransaction]:
    """Fence participating readers/writers without creating storage on first read."""
    root = Path(journal.path).resolve().parent
    initializing = write and not Path(journal.path).exists()
    if not write and not Path(journal.path).exists():
        before = _snapshot(root)
        # A first writer may have enrolled the pair while we read the files.
        if not Path(journal.path).exists():
            yield PositionTransaction(root, None, before)
            return
    try:
        with closing(journal._connect(create=write, write=True)) as connection:
            try:
                if initializing:
                    # Rollback of the first invalid ledger mutation must not
                    # leave an existing empty SQLite file with no valid schema.
                    # No intent, scope binding or position state is committed here.
                    connection.commit()
                    connection.execute("BEGIN IMMEDIATE")
                _recover(connection, root)
                yield PositionTransaction(root, connection, _snapshot(root))
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
    except (sqlite3.Error, OSError):
        raise ValueError("position transaction storage is unavailable") from None
