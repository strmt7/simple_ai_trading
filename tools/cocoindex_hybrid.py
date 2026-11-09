"""Fuse native CocoIndex results with lexical matches from its same source index."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import sys
import tempfile
import time


SCHEMA = "cocoindex-hybrid-v1"
MAX_ROWS = 250_000
MAX_TEXT_BYTES = 128 * 1024 * 1024
MAX_CACHE_BYTES = 256 * 1024 * 1024
MAX_OUTPUT_BYTES = 65_536
MAX_INPUT_BYTES = 524_288
STOP_WORDS = frozenset(
    "a an and are as at be been by can do does for from has have how in into is it "
    "its of on or our that the their these they this to was we what when where which with".split()
)


def query_tokens(query: str) -> list[str]:
    return list(
        dict.fromkeys(
            t for t in re.findall(r"[^\W_]+", query.casefold()) if t not in STOP_WORDS
        )
    )[:24]


def path_terms(path: str) -> str:
    stem = PurePosixPath(path).stem
    stem = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", stem)
    stem = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", stem)
    return " ".join(re.findall(r"[^\W_]+", stem))


def native_results(text: str, limit: int) -> list[dict[str, object]]:
    payload = json.loads(text)
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise ValueError(
            "Native semantic search did not succeed; lexical fallback is forbidden"
        )
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) > limit:
        raise ValueError("Native semantic result count differs")
    for row in rows:
        if not isinstance(row, dict) or any(
            type(row.get(k)) is not str for k in ("file_path", "language", "content")
        ):
            raise ValueError("Native semantic result schema differs")
        for key in ("start_line", "end_line"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError("Native semantic line metadata differs")
        if row["end_line"] < row["start_line"]:
            raise ValueError("Native semantic line range differs")
        score = row.get("score")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("Native semantic score is invalid")
        safe_path(row["file_path"])
    return rows


def safe_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or ":" in value
    ):
        raise ValueError("Search result path is outside the source namespace")


def source_token(path: Path) -> str:
    values = []
    for item in (path, Path(str(path) + "-wal")):
        stat = item.stat() if item.exists() else None
        if item != path and stat is not None and stat.st_size == 0:
            stat = None
        values.append(None if stat is None else (stat.st_size, stat.st_mtime_ns))
    return json.dumps(values, separators=(",", ":"))


def connect_native(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='code_chunks_vec'"
        ).fetchone()
        if schema is None:
            raise ValueError("Native CocoIndex chunk table is absent")
        if "using vec0" in schema[0].lower():
            import sqlite_vec

            conn.enable_load_extension(True)
            try:
                sqlite_vec.load(conn)
            finally:
                conn.enable_load_extension(False)
        return conn
    except BaseException:
        conn.close()
        raise


@contextmanager
def cache_lock(path: Path):
    if path.is_symlink():
        raise ValueError("Hybrid lock must not be a symbolic link")
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        deadline = time.monotonic() + 10
        while True:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise ValueError(
                        "Hybrid source cache is busy; no unverified result returned"
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def cache_metadata(path: Path) -> dict[str, str]:
    try:
        with closing(
            sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        ) as conn:
            return dict(conn.execute("SELECT key,value FROM cache_metadata"))
    except sqlite3.Error as exc:
        raise ValueError("Unrecognized hybrid cache; refusing to replace it") from exc


def cache_key(
    native: Path, owner: str, source_digest: str, mirror: Path
) -> dict[str, str]:
    return {
        "schema": SCHEMA,
        "owner": owner,
        "source_digest": source_digest,
        "native_token": source_token(native),
        "implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mirror": hashlib.sha256(str(mirror.resolve()).encode()).hexdigest(),
    }


def require_cache(
    native: Path, cache: Path, owner: str, source_digest: str, mirror: Path
) -> None:
    if not cache.is_file():
        raise ValueError("Lexical cache is absent; run the explicit index command")
    metadata = cache_metadata(cache)
    if metadata.get("owner") != owner or metadata.get("schema") != SCHEMA:
        raise ValueError("Hybrid cache belongs to another owner or schema")
    if metadata != cache_key(native, owner, source_digest, mirror):
        raise ValueError("Lexical cache is stale; run the explicit index command")


def ensure_cache(
    native: Path, cache: Path, owner: str, source_digest: str, mirror: Path
) -> None:
    if cache.is_symlink() or cache.resolve() == native.resolve():
        raise ValueError("Hybrid cache target is not an independent owned file")
    token = source_token(native)
    expected = {
        "schema": SCHEMA,
        "owner": owner,
        "source_digest": source_digest,
        "native_token": token,
        "implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mirror": hashlib.sha256(str(mirror.resolve()).encode()).hexdigest(),
    }
    if cache.exists():
        metadata = cache_metadata(cache)
        if metadata.get("owner") != owner or metadata.get("schema") != SCHEMA:
            raise ValueError("Hybrid cache belongs to another owner or schema")
        if metadata == expected:
            return
    with closing(connect_native(native)) as source:
        source.execute("BEGIN")
        for (file_path,) in source.execute(
            "SELECT DISTINCT file_path FROM code_chunks_vec"
        ):
            safe_path(file_path)
            candidate = mirror.joinpath(*PurePosixPath(file_path).parts)
            if not candidate.is_file() or not candidate.resolve().is_relative_to(
                mirror.resolve()
            ):
                raise ValueError(
                    "Native index contains a path outside the approved source mirror"
                )
        count, characters = source.execute(
            "SELECT count(*),sum(length(content)) FROM code_chunks_vec"
        ).fetchone()
        if count > MAX_ROWS or (characters or 0) > MAX_TEXT_BYTES:
            raise ValueError("Hybrid corpus exceeds its bounded build budget")
        if shutil.disk_usage(cache.parent).free < 2 * MAX_CACHE_BYTES:
            raise ValueError("Insufficient private cache disk headroom")
        fd, temporary_name = tempfile.mkstemp(
            prefix="hybrid-", suffix=".sqlite3", dir=cache.parent
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with closing(sqlite3.connect(temporary)) as target, target:
                target.execute(
                    "CREATE TABLE cache_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
                )
                target.execute(
                    "CREATE VIRTUAL TABLE chunks USING fts5(filename,body,path UNINDEXED,language UNINDEXED,start_line UNINDEXED,end_line UNINDEXED,tokenize='porter unicode61')"
                )
                byte_count = 0

                def rows():
                    nonlocal byte_count
                    for path, content, language, start, end in source.execute(
                        "SELECT file_path,content,language,start_line,end_line FROM code_chunks_vec"
                    ):
                        safe_path(path)
                        byte_count += len(content.encode("utf-8"))
                        if byte_count > MAX_TEXT_BYTES:
                            raise ValueError("Hybrid UTF-8 corpus byte budget exceeded")
                        yield path_terms(path), content, path, language, start, end

                target.executemany(
                    "INSERT INTO chunks(filename,body,path,language,start_line,end_line) VALUES(?,?,?,?,?,?)",
                    rows(),
                )
                target.executemany(
                    "INSERT INTO cache_metadata VALUES(?,?)", expected.items()
                )
            if temporary.stat().st_size > MAX_CACHE_BYTES:
                raise ValueError("Hybrid cache byte budget exceeded")
            if source_token(native) != token:
                raise ValueError(
                    "Native index changed during lexical cache construction"
                )
            os.replace(temporary, cache)
        finally:
            temporary.unlink(missing_ok=True)


def lexical_results(
    cache: Path, query: str, limit: int, path: str | None, langs: list[str]
) -> list[dict[str, object]]:
    tokens = query_tokens(query)
    if not tokens:
        return []
    conditions = ["chunks MATCH ?"]
    params: list[object] = [" OR ".join('"' + token + '"' for token in tokens)]
    if path:
        conditions.append("path GLOB ?")
        params.append(path)
    if langs:
        conditions.append("language IN (" + ",".join("?" for _ in langs) + ")")
        params.extend(langs)
    params.append(limit)
    sql = (
        "SELECT path,language,body,start_line,end_line FROM chunks WHERE "
        + " AND ".join(conditions)
        + " ORDER BY bm25(chunks,4.0,1.0),path,start_line LIMIT ?"
    )
    with closing(
        sqlite3.connect(cache.resolve().as_uri() + "?mode=ro", uri=True)
    ) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        dict(zip(("file_path", "language", "content", "start_line", "end_line"), row))
        for row in rows
    ]


def fuse(
    semantic: list[dict[str, object]], lexical: list[dict[str, object]], limit: int
) -> str:
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    representatives: dict[str, dict[str, object]] = {}
    for route, rows in (("semantic", semantic), ("lexical", lexical)):
        unique = {}
        for row in rows:
            unique.setdefault(row["file_path"], row)
        for rank, (path, row) in enumerate(unique.items(), 1):
            scores[path] += 1 / (60 + rank)
            ranks[path][route] = rank
            if path not in representatives or rank <= min(ranks[path].values()):
                representatives[path] = row
    ordered = sorted(scores, key=lambda p: (-scores[p], min(ranks[p].values()), p))[
        :limit
    ]
    output = [
        "Retrieval: native CocoIndex semantics + same-index lexical RRF; scores are ranks, not confidence.\n"
    ]
    for position, path in enumerate(ordered, 1):
        row = representatives[path]
        output.append(
            f"\n--- Result {position} (RRF: {scores[path]:.6f}; routes: {json.dumps(ranks[path], sort_keys=True)}) ---\nFile: {path}:{row['start_line']}-{row['end_line']} [{row['language']}]\n{row['content']}\n"
        )
    text = "".join(output)
    if len(text.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise ValueError(
            "Hybrid output exceeds its byte limit; narrow the query or scope"
        )
    return text


def search(
    native: Path,
    cache: Path,
    owner: str,
    source_digest: str,
    query: str,
    semantic_json: str,
    limit: int,
    mirror: Path,
    path: str | None = None,
    langs: list[str] | None = None,
) -> str:
    if type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError("Hybrid result limit must be 1..10")
    if (
        not query
        or len(query) > 4096
        or len(semantic_json.encode("utf-8")) > MAX_INPUT_BYTES
    ):
        raise ValueError("Hybrid input exceeds its bounded query/response contract")
    if cache.is_symlink() or cache.resolve() == native.resolve():
        raise ValueError("Hybrid cache must not alias the native index")
    semantic = native_results(semantic_json, limit)
    if not query_tokens(query):
        return fuse(semantic, [], limit)
    token = source_token(native)
    require_cache(native, cache, owner, source_digest, mirror)
    with cache_lock(cache.with_suffix(".lock")):
        require_cache(native, cache, owner, source_digest, mirror)
        lexical = lexical_results(cache, query, limit, path, langs or [])
        if source_token(native) != token:
            raise ValueError("Native index changed during hybrid retrieval")
        return fuse(semantic, lexical, limit)


def build_cache(
    native: Path, cache: Path, owner: str, source_digest: str, mirror: Path
) -> None:
    if cache.is_symlink() or cache.resolve() == native.resolve():
        raise ValueError("Hybrid cache must not alias the native index")
    if cache.exists():
        metadata = cache_metadata(cache)
        if metadata.get("owner") != owner or metadata.get("schema") != SCHEMA:
            raise ValueError("Hybrid cache belongs to another owner or schema")
    with cache_lock(cache.with_suffix(".lock")):
        ensure_cache(native, cache, owner, source_digest, mirror)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-db", "cache", "owner", "source-digest", "mirror"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--query")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--path")
    parser.add_argument("--lang", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        if args.build:
            build_cache(
                Path(args.native_db),
                Path(args.cache),
                args.owner,
                args.source_digest,
                Path(args.mirror),
            )
            print("Built owned same-source lexical cache.")
            return 0
        if not args.query:
            raise ValueError("A query is required outside explicit build mode")
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError("Native response exceeds input byte limit")
        text = search(
            Path(args.native_db),
            Path(args.cache),
            args.owner,
            args.source_digest,
            args.query,
            raw.decode("utf-8"),
            args.limit,
            Path(args.mirror),
            args.path,
            args.lang,
        )
    except (ValueError, OSError, sqlite3.Error, ImportError) as exc:
        detail = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print(f"Hybrid source search failed: {detail}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
