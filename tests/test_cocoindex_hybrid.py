from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from tools import cocoindex_agent_search as adapter
from tools import cocoindex_hybrid as hybrid


def fixture_index(tmp_path):
    native = tmp_path / "native.sqlite3"
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    rows = [
        (
            "src/orbit_math.py",
            "python",
            "def orbit_motion():\n    calculate elliptical orbital positions\n",
            1,
            2,
        ),
        (
            "src/vector.py",
            "python",
            "def vector():\n    combine axes coordinates\n",
            1,
            2,
        ),
        (
            "docs/orbits.md",
            "markdown",
            "An explanation of orbital motion and elliptical positions.",
            1,
            1,
        ),
    ]
    with closing(sqlite3.connect(native)) as conn, conn:
        conn.execute(
            "CREATE TABLE code_chunks_vec(file_path TEXT,language TEXT,content TEXT,start_line INT,end_line INT)"
        )
        conn.executemany("INSERT INTO code_chunks_vec VALUES(?,?,?,?,?)", rows)
    for path, lang, body, start, end in rows:
        p = mirror / path
        p.parent.mkdir(exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return native, mirror, tmp_path / "hybrid.sqlite3"


def native_json(rows=()):
    return json.dumps({"success": True, "results": list(rows)})


def row(path, content="source", start=1):
    return {
        "file_path": path,
        "language": "python",
        "content": content,
        "start_line": start,
        "end_line": start,
        "score": 0.5,
    }


def search(tmp_path, **changes):
    native, mirror, cache = fixture_index(tmp_path)
    args = dict(
        native=native,
        cache=cache,
        owner="owner",
        source_digest="source-a",
        query="elliptical orbit motion",
        semantic_json=native_json(),
        limit=5,
        mirror=mirror,
    )
    args.update(changes)
    hybrid.build_cache(native, cache, args["owner"], args["source_digest"], mirror)
    return hybrid.search(**args), args


def test_same_index_lexical_retrieval_preserves_native_database_and_caches_once(
    tmp_path,
):
    native, mirror, cache = fixture_index(tmp_path)
    before = hashlib.sha256(native.read_bytes()).hexdigest()
    args = dict(
        native=native,
        cache=cache,
        owner="owner",
        source_digest="source-a",
        query="elliptical orbit motion",
        semantic_json=native_json(),
        limit=5,
        mirror=mirror,
    )
    hybrid.build_cache(native, cache, "owner", "source-a", mirror)
    output = hybrid.search(**args)
    assert "File: src/orbit_math.py:" in output
    assert "lexical" in output and "not confidence" in output
    assert hashlib.sha256(native.read_bytes()).hexdigest() == before
    cached = cache.stat().st_mtime_ns
    assert hybrid.search(**args) == output
    assert cache.stat().st_mtime_ns == cached


def test_path_and_language_constraints_apply_to_lexical_route(tmp_path):
    output, _ = search(tmp_path, path="src/**", langs=["python"])
    assert "File: src/orbit_math.py:" in output
    assert "File: docs/orbits.md:" not in output


def test_native_failure_never_creates_a_lexical_cache(tmp_path):
    native, mirror, cache = fixture_index(tmp_path)
    with pytest.raises(ValueError, match="lexical fallback is forbidden"):
        hybrid.search(
            native,
            cache,
            "owner",
            "source",
            "orbit",
            json.dumps({"success": False, "results": []}),
            5,
            mirror,
        )
    assert not cache.exists() and not cache.with_suffix(".lock").exists()


def test_queries_do_not_implicitly_build_a_missing_cache(tmp_path):
    native, mirror, cache = fixture_index(tmp_path)
    with pytest.raises(ValueError, match="explicit index"):
        hybrid.search(
            native, cache, "owner", "source", "orbit", native_json(), 5, mirror
        )
    assert not cache.exists() and not cache.with_suffix(".lock").exists()


def test_foreign_or_unrecognized_cache_is_not_overwritten(tmp_path):
    _, args = search(tmp_path)
    cache = args["cache"]
    before = cache.read_bytes()
    args["owner"] = "different-owner"
    with pytest.raises(ValueError, match="another owner"):
        hybrid.search(**args)
    assert cache.read_bytes() == before


def test_source_fingerprint_change_rebuilds_owned_cache(tmp_path):
    _, args = search(tmp_path)
    args["source_digest"] = "source-b"
    with pytest.raises(ValueError, match="stale"):
        hybrid.search(**args)
    hybrid.build_cache(
        args["native"],
        args["cache"],
        args["owner"],
        args["source_digest"],
        args["mirror"],
    )
    hybrid.search(**args)
    assert hybrid.cache_metadata(args["cache"])["source_digest"] == "source-b"


def test_cache_cannot_alias_native_index(tmp_path):
    native, mirror, _ = fixture_index(tmp_path)
    before = native.read_bytes()
    with pytest.raises(ValueError, match="alias"):
        hybrid.search(
            native, native, "owner", "source", "orbit", native_json(), 5, mirror
        )
    assert native.read_bytes() == before


def test_unapproved_index_path_stops_before_materializing_fts(tmp_path):
    native, mirror, cache = fixture_index(tmp_path)
    with closing(sqlite3.connect(native)) as conn, conn:
        conn.execute(
            "INSERT INTO code_chunks_vec VALUES(?,?,?,?,?)",
            ("data/protected/partial.json", "json", "must not index", 1, 1),
        )
    with pytest.raises(ValueError, match="approved source mirror"):
        hybrid.build_cache(native, cache, "owner", "source", mirror)
    assert not cache.exists()


def test_oversized_rebuild_preserves_last_complete_owned_cache(tmp_path, monkeypatch):
    _, args = search(tmp_path)
    cache = args["cache"]
    before = cache.read_bytes()
    monkeypatch.setattr(hybrid, "MAX_ROWS", 0)
    args["source_digest"] = "new-source"
    with pytest.raises(ValueError, match="build budget"):
        hybrid.build_cache(
            args["native"], cache, args["owner"], args["source_digest"], args["mirror"]
        )
    assert cache.read_bytes() == before
    assert not list(tmp_path.glob("hybrid-*.sqlite3"))


def test_literal_query_tokenization_cannot_execute_fts_syntax(tmp_path):
    output, args = search(tmp_path, query='"orbit" OR body:* -NEAR(axes)')
    assert "Retrieval:" in output
    assert hybrid.cache_metadata(args["cache"])["owner"] == "owner"


def test_file_level_fusion_does_not_reward_duplicate_chunks():
    semantic = [
        row("src/alpha.py", "first"),
        row("src/alpha.py", "second"),
        row("src/beta.py"),
    ]
    lexical = [row("src/beta.py"), row("src/gamma.py")]
    output = hybrid.fuse(semantic, lexical, 2)
    assert output.count("File: ") == 2
    assert output.index("File: src/beta.py") < output.index("File: src/alpha.py")


@pytest.mark.parametrize("bad", [True, 0, 11])
def test_invalid_limit_stops_before_source_access(tmp_path, bad):
    with pytest.raises(ValueError, match="1..10"):
        hybrid.search(
            tmp_path / "absent",
            tmp_path / "cache",
            "owner",
            "source",
            "orbit",
            native_json(),
            bad,
            tmp_path,
        )


def test_semantic_failure_receipt_precedes_and_prevents_hybrid_child(
    tmp_path, monkeypatch
):
    native = tmp_path / "native"
    native.touch()
    events = []
    context = SimpleNamespace(
        repo_root=tmp_path,
        mirror_repo=tmp_path,
        db_dir=tmp_path,
        mirror_digest="source",
        venv_python=Path(sys.executable),
    )
    monkeypatch.setattr(adapter, "target_sqlite_db", lambda _: native)
    monkeypatch.setattr(adapter, "require_current_index", lambda *_a, **_k: None)
    monkeypatch.setattr(
        adapter,
        "run_ccc_existing",
        lambda *_a, **_k: subprocess.CompletedProcess([], 1, '{"success":false}', ""),
    )

    def receipt(*args, **kwargs):
        events.append("native-receipt")
        return tmp_path / "native-receipt.json"

    monkeypatch.setattr(adapter, "retain_search_receipt", receipt)
    child = mock.Mock(side_effect=AssertionError("lexical fallback attempted"))
    monkeypatch.setattr(adapter, "run_captured_process", child)
    with pytest.raises(RuntimeError, match="no lexical fallback"):
        adapter.run_search(
            context,
            query=["orbit"],
            limit=5,
            path=None,
            langs=[],
            refresh=False,
            allow_index=False,
            hybrid=True,
        )
    assert events == ["native-receipt"]
    child.assert_not_called()


def test_parent_retains_native_and_final_receipts_in_order(tmp_path, monkeypatch):
    native = tmp_path / "native"
    native.touch()
    events = []
    provenance = []
    context = SimpleNamespace(
        repo_root=tmp_path,
        mirror_repo=tmp_path,
        db_dir=tmp_path,
        mirror_digest="source",
        venv_python=Path(sys.executable),
    )
    monkeypatch.setattr(adapter, "target_sqlite_db", lambda _: native)
    monkeypatch.setattr(
        adapter, "require_current_index", lambda *_a, **_k: events.append("verify")
    )

    def semantic(*args, **kwargs):
        events.append("native")
        assert "--json" in args[1] and kwargs["check"] is False
        return subprocess.CompletedProcess([], 0, native_json(), "")

    def retain(*args, **kwargs):
        events.append("receipt")
        provenance.append(kwargs.get("provenance"))
        return tmp_path / (
            "native-receipt.json" if len(provenance) == 1 else "final-receipt.json"
        )

    def lexical(*args, **kwargs):
        events.append("lexical")
        assert kwargs["input_text"] == native_json()
        return subprocess.CompletedProcess([], 0, "File: src/alpha.py:1 result\n", "")

    monkeypatch.setattr(adapter, "run_ccc_existing", semantic)
    monkeypatch.setattr(adapter, "retain_search_receipt", retain)
    monkeypatch.setattr(adapter, "run_hybrid_process", lexical)
    result = adapter.run_search(
        context,
        query=["orbit"],
        limit=5,
        path=None,
        langs=[],
        refresh=False,
        allow_index=False,
        hybrid=True,
    )
    assert result == "File: src/alpha.py:1 result\n"
    assert events == [
        "verify",
        "native",
        "receipt",
        "verify",
        "lexical",
        "receipt",
        "verify",
    ]
    assert provenance[-1]["native_semantic_receipt"] == "native-receipt.json"


def test_mcp_native_only_mode_remains_explicit_and_type_checked(monkeypatch):
    context = object()
    monkeypatch.setattr(adapter, "resolve_active_index_context", lambda: context)
    search = mock.Mock(return_value="native")
    monkeypatch.setattr(adapter, "run_search", search)
    assert (
        adapter.run_mcp_search_tool({"query": "orbit", "semantic_only": True})
        == "native"
    )
    assert search.call_args.kwargs["hybrid"] is False
    with pytest.raises(adapter.JsonRpcError, match="boolean"):
        adapter.mcp_search_arguments({"query": "orbit", "semantic_only": "true"})
