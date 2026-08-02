"""Durable pretest seal and single-use test access for Polymarket Round 21."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import time

from .polymarket_round21_ai_selection import Round21AICandidateSelection
from .polymarket_round21_comparison import (
    Round21MatchedEconomicComparison,
    round21_replay_matrix_sha256,
)
from .polymarket_round21_contract import POLYMARKET_ROUND21_CONTRACT_SHA256
from .polymarket_round21_model import validate_round21_development_artifact
from .polymarket_round21_replay import Round21EconomicReplay
from .polymarket_round21_sealed import (
    POLYMARKET_ROUND21_SEALED_DESIGN_SHA256,
    POLYMARKET_ROUND21_SEALED_RESULT_SCHEMA_VERSION,
    Round21SealedEvaluationResult,
    load_round21_sealed_design,
)


POLYMARKET_ROUND21_PRETEST_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round21-pretest-manifest-v1"
)
POLYMARKET_ROUND21_ONE_USE_CLAIM_SCHEMA_VERSION = (
    "polymarket-round21-one-use-claim-v1"
)
POLYMARKET_ROUND21_ONE_USE_STORE_SCHEMA_VERSION = (
    "polymarket-round21-one-use-store-v1"
)
POLYMARKET_ROUND21_TEST_ACCESS_SCHEMA_VERSION = (
    "polymarket-round21-test-access-v1"
)
_LAYERS = ("core", "core_spot", "core_spot_usdm")
_STATUSES = frozenset({"claim_open", "test_access_consumed", "completed", "failed"})
_REQUIRED_FILES = (
    "docs/model-research/polymarket/"
    "round-021-core-corpus-materialization-design-v1.json",
    "docs/model-research/polymarket/"
    "round-021-independent-matched-edge-contract-v1.json",
    "docs/model-research/polymarket/"
    "round-021-terminal-sealed-evaluation-design-v1.json",
    "docs/model-research/polymarket/"
    "round-021-terminal-transport-manifest-design-v1.json",
    "src/simple_ai_trading/polymarket_round21_ai.py",
    "src/simple_ai_trading/polymarket_round21_ai_comparison.py",
    "src/simple_ai_trading/polymarket_round21_ai_selection.py",
    "src/simple_ai_trading/polymarket_round21_comparison.py",
    "src/simple_ai_trading/polymarket_round21_core_features.py",
    "src/simple_ai_trading/polymarket_round21_corpus.py",
    "src/simple_ai_trading/polymarket_round21_dataset.py",
    "src/simple_ai_trading/polymarket_round21_execution.py",
    "src/simple_ai_trading/polymarket_round21_model.py",
    "src/simple_ai_trading/polymarket_round21_one_use.py",
    "src/simple_ai_trading/polymarket_round21_policy.py",
    "src/simple_ai_trading/polymarket_round21_replay.py",
    "src/simple_ai_trading/polymarket_round21_sealed.py",
    "src/simple_ai_trading/polymarket_round21_terminal.py",
    "tests/test_polymarket_round21_core_features.py",
    "tests/test_polymarket_round21_corpus.py",
    "tests/test_polymarket_round21_one_use.py",
    "tests/test_polymarket_round21_sealed.py",
    "tests/test_polymarket_round21_terminal.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID = re.compile(r"^[0-9a-f]{40,64}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_MAXIMUM_JSON_BYTES = 4 * 1024 * 1024


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


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 21 one-use JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 21 one-use JSON contains {value}")


def _strict_json(value: str, *, label: str) -> dict[str, object]:
    if not 2 <= len(value.encode("utf-8")) <= _MAXIMUM_JSON_BYTES:
        raise ValueError(f"{label} size differs")
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{label} is not an object")
    return dict(parsed)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if completed.returncode:
        raise ValueError(
            "Round 21 pretest Git operation failed: "
            + (completed.stderr.strip() or completed.stdout.strip())[:500]
        )
    return completed.stdout.strip().lower()


def _repository_attestation(
    repository: Path,
) -> tuple[str, str, dict[str, str]]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Round 21 pretest seal requires a clean worktree")
    commit = _git(repository, "rev-parse", "HEAD")
    tree = _git(repository, "rev-parse", "HEAD^{tree}")
    files: dict[str, str] = {}
    for relative in _REQUIRED_FILES:
        path = repository / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Round 21 pretest file is unavailable: {relative}")
        files[relative] = _file_sha256(path)
    return commit, tree, files


def _digest(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if _SHA256.fullmatch(selected) is None or selected == _EMPTY_SHA256:
        raise ValueError(f"Round 21 {name} digest differs")
    return selected


@dataclass(frozen=True, slots=True)
class Round21PretestManifest:
    created_at_ms: int
    selected_population_layer: str
    core_campaign_terminal_sha256: str
    optional_campaign_terminal_sha256: str | None
    sealed_test_population_manifest_sha256: str
    development_model_artifact_sha256: str
    development_economic_matrix_sha256: str
    development_optional_comparison_sha256: str | None
    development_ai_selection_sha256: str
    nominated_ai_model: str | None
    nominated_ai_model_digest: str | None
    nominated_ai_comparison_sha256: str | None
    repository_commit_oid: str
    repository_tree_oid: str
    repository_file_sha256: Mapping[str, str]
    manifest_sha256: str
    test_features_accessed: bool = False
    test_targets_accessed: bool = False
    test_execution_accessed: bool = False
    automatic_promotion: bool = False
    profitability_claim: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_PRETEST_MANIFEST_SCHEMA_VERSION,
            "sealed_design_sha256": POLYMARKET_ROUND21_SEALED_DESIGN_SHA256,
            "created_at_ms": self.created_at_ms,
            "selected_population_layer": self.selected_population_layer,
            "core_campaign_terminal_sha256": self.core_campaign_terminal_sha256,
            "optional_campaign_terminal_sha256": (
                self.optional_campaign_terminal_sha256
            ),
            "sealed_test_population_manifest_sha256": (
                self.sealed_test_population_manifest_sha256
            ),
            "development_model_artifact_sha256": (
                self.development_model_artifact_sha256
            ),
            "development_economic_matrix_sha256": (
                self.development_economic_matrix_sha256
            ),
            "development_optional_comparison_sha256": (
                self.development_optional_comparison_sha256
            ),
            "development_ai_selection_sha256": (
                self.development_ai_selection_sha256
            ),
            "nominated_ai_model": self.nominated_ai_model,
            "nominated_ai_model_digest": self.nominated_ai_model_digest,
            "nominated_ai_comparison_sha256": (
                self.nominated_ai_comparison_sha256
            ),
            "repository_commit_oid": self.repository_commit_oid,
            "repository_tree_oid": self.repository_tree_oid,
            "repository_file_sha256": dict(sorted(self.repository_file_sha256.items())),
            "test_features_accessed": False,
            "test_targets_accessed": False,
            "test_execution_accessed": False,
            "automatic_promotion": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "manifest_sha256": self.manifest_sha256}

    def validated(self) -> Round21PretestManifest:
        optional_required = self.selected_population_layer != "core"
        ai_triplet = (
            self.nominated_ai_model,
            self.nominated_ai_model_digest,
            self.nominated_ai_comparison_sha256,
        )
        if (
            self.created_at_ms <= 0
            or self.selected_population_layer not in _LAYERS
            or any(
                _SHA256.fullmatch(value) is None or value == _EMPTY_SHA256
                for value in (
                    self.core_campaign_terminal_sha256,
                    self.sealed_test_population_manifest_sha256,
                    self.development_model_artifact_sha256,
                    self.development_economic_matrix_sha256,
                    self.development_ai_selection_sha256,
                    self.manifest_sha256,
                )
            )
            or optional_required
            != (self.optional_campaign_terminal_sha256 is not None)
            or optional_required
            != (self.development_optional_comparison_sha256 is not None)
            or any(
                value is not None
                and (_SHA256.fullmatch(value) is None or value == _EMPTY_SHA256)
                for value in (
                    self.optional_campaign_terminal_sha256,
                    self.development_optional_comparison_sha256,
                )
            )
            or any(value is None for value in ai_triplet)
            != all(value is None for value in ai_triplet)
            or (ai_triplet[0] is not None and not str(ai_triplet[0]).strip())
            or (
                any(
                    _SHA256.fullmatch(str(value)) is None or value == _EMPTY_SHA256
                    for value in ai_triplet[1:]
                    if value is not None
                )
            )
            or _GIT_OID.fullmatch(self.repository_commit_oid) is None
            or _GIT_OID.fullmatch(self.repository_tree_oid) is None
            or set(self.repository_file_sha256) != set(_REQUIRED_FILES)
            or any(
                _SHA256.fullmatch(value) is None or value == _EMPTY_SHA256
                for value in self.repository_file_sha256.values()
            )
            or self.test_features_accessed
            or self.test_targets_accessed
            or self.test_execution_accessed
            or self.automatic_promotion
            or self.profitability_claim
            or self.paper_trading_authority
            or self.live_trading_authority
            or any(
                type(value) is not bool
                for value in (
                    self.test_features_accessed,
                    self.test_targets_accessed,
                    self.test_execution_accessed,
                    self.automatic_promotion,
                    self.profitability_claim,
                    self.paper_trading_authority,
                    self.live_trading_authority,
                )
            )
            or self.manifest_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 pretest manifest differs")
        return self


def build_round21_pretest_manifest(
    repository: str | Path,
    *,
    selected_population_layer: str,
    core_campaign_terminal_sha256: str,
    optional_campaign_terminal_sha256: str | None,
    sealed_test_population_manifest_sha256: str,
    development_model_artifact: Mapping[str, object],
    development_economic_matrix: Sequence[Round21EconomicReplay],
    development_optional_comparison: Round21MatchedEconomicComparison | None,
    development_ai_selection: Round21AICandidateSelection,
    created_at_ms: int | None = None,
) -> Round21PretestManifest:
    """Seal every development choice before any test-role evidence is opened."""

    root = Path(repository).resolve()
    load_round21_sealed_design(root)
    artifact = validate_round21_development_artifact(development_model_artifact)
    matrix = tuple(value.validated() for value in development_economic_matrix)
    if (
        len(matrix) != 81
        or len({(value.profile, value.scenario) for value in matrix}) != 81
        or not all(value.economic_gate_passed for value in matrix)
    ):
        raise RuntimeError("Round 21 development economic matrix did not qualify")
    layer = str(selected_population_layer or "").strip()
    layers = artifact.get("layers")
    if not isinstance(layers, Mapping) or layer not in _LAYERS:
        raise ValueError("Round 21 pretest selected layer differs")
    core = layers.get("core")
    selected = layers.get(layer)
    if (
        not isinstance(core, Mapping)
        or not isinstance(core.get("comparison"), Mapping)
        or core["comparison"].get("predictive_development_accepted") is not True
        or not isinstance(selected, Mapping)
        or not isinstance(selected.get("comparison"), Mapping)
        or selected["comparison"].get("predictive_development_accepted") is not True
    ):
        raise RuntimeError("Round 21 development predictive layer did not qualify")
    optional = (
        None
        if development_optional_comparison is None
        else development_optional_comparison.validated()
    )
    if (layer == "core") != (optional is None):
        raise ValueError("Round 21 pretest optional comparison differs")
    if optional is not None and (
        optional.challenger_layer != layer or not optional.all_replays_accepted
    ):
        raise RuntimeError("Round 21 development optional layer did not qualify")
    if (layer == "core") != (optional_campaign_terminal_sha256 is None):
        raise ValueError("Round 21 pretest optional campaign binding differs")
    ai = development_ai_selection.validated()
    commit, tree, files = _repository_attestation(root)
    now = time.time_ns() // 1_000_000 if created_at_ms is None else int(created_at_ms)
    provisional = Round21PretestManifest(
        created_at_ms=now,
        selected_population_layer=layer,
        core_campaign_terminal_sha256=_digest(
            core_campaign_terminal_sha256,
            name="core campaign terminal",
        ),
        optional_campaign_terminal_sha256=(
            None
            if optional_campaign_terminal_sha256 is None
            else _digest(
                optional_campaign_terminal_sha256,
                name="optional campaign terminal",
            )
        ),
        sealed_test_population_manifest_sha256=_digest(
            sealed_test_population_manifest_sha256,
            name="sealed test population manifest",
        ),
        development_model_artifact_sha256=str(artifact["artifact_sha256"]),
        development_economic_matrix_sha256=round21_replay_matrix_sha256(matrix),
        development_optional_comparison_sha256=(
            None if optional is None else optional.comparison_sha256
        ),
        development_ai_selection_sha256=ai.selection_sha256,
        nominated_ai_model=ai.nominated_model,
        nominated_ai_model_digest=ai.nominated_model_digest,
        nominated_ai_comparison_sha256=ai.nominated_comparison_sha256,
        repository_commit_oid=commit,
        repository_tree_oid=tree,
        repository_file_sha256=files,
        manifest_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional,
        manifest_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


@dataclass(frozen=True, slots=True)
class Round21OneUseClaim:
    pretest_manifest_sha256: str
    selected_population_layer: str
    sealed_test_population_manifest_sha256: str
    repository_commit_oid: str
    nominated_ai_model: str | None
    nominated_ai_model_digest: str | None
    opened_at_ms: int
    claim_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_ONE_USE_CLAIM_SCHEMA_VERSION,
            "sealed_design_sha256": POLYMARKET_ROUND21_SEALED_DESIGN_SHA256,
            "pretest_manifest_sha256": self.pretest_manifest_sha256,
            "selected_population_layer": self.selected_population_layer,
            "sealed_test_population_manifest_sha256": (
                self.sealed_test_population_manifest_sha256
            ),
            "repository_commit_oid": self.repository_commit_oid,
            "nominated_ai_model": self.nominated_ai_model,
            "nominated_ai_model_digest": self.nominated_ai_model_digest,
            "opened_at_ms": self.opened_at_ms,
            "test_features_accessed": False,
            "test_targets_accessed": False,
            "test_execution_accessed": False,
            "return_to_development": False,
            "automatic_promotion": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "claim_sha256": self.claim_sha256}

    def validated(self) -> Round21OneUseClaim:
        ai_pair = (self.nominated_ai_model, self.nominated_ai_model_digest)
        if (
            any(
                _SHA256.fullmatch(value) is None or value == _EMPTY_SHA256
                for value in (
                    self.pretest_manifest_sha256,
                    self.sealed_test_population_manifest_sha256,
                    self.claim_sha256,
                )
            )
            or self.selected_population_layer not in _LAYERS
            or _GIT_OID.fullmatch(self.repository_commit_oid) is None
            or (ai_pair[0] is None) != (ai_pair[1] is None)
            or (ai_pair[0] is not None and not str(ai_pair[0]).strip())
            or (
                ai_pair[1] is not None
                and (_SHA256.fullmatch(ai_pair[1]) is None or ai_pair[1] == _EMPTY_SHA256)
            )
            or self.opened_at_ms <= 0
            or self.claim_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 one-use claim differs")
        return self


def create_round21_one_use_claim(
    pretest: Round21PretestManifest,
    *,
    opened_at_ms: int | None = None,
) -> Round21OneUseClaim:
    selected = pretest.validated()
    now = time.time_ns() // 1_000_000 if opened_at_ms is None else int(opened_at_ms)
    if now < selected.created_at_ms:
        raise ValueError("Round 21 one-use claim predates the pretest seal")
    provisional = Round21OneUseClaim(
        pretest_manifest_sha256=selected.manifest_sha256,
        selected_population_layer=selected.selected_population_layer,
        sealed_test_population_manifest_sha256=(
            selected.sealed_test_population_manifest_sha256
        ),
        repository_commit_oid=selected.repository_commit_oid,
        nominated_ai_model=selected.nominated_ai_model,
        nominated_ai_model_digest=selected.nominated_ai_model_digest,
        opened_at_ms=now,
        claim_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional,
        claim_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def _claim_from_mapping(value: Mapping[str, object]) -> Round21OneUseClaim:
    expected = {
        "schema_version",
        "sealed_design_sha256",
        "pretest_manifest_sha256",
        "selected_population_layer",
        "sealed_test_population_manifest_sha256",
        "repository_commit_oid",
        "nominated_ai_model",
        "nominated_ai_model_digest",
        "opened_at_ms",
        "test_features_accessed",
        "test_targets_accessed",
        "test_execution_accessed",
        "return_to_development",
        "automatic_promotion",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
        "claim_sha256",
    }
    false_fields = (
        "test_features_accessed",
        "test_targets_accessed",
        "test_execution_accessed",
        "return_to_development",
        "automatic_promotion",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    )
    if (
        set(value) != expected
        or value.get("schema_version")
        != POLYMARKET_ROUND21_ONE_USE_CLAIM_SCHEMA_VERSION
        or value.get("sealed_design_sha256")
        != POLYMARKET_ROUND21_SEALED_DESIGN_SHA256
        or any(value.get(name) is not False for name in false_fields)
    ):
        raise ValueError("Round 21 one-use claim schema differs")
    try:
        return Round21OneUseClaim(
            pretest_manifest_sha256=str(value["pretest_manifest_sha256"]),
            selected_population_layer=str(value["selected_population_layer"]),
            sealed_test_population_manifest_sha256=str(
                value["sealed_test_population_manifest_sha256"]
            ),
            repository_commit_oid=str(value["repository_commit_oid"]),
            nominated_ai_model=(
                None
                if value["nominated_ai_model"] is None
                else str(value["nominated_ai_model"])
            ),
            nominated_ai_model_digest=(
                None
                if value["nominated_ai_model_digest"] is None
                else str(value["nominated_ai_model_digest"])
            ),
            opened_at_ms=int(value["opened_at_ms"]),
            claim_sha256=str(value["claim_sha256"]),
        ).validated()
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Round 21 one-use claim schema differs") from exc


def _test_access_payload(
    claim: Round21OneUseClaim,
    access_started_ms: int,
) -> dict[str, object]:
    return {
        "schema_version": POLYMARKET_ROUND21_TEST_ACCESS_SCHEMA_VERSION,
        "claim_sha256": claim.claim_sha256,
        "pretest_manifest_sha256": claim.pretest_manifest_sha256,
        "sealed_test_population_manifest_sha256": (
            claim.sealed_test_population_manifest_sha256
        ),
        "access_started_ms": access_started_ms,
        "test_features_accessed": True,
        "test_targets_accessed": True,
        "test_execution_accessed": True,
        "return_to_development": False,
        "automatic_promotion": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }


class Round21OneUseStore:
    """Singleton claim with synchronous durability and a hash-chained journal."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.is_symlink() or (self.path.exists() and not self.path.is_file()):
            raise ValueError("Round 21 one-use store path differs")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode = DELETE")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.execute("PRAGMA temp_store = MEMORY")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self._initialize()

    def __enter__(self) -> Round21OneUseStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS round21_one_use_metadata (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS round21_one_use_claim (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                claim_sha256 TEXT NOT NULL UNIQUE CHECK (length(claim_sha256) = 64),
                claim_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('claim_open', 'test_access_consumed', 'completed', 'failed')
                ),
                test_access_sha256 TEXT UNIQUE,
                access_started_ms INTEGER,
                result_sha256 TEXT UNIQUE,
                result_json TEXT,
                failure_json TEXT
            );
            CREATE TABLE IF NOT EXISTS round21_one_use_event (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_sha256 TEXT NOT NULL UNIQUE CHECK (length(event_sha256) = 64),
                previous_event_sha256 TEXT NOT NULL,
                event_json TEXT NOT NULL
            );
            """
        )
        self.connection.execute(
            """
            INSERT OR IGNORE INTO round21_one_use_metadata(singleton, schema_version)
            VALUES (1, ?)
            """,
            (POLYMARKET_ROUND21_ONE_USE_STORE_SCHEMA_VERSION,),
        )
        row = self.connection.execute(
            "SELECT schema_version FROM round21_one_use_metadata WHERE singleton = 1"
        ).fetchone()
        if (
            row is None
            or row["schema_version"] != POLYMARKET_ROUND21_ONE_USE_STORE_SCHEMA_VERSION
        ):
            raise ValueError("Round 21 one-use store schema differs")

    def _append_event(
        self,
        *,
        event_type: str,
        claim_sha256: str,
        observed_at_ms: int,
        details: Mapping[str, object],
    ) -> None:
        prior = self.connection.execute(
            "SELECT event_sha256 FROM round21_one_use_event ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous = "" if prior is None else str(prior["event_sha256"])
        payload = {
            "schema_version": "polymarket-round21-one-use-event-v1",
            "event_type": event_type,
            "claim_sha256": claim_sha256,
            "observed_at_ms": int(observed_at_ms),
            "previous_event_sha256": previous,
            "details": dict(details),
        }
        digest = _canonical_sha256(payload)
        self.connection.execute(
            """
            INSERT INTO round21_one_use_event(
                event_sha256, previous_event_sha256, event_json
            ) VALUES (?, ?, ?)
            """,
            (digest, previous, _canonical_json(payload)),
        )

    def open_claim(self, claim: Round21OneUseClaim) -> Round21OneUseClaim:
        selected = claim.validated()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT claim_json FROM round21_one_use_claim WHERE singleton = 1"
            ).fetchone()
            if row is not None:
                existing = _claim_from_mapping(
                    _strict_json(str(row["claim_json"]), label="stored Round 21 claim")
                )
                if existing.claim_sha256 != selected.claim_sha256:
                    raise RuntimeError("Round 21 one-use store already has a claim")
                output = existing
            else:
                self.connection.execute(
                    """
                    INSERT INTO round21_one_use_claim(
                        singleton, claim_sha256, claim_json, status
                    ) VALUES (1, ?, ?, 'claim_open')
                    """,
                    (selected.claim_sha256, _canonical_json(selected.asdict())),
                )
                self._append_event(
                    event_type="claim_opened",
                    claim_sha256=selected.claim_sha256,
                    observed_at_ms=selected.opened_at_ms,
                    details={
                        "pretest_manifest_sha256": selected.pretest_manifest_sha256
                    },
                )
                output = selected
            self.connection.execute("COMMIT")
            return output
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def _row(self, claim: Round21OneUseClaim) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM round21_one_use_claim WHERE singleton = 1 AND claim_sha256 = ?",
            (claim.claim_sha256,),
        ).fetchone()
        if row is None:
            raise ValueError("Round 21 one-use claim is unavailable")
        return row

    def consume_test_access(
        self,
        claim: Round21OneUseClaim,
        *,
        observed_at_ms: int | None = None,
    ) -> str:
        selected = claim.validated()
        now = time.time_ns() // 1_000_000 if observed_at_ms is None else int(observed_at_ms)
        if now < selected.opened_at_ms:
            raise ValueError("Round 21 test access time differs")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._row(selected)
            if row["status"] != "claim_open" or row["test_access_sha256"] is not None:
                raise RuntimeError("Round 21 test access is already consumed or terminal")
            payload = _test_access_payload(selected, now)
            access_sha256 = _canonical_sha256(payload)
            self.connection.execute(
                """
                UPDATE round21_one_use_claim
                SET status = 'test_access_consumed', test_access_sha256 = ?,
                    access_started_ms = ? WHERE singleton = 1
                """,
                (access_sha256, now),
            )
            self._append_event(
                event_type="test_access_consumed",
                claim_sha256=selected.claim_sha256,
                observed_at_ms=now,
                details={"test_access_sha256": access_sha256},
            )
            self.connection.execute("COMMIT")
            return access_sha256
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def complete(
        self,
        claim: Round21OneUseClaim,
        result: Round21SealedEvaluationResult,
        *,
        observed_at_ms: int | None = None,
    ) -> Round21SealedEvaluationResult:
        selected = claim.validated()
        sealed = result.validated()
        now = time.time_ns() // 1_000_000 if observed_at_ms is None else int(observed_at_ms)
        if (
            now <= 0
            or sealed.claim_sha256 != selected.claim_sha256
            or sealed.selected_population_layer != selected.selected_population_layer
            or sealed.sealed_test_population_manifest_sha256
            != selected.sealed_test_population_manifest_sha256
            or (sealed.ai_model, sealed.ai_model_digest)
            != (selected.nominated_ai_model, selected.nominated_ai_model_digest)
        ):
            raise ValueError("Round 21 sealed result claim differs")
        raw = _canonical_json(sealed.asdict())
        if len(raw.encode("ascii")) > _MAXIMUM_JSON_BYTES:
            raise ValueError("Round 21 sealed result is too large")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._row(selected)
            if row["status"] != "test_access_consumed":
                raise RuntimeError("Round 21 one-use claim cannot complete")
            if now < int(row["access_started_ms"]):
                raise ValueError("Round 21 sealed result time differs")
            if sealed.test_access_sha256 != row["test_access_sha256"]:
                raise ValueError("Round 21 sealed result access differs")
            self.connection.execute(
                """
                UPDATE round21_one_use_claim
                SET status = 'completed', result_sha256 = ?, result_json = ?
                WHERE singleton = 1
                """,
                (sealed.result_sha256, raw),
            )
            self._append_event(
                event_type="evaluation_completed",
                claim_sha256=selected.claim_sha256,
                observed_at_ms=now,
                details={"result_sha256": sealed.result_sha256},
            )
            self.connection.execute("COMMIT")
            return sealed
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def fail(
        self,
        claim: Round21OneUseClaim,
        *,
        reason: str,
        observed_at_ms: int | None = None,
    ) -> None:
        selected = claim.validated()
        selected_reason = str(reason or "").strip()
        now = time.time_ns() // 1_000_000 if observed_at_ms is None else int(observed_at_ms)
        if not selected_reason or len(selected_reason) > 500 or now < selected.opened_at_ms:
            raise ValueError("Round 21 one-use failure differs")
        failure = {
            "schema_version": "polymarket-round21-one-use-failure-v1",
            "claim_sha256": selected.claim_sha256,
            "failed_at_ms": now,
            "reason": selected_reason,
            "return_to_development": False,
        }
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._row(selected)
            if row["status"] == "completed":
                raise RuntimeError("Round 21 one-use claim already completed")
            if row["access_started_ms"] is not None and now < int(
                row["access_started_ms"]
            ):
                raise ValueError("Round 21 one-use failure time differs")
            if row["status"] != "failed":
                self.connection.execute(
                    """
                    UPDATE round21_one_use_claim
                    SET status = 'failed', failure_json = ? WHERE singleton = 1
                    """,
                    (_canonical_json(failure),),
                )
                self._append_event(
                    event_type="evaluation_failed",
                    claim_sha256=selected.claim_sha256,
                    observed_at_ms=now,
                    details={"failure_sha256": _canonical_sha256(failure)},
                )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def snapshot(self) -> dict[str, object]:
        row = self.connection.execute(
            "SELECT * FROM round21_one_use_claim WHERE singleton = 1"
        ).fetchone()
        if row is None:
            event_count = int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM round21_one_use_event"
                ).fetchone()[0]
            )
            if event_count:
                raise ValueError("Round 21 one-use events exist without a claim")
            return {
                "schema_version": POLYMARKET_ROUND21_ONE_USE_STORE_SCHEMA_VERSION,
                "status": "empty",
                "claim": None,
                "test_access_consumed": False,
                "event_count": event_count,
            }
        claim = _claim_from_mapping(
            _strict_json(str(row["claim_json"]), label="stored Round 21 claim")
        )
        status = str(row["status"])
        expected_access_sha256 = (
            None
            if row["access_started_ms"] is None
            else _canonical_sha256(
                _test_access_payload(claim, int(row["access_started_ms"]))
            )
        )
        if (
            status not in _STATUSES
            or row["claim_sha256"] != claim.claim_sha256
            or row["test_access_sha256"] != expected_access_sha256
            or (
                status == "claim_open"
                and (
                    row["test_access_sha256"] is not None
                    or row["access_started_ms"] is not None
                )
            )
            or (
                status in {"test_access_consumed", "completed"}
                and (
                    row["test_access_sha256"] is None
                    or row["access_started_ms"] is None
                )
            )
            or (
                status == "failed"
                and (row["test_access_sha256"] is None)
                != (row["access_started_ms"] is None)
            )
            or (
                status == "completed"
                and (
                    row["result_sha256"] is None
                    or row["result_json"] is None
                    or row["failure_json"] is not None
                )
            )
            or (
                status == "failed"
                and (
                    row["failure_json"] is None
                    or row["result_sha256"] is not None
                    or row["result_json"] is not None
                )
            )
            or (
                status in {"claim_open", "test_access_consumed"}
                and any(
                    value is not None
                    for value in (
                        row["result_sha256"],
                        row["result_json"],
                        row["failure_json"],
                    )
                )
            )
        ):
            raise ValueError("Round 21 one-use status differs")
        events = self.connection.execute(
            """
            SELECT sequence, event_sha256, previous_event_sha256, event_json
            FROM round21_one_use_event ORDER BY sequence
            """
        ).fetchall()
        previous = ""
        previous_time = 0
        event_payloads: list[dict[str, object]] = []
        for event in events:
            payload = _strict_json(str(event["event_json"]), label="stored Round 21 event")
            if (
                set(payload)
                != {
                    "schema_version",
                    "event_type",
                    "claim_sha256",
                    "observed_at_ms",
                    "previous_event_sha256",
                    "details",
                }
                or payload.get("schema_version")
                != "polymarket-round21-one-use-event-v1"
                or not isinstance(payload.get("details"), Mapping)
                or type(payload.get("observed_at_ms")) is not int
                or int(payload["observed_at_ms"]) < previous_time
                or event["previous_event_sha256"] != previous
                or payload.get("previous_event_sha256") != previous
                or event["event_sha256"] != _canonical_sha256(payload)
                or payload.get("claim_sha256") != claim.claim_sha256
            ):
                raise ValueError("Round 21 one-use event chain differs")
            previous_time = int(payload["observed_at_ms"])
            previous = str(event["event_sha256"])
            event_payloads.append(payload)
        result = (
            None
            if row["result_json"] is None
            else _strict_json(str(row["result_json"]), label="stored Round 21 result")
        )
        failure = (
            None
            if row["failure_json"] is None
            else _strict_json(str(row["failure_json"]), label="stored Round 21 failure")
        )
        if result is not None:
            body = dict(result)
            claimed = str(body.pop("result_sha256", ""))
            if (
                set(result)
                != {
                    "schema_version",
                    "design_sha256",
                    "contract_sha256",
                    "claim_sha256",
                    "test_access_sha256",
                    "selected_population_layer",
                    "sealed_test_population_manifest_sha256",
                    "predictive_result_sha256",
                    "economic_result_sha256",
                    "optional_comparison_sha256",
                    "optional_uplift_gate_passed",
                    "ai_comparison_sha256",
                    "ai_model",
                    "ai_model_digest",
                    "ai_uplift_gate_passed",
                    "ai_enabled_candidate",
                    "candidate_accepted",
                    "automatic_promotion",
                    "profitability_claim",
                    "paper_trading_authority",
                    "live_trading_authority",
                    "result_sha256",
                }
                or result.get("schema_version")
                != POLYMARKET_ROUND21_SEALED_RESULT_SCHEMA_VERSION
                or result.get("design_sha256")
                != POLYMARKET_ROUND21_SEALED_DESIGN_SHA256
                or result.get("contract_sha256")
                != POLYMARKET_ROUND21_CONTRACT_SHA256
                or claimed != row["result_sha256"]
                or claimed != _canonical_sha256(body)
                or result.get("claim_sha256") != claim.claim_sha256
                or result.get("test_access_sha256") != row["test_access_sha256"]
                or result.get("selected_population_layer")
                != claim.selected_population_layer
                or result.get("sealed_test_population_manifest_sha256")
                != claim.sealed_test_population_manifest_sha256
                or (result.get("ai_model"), result.get("ai_model_digest"))
                != (claim.nominated_ai_model, claim.nominated_ai_model_digest)
                or any(
                    result.get(name) is not False
                    for name in (
                        "automatic_promotion",
                        "profitability_claim",
                        "paper_trading_authority",
                        "live_trading_authority",
                    )
                )
            ):
                raise ValueError("Round 21 stored sealed result differs")
        if failure is not None and (
            set(failure)
            != {
                "schema_version",
                "claim_sha256",
                "failed_at_ms",
                "reason",
                "return_to_development",
            }
            or failure.get("schema_version")
            != "polymarket-round21-one-use-failure-v1"
            or failure.get("claim_sha256") != claim.claim_sha256
            or failure.get("return_to_development") is not False
            or not str(failure.get("reason") or "").strip()
            or type(failure.get("failed_at_ms")) is not int
            or int(failure["failed_at_ms"]) < claim.opened_at_ms
        ):
            raise ValueError("Round 21 stored one-use failure differs")
        expected_event_types = ["claim_opened"]
        if row["test_access_sha256"] is not None:
            expected_event_types.append("test_access_consumed")
        if status == "completed":
            expected_event_types.append("evaluation_completed")
        elif status == "failed":
            expected_event_types.append("evaluation_failed")
        event_types = [str(value["event_type"]) for value in event_payloads]
        if (
            event_types != expected_event_types
            or event_payloads[0]["details"]
            != {"pretest_manifest_sha256": claim.pretest_manifest_sha256}
            or event_payloads[0]["observed_at_ms"] != claim.opened_at_ms
            or (
                row["test_access_sha256"] is not None
                and (
                    event_payloads[1]["details"]
                    != {"test_access_sha256": row["test_access_sha256"]}
                    or event_payloads[1]["observed_at_ms"]
                    != row["access_started_ms"]
                )
            )
            or (
                status == "completed"
                and event_payloads[-1]["details"]
                != {"result_sha256": row["result_sha256"]}
            )
            or (
                status == "failed"
                and (
                    event_payloads[-1]["details"]
                    != {"failure_sha256": _canonical_sha256(failure)}
                    or event_payloads[-1]["observed_at_ms"]
                    != failure["failed_at_ms"]
                )
            )
        ):
            raise ValueError("Round 21 one-use event sequence differs")
        return {
            "schema_version": POLYMARKET_ROUND21_ONE_USE_STORE_SCHEMA_VERSION,
            "status": status,
            "claim": claim.asdict(),
            "test_access_sha256": row["test_access_sha256"],
            "test_access_consumed": row["test_access_sha256"] is not None,
            "result": result,
            "failure": failure,
            "event_count": len(events),
            "event_chain_head_sha256": previous,
        }


def execute_round21_one_use(
    *,
    store_path: str | Path,
    claim: Round21OneUseClaim,
    evaluator: Callable[[str], Round21SealedEvaluationResult],
) -> Round21SealedEvaluationResult:
    """Consume access before invoking the only callback allowed to load test data."""

    selected = claim.validated()
    with Round21OneUseStore(store_path) as store:
        stored = store.open_claim(selected)
        snapshot = store.snapshot()
        if snapshot["status"] != "claim_open":
            raise RuntimeError("Round 21 one-use evaluation cannot be reopened")
        access = store.consume_test_access(stored)
        try:
            result = evaluator(access)
            if not isinstance(result, Round21SealedEvaluationResult):
                raise TypeError("Round 21 evaluator returned an invalid result type")
            return store.complete(stored, result)
        except BaseException as exc:
            store.fail(
                stored,
                reason=f"{type(exc).__name__}: {str(exc)[:400]}",
            )
            raise


credentials_used = False
account_connected = False
binance_execution_connected = False
automatic_promotion = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_ONE_USE_CLAIM_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_ONE_USE_STORE_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_PRETEST_MANIFEST_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_TEST_ACCESS_SCHEMA_VERSION",
    "Round21OneUseClaim",
    "Round21OneUseStore",
    "Round21PretestManifest",
    "build_round21_pretest_manifest",
    "create_round21_one_use_claim",
    "execute_round21_one_use",
]
