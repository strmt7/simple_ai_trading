"""Shared, restart-safe operator primitives for Polymarket Round 27."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_replay import PolymarketEvidenceReplay
from .polymarket_round27_economics import Round27EconomicBookBatch
from .polymarket_round27_model import Round27ProbabilityModel
from .storage import write_json_atomic


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 27 operator artifact contains duplicate keys")
        output[key] = value
    return output


def load_mapping(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.resolve(strict=True).read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Round 27 operator artifact is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Round 27 operator artifact must be an object")
    return value


def artifact_writer(path: Path, hash_field: str):
    def write(value: Mapping[str, object]) -> str:
        payload = dict(value)
        claimed = str(payload.get(hash_field) or "")
        body = dict(payload)
        body.pop(hash_field, None)
        if claimed != canonical_sha256(body):
            raise ValueError("Round 27 operator artifact hash differs")
        if path.exists():
            persisted = load_mapping(path)
        else:
            write_json_atomic(path, payload, indent=2, sort_keys=True)
            persisted = load_mapping(path)
        if persisted != payload:
            raise ValueError("Round 27 operator artifact persistence differs")
        return claimed

    return write


def source_recomputed_artifact(
    path: Path,
    hash_field: str,
    build: Callable[[], Mapping[str, object]],
) -> dict[str, object]:
    """Recompute a source-derived artifact before accepting a restart checkpoint."""

    recomputed = dict(build())
    artifact_writer(path, hash_field)(recomputed)
    return recomputed


def model_identity(
    model: Round27ProbabilityModel | None,
    *,
    contract_sha256: str,
) -> tuple[str, str]:
    if model is None:
        return (
            "market_prior",
            canonical_sha256(
                {
                    "model_name": "market_prior",
                    "contract_sha256": contract_sha256,
                }
            ),
        )
    payload = model.asdict()
    return model.model_name, str(payload["model_sha256"])


def economic_book_batches(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    condition_ids: Sequence[str],
    maximum_conditions: int,
) -> Iterable[Round27EconomicBookBatch]:
    for start in range(0, len(condition_ids), maximum_conditions):
        batch_ids = tuple(condition_ids[start : start + maximum_conditions])
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id=run_id,
            allow_segmented_gaps=True,
            include_resolutions=False,
            condition_ids=batch_ids,
            materialized_minimum_depth_levels=0,
            cap_materialized_depth_to_minimum_order_size=True,
        )
        if {market.condition_id for market in replay.markets} != set(batch_ids):
            raise ValueError("Round 27 operator replay market population differs")
        yield Round27EconomicBookBatch(
            condition_ids=batch_ids,
            books=replay.books,
        ).validated()


__all__ = [
    "artifact_writer",
    "canonical_json",
    "canonical_sha256",
    "economic_book_batches",
    "load_mapping",
    "model_identity",
    "source_recomputed_artifact",
]
