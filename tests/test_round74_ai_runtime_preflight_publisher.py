from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


REPOSITORY = Path(__file__).resolve().parents[1]
TOOL_PATH = REPOSITORY / "tools/publish_round74_ai_runtime_preflight.py"
SPEC = importlib.util.spec_from_file_location(
    "publish_round74_ai_runtime_preflight",
    TOOL_PATH,
)
assert SPEC is not None and SPEC.loader is not None
PUBLISHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISHER)
assert isinstance(PUBLISHER, ModuleType)


def test_preflight_schema_tracks_current_ai_contract() -> None:
    assert PUBLISHER.SCHEMA_VERSION == "round-074-local-ai-runtime-preflight-v3"
    assert PUBLISHER.ROUND74_AI_REVIEW_REQUEST_SCHEMA_VERSION == (
        "round-074-ai-review-request-v5"
    )
    assert PUBLISHER.ROUND74_AI_PROMPT_PAYLOAD_SCHEMA_VERSION == (
        "round-074-ai-prompt-payload-v7"
    )
    assert PUBLISHER.ROUND74_AI_REVIEW_PANEL_SCHEMA_VERSION == (
        "round-074-ai-review-panel-v9"
    )


def test_synthetic_request_exercises_profile_and_temporal_path() -> None:
    request = PUBLISHER._synthetic_request()
    request.validate()

    assert request.risk_profile == "conservative"
    assert request.feature_last[0] == 1.0
    assert request.feature_last[5:8] == (1.0, 0.0, 0.0)
    assert sum(request.feature_mean[:5]) == 1.0
    assert request.feature_mean[5:8] == (1.0, 0.0, 0.0)
    assert len(request.feature_recent_block_means) == (
        PUBLISHER.ROUND74_AI_TEMPORAL_BLOCK_COUNT
    )
    assert all(
        len(row) == len(PUBLISHER.ROUND74_AI_TEMPORAL_FEATURE_NAMES)
        for row in request.feature_recent_block_means
    )
    spread_path = tuple(row[0] for row in request.feature_recent_block_means)
    volatility_path = tuple(row[10] for row in request.feature_recent_block_means)
    assert spread_path == tuple(sorted(spread_path))
    assert volatility_path == spread_path
    assert len(set(spread_path)) == PUBLISHER.ROUND74_AI_TEMPORAL_BLOCK_COUNT


def test_synthetic_request_is_hash_stable_after_round_trip() -> None:
    request = PUBLISHER._synthetic_request()
    restored = PUBLISHER.Round74AIReviewRequest.from_dict(request.as_dict())

    assert restored == request
    assert restored.request_sha256 == request.request_sha256
