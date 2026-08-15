from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / (
    "docs/model-research/polymarket/"
    "round-028-community-hypothesis-review-v1.json"
)
EXPECTED_SHA256 = "e95c18ffb7972e0b785799f109fb8c38f2e7d6ee5671622218b739aa210ab859"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def test_round28_community_review_is_tamper_evident_and_non_authoritative() -> None:
    review = json.loads(REVIEW.read_text(encoding="ascii"))
    claimed = review.pop("review_sha256")

    assert claimed == EXPECTED_SHA256
    assert _canonical_sha256(review) == claimed
    assert review["evidence_classification"]["status"] == "hypothesis_source_only"
    assert review["product_regime_mismatch"][
        "historical_results_transferable_without_retest"
    ] is False
    assert review["side_effect_policy"]["active_model_contract_changed"] is False
    assert set(review["authority"].values()) == {False}
