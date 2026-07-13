from __future__ import annotations

import pytest

from simple_ai_trading.causal_meta_label_ai_veto import (
    HASH_SAMPLE_MODULUS,
    case_is_sampled,
)


def test_round40_ai_case_sampling_is_hash_only_and_deterministic() -> None:
    sampled = "0" * 64
    rejected = "0" * 15 + "1" + "0" * 48

    assert HASH_SAMPLE_MODULUS == 2
    assert case_is_sampled(sampled) is True
    assert case_is_sampled(sampled) is True
    assert case_is_sampled(rejected) is False


def test_round40_ai_case_sampling_rejects_non_sha_identity() -> None:
    with pytest.raises(ValueError, match="not SHA-256"):
        case_is_sampled("abc")
