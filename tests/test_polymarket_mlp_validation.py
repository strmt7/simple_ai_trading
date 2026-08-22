from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from simple_ai_trading import polymarket_mlp as mlp
from simple_ai_trading.polymarket_action_value import POLYMARKET_ACTION_FEATURE_NAMES


_BACKEND_IDENTITY_SHA256 = (
    "a301c1208b5c09a7f81248a068c7ece46daa829fd8b600c31ca857ebfe14f2e8"
)
_MEMBER_SHA256 = {
    4701: "6acdd3b2937496b70e7ace3a74fdd74f0dc24ed2d5b7f66b41d6dc9f60c2c54c",
    4702: "9b0bb6a614635774341790b977df578502274faa7d7681ca6adfa2fd9033f947",
    4703: "9076ecbe6bfd1cfc6dc2aba80d9516bf2012d593e380d675742065eb68a64716",
}
_ENSEMBLE_SHA256 = "da8e3a83be90a57fa4bb85c0b6409ec583b73f19971cb2ae8307fdffc4787690"
_FEATURE_COUNT = len(POLYMARKET_ACTION_FEATURE_NAMES)


def _sha256(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _backend() -> mlp.PolymarketMLPBackendEvidence:
    return mlp.PolymarketMLPBackendEvidence(
        requested="cpu",
        kind="cpu",
        device="cpu",
        vendor="test",
        fallback_reason="",
        torch_version="test",
        torch_directml_version="",
        preflight_objective=1.0,
        preflight_parameter_delta=1.0,
        preflight_seconds=0.1,
        training_seconds=1.0,
        canonical_replay_max_probability_drift=0.0,
    )


def _member(seed: int) -> mlp.PolymarketMLPMember:
    trace = (
        mlp.PolymarketMLPEpoch(
            seed=seed,
            epoch=1,
            training_loss=0.5,
            validation_log_loss=0.4,
        ),
    )
    provisional = mlp.PolymarketMLPMember(
        seed=seed,
        best_epoch=1,
        epochs_ran=1,
        hidden1_weight=(0.0,) * (64 * _FEATURE_COUNT),
        hidden1_bias=(0.0,) * 64,
        hidden2_weight=(0.0,) * (32 * 64),
        hidden2_bias=(0.0,) * 32,
        output_weight=(0.0,) * 32,
        output_bias=0.0,
        trace=trace,
        member_sha256="",
    )
    return replace(
        provisional,
        member_sha256=_sha256(provisional.identity_payload()),
    )


def _ensemble() -> mlp.PolymarketMLPEnsemble:
    provisional = mlp.PolymarketMLPEnsemble(
        dataset_sha256="a" * 64,
        feature_mean=(0.0,) * _FEATURE_COUNT,
        feature_scale=(1.0,) * _FEATURE_COUNT,
        members=tuple(_member(seed) for seed in mlp.POLYMARKET_MLP_SEEDS),
        backend=_backend(),
        reproducibility_max_probability_drift=0.0,
        ensemble_sha256="",
    )
    return replace(
        provisional,
        ensemble_sha256=_sha256(provisional.identity_payload()),
    )


def test_mlp_validation_preserves_canonical_identities() -> None:
    backend = _backend()
    ensemble = _ensemble()

    assert backend.validated() is backend
    assert _sha256(backend.identity_payload()) == _BACKEND_IDENTITY_SHA256
    assert ensemble.validated() is ensemble
    assert {
        item.seed: item.member_sha256 for item in ensemble.members
    } == _MEMBER_SHA256
    assert ensemble.ensemble_sha256 == _ENSEMBLE_SHA256


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        (replace(_backend(), requested="directml"), "backend evidence"),
        (replace(_backend(), device=""), "backend evidence"),
        (replace(_backend(), training_seconds=0.0), "backend evidence"),
        (replace(_member(4701), best_epoch=2), "member"),
        (replace(_member(4701), hidden1_bias=()), "member"),
        (
            replace(
                _member(4701),
                trace=(replace(_member(4701).trace[0], seed=4702),),
            ),
            "member",
        ),
        (
            replace(
                _ensemble(),
                feature_scale=(0.0,) * _FEATURE_COUNT,
            ),
            "ensemble",
        ),
        (replace(_ensemble(), members=_ensemble().members[:-1]), "ensemble"),
        (
            replace(
                _ensemble(),
                reproducibility_max_probability_drift=0.1,
            ),
            "ensemble",
        ),
    ],
)
def test_mlp_validation_preserves_fail_closed_rejections(
    evidence: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evidence.validated()  # type: ignore[attr-defined]
