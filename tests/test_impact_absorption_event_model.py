from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import pytest


torch = pytest.importorskip("torch")

from simple_ai_trading.impact_absorption_event_model import (  # noqa: E402
    ROUND74_EVENT_MODEL_CANDIDATES,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_QUANTILES,
    ROUND74_EVENT_PAYOFF_SIDES,
    Round74CausalEventTCN,
    Round74EventPoolingMLP,
    build_round74_event_model,
    round74_event_model_loss,
)
from simple_ai_trading.impact_absorption_event_sequence import (  # noqa: E402
    ROUND74_EVENT_FEATURE_NAMES,
)


def _inputs(batch_size: int = 3, sequence_length: int = 32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7402)
    values = torch.randn(
        batch_size,
        sequence_length,
        len(ROUND74_EVENT_FEATURE_NAMES),
        generator=generator,
    )
    values[:, :, :8] = 0.0
    for batch_index in range(batch_size):
        for event_index in range(sequence_length):
            values[batch_index, event_index, event_index % 5] = 1.0
            values[batch_index, event_index, 5 + batch_index % 3] = 1.0
    return values


@pytest.mark.parametrize(
    "model",
    (
        Round74EventPoolingMLP(dropout=0.0),
        Round74CausalEventTCN(dropout=0.0),
    ),
)
def test_round74_candidate_outputs_are_finite_and_monotone(model: object) -> None:
    output = model(_inputs())
    output.validate(3)

    assert output.payoff_quantiles_bps.shape == (
        3,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
        len(ROUND74_EVENT_PAYOFF_QUANTILES),
    )
    for quantiles in (
        output.payoff_quantiles_bps,
        output.maximum_adverse_excursion_quantiles_bps,
    ):
        differences = quantiles[..., 1:] - quantiles[..., :-1]
        assert bool((differences >= 0.0).all())
    assert bool(
        (output.maximum_adverse_excursion_quantiles_bps >= 0.0).all()
    )


def test_round74_tcn_is_strictly_causal() -> None:
    model = Round74CausalEventTCN(dropout=0.0).eval()
    values = _inputs(batch_size=2, sequence_length=41)
    prefix_length = 23

    with torch.no_grad():
        full = model._encode_events(values)
        prefix = model._encode_events(values[:, :prefix_length, :])

    torch.testing.assert_close(
        full[:, :, prefix_length - 1],
        prefix[:, :, -1],
        rtol=1e-6,
        atol=1e-6,
    )


def test_round74_loss_is_finite_and_backpropagates() -> None:
    model = Round74CausalEventTCN(dropout=0.0)
    output = model(_inputs())
    generator = torch.Generator().manual_seed(7403)
    action_shape = (
        3,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    payoff = torch.randn(action_shape, generator=generator)
    maximum_adverse_excursion = torch.rand(
        action_shape,
        generator=generator,
    )
    adverse = torch.randint(
        0,
        2,
        action_shape,
        generator=generator,
    ).float()
    unpredictable = torch.randint(
        0,
        2,
        (3, len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)),
        generator=generator,
    ).float()

    loss, components = round74_event_model_loss(
        output,
        net_payoff_bps=payoff,
        maximum_adverse_excursion_bps=maximum_adverse_excursion,
        adverse_selection=adverse,
        regime_unpredictable=unpredictable,
    )
    loss.backward()

    assert bool(torch.isfinite(loss))
    assert set(components) == {
        "payoff_pinball",
        "maximum_adverse_excursion_pinball",
        "positive_bce",
        "adverse_bce",
        "unpredictability_bce",
    }
    assert any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool((parameter.grad != 0.0).any())
        for parameter in model.parameters()
    )


def test_round74_loss_excludes_censored_actions_and_rejects_empty_batches() -> None:
    output = Round74EventPoolingMLP(dropout=0.0)(_inputs())
    action_shape = (
        3,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    payoff = torch.zeros(action_shape)
    maximum_adverse_excursion = torch.zeros(action_shape)
    adverse = torch.zeros(action_shape)
    unpredictable = torch.zeros(
        3,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
    )
    eligibility = torch.ones(action_shape)
    eligibility[0, 0, 0] = 0.0
    first, _ = round74_event_model_loss(
        output,
        net_payoff_bps=payoff,
        maximum_adverse_excursion_bps=maximum_adverse_excursion,
        adverse_selection=adverse,
        regime_unpredictable=unpredictable,
        action_eligibility=eligibility,
    )
    changed_payoff = payoff.clone()
    changed_payoff[0, 0, 0] = 1e9
    changed_excursion = maximum_adverse_excursion.clone()
    changed_excursion[0, 0, 0] = 1e9
    changed_adverse = adverse.clone()
    changed_adverse[0, 0, 0] = 1.0
    second, _ = round74_event_model_loss(
        output,
        net_payoff_bps=changed_payoff,
        maximum_adverse_excursion_bps=changed_excursion,
        adverse_selection=changed_adverse,
        regime_unpredictable=unpredictable,
        action_eligibility=eligibility,
    )

    torch.testing.assert_close(first, second)
    with pytest.raises(ValueError, match="no eligible targets"):
        round74_event_model_loss(
            output,
            net_payoff_bps=payoff,
            maximum_adverse_excursion_bps=maximum_adverse_excursion,
            adverse_selection=adverse,
            regime_unpredictable=unpredictable,
            action_eligibility=torch.zeros(action_shape),
        )
    fractional = torch.ones(action_shape)
    fractional[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="not binary"):
        round74_event_model_loss(
            output,
            net_payoff_bps=payoff,
            maximum_adverse_excursion_bps=maximum_adverse_excursion,
            adverse_selection=adverse,
            regime_unpredictable=unpredictable,
            action_eligibility=fractional,
        )


@pytest.mark.parametrize(
    ("target_name", "invalid_kind"),
    (
        ("maximum_adverse_excursion_bps", "negative"),
        ("adverse_selection", "outside_probability"),
        ("regime_unpredictable", "wrong_shape"),
    ),
)
def test_round74_loss_rejects_invalid_targets(
    target_name: str,
    invalid_kind: str,
) -> None:
    output = Round74EventPoolingMLP(dropout=0.0)(_inputs())
    action_shape = (
        3,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    arguments = {
        "net_payoff_bps": torch.zeros(action_shape),
        "maximum_adverse_excursion_bps": torch.zeros(action_shape),
        "adverse_selection": torch.zeros(action_shape),
        "regime_unpredictable": torch.zeros(
            3,
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        ),
    }
    if invalid_kind == "negative":
        arguments[target_name][0, 0, 0] = -1.0
    elif invalid_kind == "outside_probability":
        arguments[target_name][0, 0, 0] = 1.1
    else:
        arguments[target_name] = torch.zeros(action_shape)

    with pytest.raises(ValueError):
        round74_event_model_loss(output, **arguments)


def test_round74_candidate_registry_fails_closed() -> None:
    assert tuple(
        type(build_round74_event_model(candidate_id)).__name__
        for candidate_id in ROUND74_EVENT_MODEL_CANDIDATES
    ) == ("Round74EventPoolingMLP", "Round74CausalEventTCN")
    with pytest.raises(ValueError, match="unsupported"):
        build_round74_event_model("future-model")


@pytest.mark.skipif(
    importlib.util.find_spec("torch_directml") is None,
    reason="torch-directml is not installed",
)
def test_round74_directml_preflight_has_no_cpu_fallback() -> None:
    # PyTorch 2.4 can initialize its autograd worker pool for CPU-only work and
    # then fail internally when a private-use device is introduced. A fresh
    # process is the actual backend boundary used by production training jobs.
    script = (
        "import json;"
        "from simple_ai_trading.impact_absorption_event_model "
        "import round74_event_model_preflight;"
        "_,e=round74_event_model_preflight('directml');"
        "print(json.dumps(e,sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    evidence = json.loads(completed.stdout)

    assert evidence["backend_kind"] == "directml"
    assert evidence["backend_accelerated"] is True
    assert evidence["backend_vendor"]
    assert evidence["cpu_fallback_warning_count"] == 0
    assert evidence["financial_edge_tested"] is False
    assert evidence["profitability_claim"] is False
    assert set(evidence["candidates"]) == set(ROUND74_EVENT_MODEL_CANDIDATES)
