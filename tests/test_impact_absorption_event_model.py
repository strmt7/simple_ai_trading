from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import pytest


torch = pytest.importorskip("torch")

from simple_ai_trading.impact_absorption_event_model import (  # noqa: E402
    ROUND74_EVENT_MODEL_CANDIDATES,
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
    return torch.randn(
        batch_size,
        sequence_length,
        len(ROUND74_EVENT_FEATURE_NAMES),
        generator=generator,
    )


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
        len(ROUND74_EVENT_PAYOFF_SIDES),
        len(ROUND74_EVENT_PAYOFF_QUANTILES),
    )
    differences = (
        output.payoff_quantiles_bps[:, :, 1:]
        - output.payoff_quantiles_bps[:, :, :-1]
    )
    assert bool((differences >= 0.0).all())


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
    payoff = torch.tensor(((1.0, -2.0), (0.5, 0.2), (-0.7, 1.3)))
    adverse = torch.tensor(((0.0, 1.0), (0.0, 0.0), (1.0, 0.0)))
    unpredictable = torch.tensor(((0.0,), (1.0,), (0.0,)))

    loss, components = round74_event_model_loss(
        output,
        net_payoff_bps=payoff,
        adverse_selection=adverse,
        regime_unpredictable=unpredictable,
    )
    loss.backward()

    assert bool(torch.isfinite(loss))
    assert set(components) == {
        "pinball",
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


@pytest.mark.parametrize(
    ("target_name", "target"),
    (
        ("adverse_selection", torch.tensor(((0.0, 1.1),) * 3)),
        ("regime_unpredictable", torch.tensor(((0.0, 0.0),) * 3)),
    ),
)
def test_round74_loss_rejects_invalid_targets(
    target_name: str,
    target: torch.Tensor,
) -> None:
    output = Round74EventPoolingMLP(dropout=0.0)(_inputs())
    arguments = {
        "net_payoff_bps": torch.zeros(3, 2),
        "adverse_selection": torch.zeros(3, 2),
        "regime_unpredictable": torch.zeros(3, 1),
    }
    arguments[target_name] = target

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
