"""Run a bounded Round 25 TCN mechanics probe on one explicit backend."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np

from simple_ai_trading.compute import resolve_backend, torch_device_for_backend
from simple_ai_trading.polymarket_round25_controls import (
    POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND,
)
from simple_ai_trading.polymarket_round25_tcn import (
    POLYMARKET_ROUND25_TCN_FIT_CONTRACT_SHA256,
    POLYMARKET_ROUND25_TCN_GRADIENT_CLIP_NORM,
    _adamw_step,
    _load_state_bytes,
    _model,
    _state_bytes,
    round25_tcn_loss,
    round25_tcn_parameter_count,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-tcn-directml-host-probe-2026-08-10.json"
)


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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def run_probe(*, backend_name: str) -> dict[str, object]:
    import torch

    backend = resolve_backend(backend_name, require=True)
    device = torch_device_for_backend(backend)
    seed = 1729
    torch.manual_seed(seed)
    model = _model().to(device)
    parameters = tuple(model.parameters())
    first_moments = tuple(torch.zeros_like(parameter) for parameter in parameters)
    second_moments = tuple(torch.zeros_like(parameter) for parameter in parameters)
    sequences = torch.zeros((16, 64, 149), dtype=torch.float32, device=device)
    sequences[:, -1, -1] = 1.0
    labels = torch.ones(16, dtype=torch.float32, device=device)
    prior = torch.full((16,), 0.5, dtype=torch.float32, device=device)
    weights = torch.full((16,), 1.0 / 16.0, dtype=torch.float32, device=device)
    auxiliary_targets = torch.zeros((16, 2), dtype=torch.float32, device=device)
    auxiliary_mask = torch.zeros((16, 2), dtype=torch.bool, device=device)
    before = _state_bytes(model)
    started = time.perf_counter()
    terminal_raw, auxiliary_prediction = model(sequences)
    total, terminal, auxiliary = round25_tcn_loss(
        terminal_raw,
        auxiliary_prediction,
        labels,
        prior,
        weights,
        auxiliary_targets,
        auxiliary_mask,
    )
    total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        POLYMARKET_ROUND25_TCN_GRADIENT_CLIP_NORM,
    )
    _adamw_step(
        parameters,
        first_moments,
        second_moments,
        step=1,
    )
    after = _state_bytes(model)
    reloaded = _model().to(device)
    _load_state_bytes(reloaded, after)
    roundtrip = _state_bytes(reloaded)
    reloaded.eval()
    with torch.no_grad():
        reloaded_raw, _unused_auxiliary = reloaded(sequences)
        bounded = POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND * torch.tanh(
            reloaded_raw / POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND
        )
        probabilities = torch.sigmoid(bounded).detach().cpu().numpy().astype("<f4")
    elapsed = time.perf_counter() - started
    losses = (
        float(total.detach().cpu().item()),
        float(terminal.detach().cpu().item()),
        *(float(value.detach().cpu().item()) for value in auxiliary),
    )
    observed_gradient_norm = float(gradient_norm.detach().cpu().item())
    if (
        not all(np.isfinite(value) for value in losses)
        or not np.isfinite(observed_gradient_norm)
        or before == after
        or after != roundtrip
        or probabilities.shape != (16,)
        or not np.all(np.isfinite(probabilities))
        or not np.all((probabilities > 0.0) & (probabilities < 1.0))
    ):
        raise RuntimeError("Round 25 TCN host probe failed its mechanics gates")
    source = ROOT / "src" / "simple_ai_trading" / "polymarket_round25_tcn.py"
    tool = Path(__file__).resolve()
    return {
        "schema_version": "polymarket-round25-tcn-host-probe-v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "runtime_mechanics_verified",
        "fixture": {
            "kind": "deterministic_numerical_runtime_fixture_no_market_data",
            "seed": seed,
            "conditions": 1,
            "endpoints": 16,
            "sequence_shape": [16, 64, 149],
            "target_class": 1,
            "market_prior_probability": 0.5,
            "auxiliary_targets_available": False,
        },
        "backend": {
            "requested": backend.requested,
            "kind": backend.kind,
            "device": backend.device,
            "vendor": backend.vendor,
            "selection": backend.selection,
            "reason": backend.reason,
            "request_satisfied": backend.request_satisfied,
            "accelerated": backend.accelerated,
        },
        "runtime": {
            "operating_system": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "torch_directml": _distribution_version("torch-directml"),
        },
        "source": {
            "fit_contract_sha256": POLYMARKET_ROUND25_TCN_FIT_CONTRACT_SHA256,
            "tcn_module_path": source.relative_to(ROOT).as_posix(),
            "tcn_module_sha256": _file_sha256(source),
            "probe_tool_path": tool.relative_to(ROOT).as_posix(),
            "probe_tool_sha256": _file_sha256(tool),
        },
        "mechanics": {
            "parameter_count": round25_tcn_parameter_count(),
            "state_bytes": len(after),
            "state_before_sha256": hashlib.sha256(before).hexdigest(),
            "state_after_sha256": hashlib.sha256(after).hexdigest(),
            "state_changed": before != after,
            "state_roundtrip_byte_identity": after == roundtrip,
            "total_loss": losses[0],
            "terminal_loss": losses[1],
            "auxiliary_1000ms_loss": losses[2],
            "auxiliary_5000ms_loss": losses[3],
            "gradient_norm_before_clip": observed_gradient_norm,
            "probability_sha256": hashlib.sha256(
                probabilities.tobytes(order="C")
            ).hexdigest(),
            "minimum_probability": float(np.min(probabilities)),
            "maximum_probability": float(np.max(probabilities)),
            "elapsed_seconds": elapsed,
        },
        "claims": {
            "market_data_used": False,
            "model_fitted": False,
            "predictive_edge_verified": False,
            "profitability_verified": False,
            "ai_uplift_verified": False,
            "paper_authority": False,
            "live_authority": False,
            "order_submitted": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="directml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = run_probe(backend_name=args.backend)
    result["evidence_sha256"] = _canonical_sha256(result)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
