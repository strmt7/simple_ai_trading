"""Entry-only compatibility for payoff experts dependent on simulated costs."""

from collections.abc import Mapping

from .execution_simulation import EXECUTION_MODEL_VERSION


def payoff_execution_version_error(model: object) -> str | None:
    """Return an entry veto; never modify or prevent reading an old artifact.

    Other model families retain their existing gates. A matching version is
    compatibility metadata, not training, promotion or profitability evidence.
    """
    experts = getattr(model, "hybrid_experts", None)
    if not isinstance(experts, (list, tuple)):
        return "model-execution-cost-metadata-invalid"
    for expert in experts:
        kind = getattr(expert, "kind", None)
        if not isinstance(kind, str):
            return "model-execution-cost-metadata-invalid"
        if not kind.startswith("signed_payoff_"):
            continue
        params = getattr(expert, "params", None)
        if not isinstance(params, Mapping):
            return "model-execution-cost-metadata-invalid"
        version = params.get("execution_model_version")
        if version is None:
            return "model-execution-cost-version-missing"
        if not isinstance(version, str):
            return "model-execution-cost-metadata-invalid"
        if not version:
            return "model-execution-cost-version-missing"
        if version != EXECUTION_MODEL_VERSION:
            return "model-execution-cost-version-mismatch"
    return None
