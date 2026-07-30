from __future__ import annotations

import ast
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "simple_ai_trading"
LIVE_BOUNDARY_IMPORTS = {
    "polymarket_live.py": frozenset(),
    "polymarket_live_v2.py": frozenset(
        {"paper_execution", "polymarket", "polymarket_live"}
    ),
    "polymarket_live_runtime.py": frozenset({"polymarket_live", "polymarket_live_v2"}),
    "polymarket_live_stop.py": frozenset(
        {"polymarket_live", "polymarket_runtime_control"}
    ),
    "polymarket_live_settlement.py": frozenset(
        {"polymarket_live", "polymarket_live_v2"}
    ),
    "polymarket_live_promotion.py": frozenset({"polymarket_live"}),
    "polymarket_autonomous.py": frozenset(
        {
            "paper_execution",
            "polymarket_external_signal",
            "polymarket_live",
            "polymarket_live_promotion",
        }
    ),
    "polymarket_autonomous_runtime.py": frozenset(
        {
            "polymarket",
            "polymarket_autonomous",
            "polymarket_external_signal",
            "polymarket_live",
            "polymarket_live_promotion",
            "polymarket_live_runtime",
            "polymarket_live_settlement",
            "polymarket_live_stop",
        }
    ),
    "polymarket_binance_signal.py": frozenset(
        {"polymarket_autonomous", "polymarket_external_signal"}
    ),
    "polymarket_round16_shadow.py": frozenset(
        {
            "polymarket_historical_shadow",
            "polymarket_round16",
            "polymarket_round16_dataset",
            "polymarket_round16_evaluation",
            "polymarket_round16_model",
        }
    ),
    "polymarket_round16_decision.py": frozenset(
        {
            "paper_execution",
            "polymarket",
            "polymarket_autonomous",
            "polymarket_autonomous_runtime",
            "polymarket_live",
            "polymarket_live_promotion",
            "polymarket_round16",
            "polymarket_round16_shadow",
        }
    ),
    "polymarket_round17_execution.py": frozenset(
        {
            "paper_execution",
            "polymarket",
            "polymarket_round14_contract",
            "polymarket_round17_features",
        }
    ),
    "polymarket_round17_economic.py": frozenset(
        {
            "polymarket_round14_contract",
            "polymarket_round17_execution",
            "polymarket_round17_features",
        }
    ),
    "polymarket_round17_outcomes.py": frozenset(
        {
            "paper_execution",
            "polymarket",
            "polymarket_replay",
            "polymarket_round14_contract",
            "polymarket_round17_dataset",
            "polymarket_round17_economic",
            "polymarket_round17_execution",
            "polymarket_round17_features",
        }
    ),
}


def _local_imports(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return frozenset(
        str(node.module or "").split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0
    )


def test_live_polymarket_local_import_graph_is_explicitly_isolated() -> None:
    observed = {
        name: _local_imports(SOURCE_ROOT / name) for name in LIVE_BOUNDARY_IMPORTS
    }

    assert observed == LIVE_BOUNDARY_IMPORTS
    assert all(
        "binance" not in imported
        for modules in observed.values()
        for imported in modules
    )


def test_public_binance_advisor_has_no_execution_surface() -> None:
    path = SOURCE_ROOT / "polymarket_binance_signal.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    provider = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "BinanceBtcPublicSignalProvider"
    )
    public_methods = {
        node.name
        for node in provider.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }

    assert public_methods == {"evaluate", "run", "snapshot"}
    assert not public_methods & {
        "buy",
        "cancel",
        "close_position",
        "open_position",
        "order",
        "sell",
        "submit",
    }
