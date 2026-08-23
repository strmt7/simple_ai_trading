from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import textwrap


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "simple_ai_trading"
LIVE_BOUNDARY_IMPORTS = {
    "polymarket_live.py": frozenset(),
    "polymarket_live_activation.py": frozenset({"polymarket_live"}),
    "polymarket_live_v2.py": frozenset(
        {"polymarket", "polymarket_fees", "polymarket_live"}
    ),
    "polymarket_live_runtime.py": frozenset({"polymarket_live", "polymarket_live_v2"}),
    "polymarket_live_stop.py": frozenset(
        {"polymarket_live", "polymarket_runtime_control"}
    ),
    "polymarket_live_settlement.py": frozenset(
        {"polymarket_live", "polymarket_live_v2"}
    ),
    "polymarket_live_promotion.py": frozenset(
        {
            "polymarket_cross_regime_evaluation",
            "polymarket_live",
            "polymarket_live_manifest",
            "polymarket_live_risk",
        }
    ),
    "polymarket_live_risk.py": frozenset({"polymarket_live"}),
    "polymarket_live_qualification.py": frozenset(
        {
            "polymarket_live",
            "polymarket_live_promotion",
            "polymarket_live_v2",
        }
    ),
    "polymarket_autonomous.py": frozenset(
        {
            "polymarket_external_signal",
            "polymarket_fees",
            "polymarket_live",
            "polymarket_live_promotion",
            "polymarket_live_qualification",
        }
    ),
    "polymarket_autonomous_runtime.py": frozenset(
        {
            "polymarket",
            "polymarket_autonomous",
            "polymarket_live",
            "polymarket_live_promotion",
            "polymarket_live_qualification",
            "polymarket_live_risk",
            "polymarket_live_runtime",
            "polymarket_live_settlement",
            "polymarket_live_stop",
        }
    ),
    "polymarket_round16_shadow.py": frozenset(
        {
            "polymarket_historical_shadow",
            "polymarket_round16",
            "polymarket_round16_dataset",
            "polymarket_round16_evaluation",
            "polymarket_round16_model",
            "polymarket_round16_targets",
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
    "polymarket_round17_cohort.py": frozenset(
        {
            "polymarket",
            "polymarket_replay",
            "polymarket_round17_dataset",
            "polymarket_round17_features",
            "polymarket_round17_model",
        }
    ),
    "polymarket_round17_campaign_operator.py": frozenset(
        {
            "polymarket",
            "polymarket_recorder",
            "polymarket_replay",
            "polymarket_round14_campaign",
            "polymarket_round14_dataset",
            "polymarket_round17_cohort",
            "polymarket_round17_dataset",
        }
    ),
    "polymarket_round17_development_operator.py": frozenset(
        {
            "polymarket",
            "polymarket_round14_contract",
            "polymarket_round17_campaign_operator",
            "polymarket_round17_cohort",
            "polymarket_round17_economic",
            "polymarket_round17_features",
            "polymarket_round17_model",
            "polymarket_round17_outcomes",
            "polymarket_round17_resolution",
            "polymarket_round17_uncertainty",
            "storage",
        }
    ),
    "polymarket_round17_resolution.py": frozenset(
        {
            "polymarket",
            "polymarket_replay",
            "polymarket_resolution",
            "polymarket_round17_cohort",
        }
    ),
    "polymarket_round17_economic.py": frozenset(
        {
            "polymarket_round14_contract",
            "polymarket_round17_execution",
            "polymarket_round17_features",
            "polymarket_round17_model",
            "polymarket_round17_uncertainty",
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
            "polymarket_round17_uncertainty",
        }
    ),
    "polymarket_round17_uncertainty.py": frozenset(
        {
            "polymarket_round17_execution",
            "polymarket_round17_features",
            "polymarket_round17_model",
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


def test_polymarket_live_cli_has_no_direct_binance_dependency() -> None:
    path = SOURCE_ROOT / "polymarket_live_cli.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    assert all(
        "binance" not in str(node.module or "").lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )


def test_operator_status_imports_without_optional_advisory_stack(
    tmp_path: Path,
) -> None:
    blocked_modules = (
        "simple_ai_trading.polymarket_autonomous_runtime",
        "simple_ai_trading.polymarket_historical_shadow",
        "simple_ai_trading.polymarket_historical_shadow_feed",
        "simple_ai_trading.polymarket_live_promotion",
        "simple_ai_trading.polymarket_live_qualification",
        "simple_ai_trading.polymarket_round16_decision",
        "simple_ai_trading.polymarket_round16_shadow",
    )
    code = textwrap.dedent(
        f"""
        import importlib.abc
        from pathlib import Path
        import sys

        blocked = {blocked_modules!r}

        class BlockOptionalStack(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in blocked:
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, BlockOptionalStack())
        from simple_ai_trading import polymarket_live_cli

        payload = polymarket_live_cli._local_status(
            Path({str(tmp_path / "missing.sqlite3")!r})
        )
        assert payload["venue"] == "polymarket"
        assert payload["ledger_exists"] is False
        """
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(SOURCE_ROOT.parent),
            environment.get("PYTHONPATH", ""),
        )
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
