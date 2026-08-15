from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "simple_ai_trading"

REQUIRED_POLYMARKET_AUTHORITY_MODULES = frozenset(
    {
        "polymarket_live_activation.py",
        "polymarket_live.py",
        "polymarket_live_v2.py",
        "polymarket_live_runtime.py",
        "polymarket_live_stop.py",
        "polymarket_live_settlement.py",
        "polymarket_live_risk.py",
        "polymarket_live_promotion.py",
        "polymarket_live_qualification.py",
        "polymarket_runtime_control.py",
        "polymarket_autonomous.py",
        "polymarket_autonomous_runtime.py",
        "polymarket_live_cli.py",
    }
)

POLYMARKET_AUTHORITY_MODULES = tuple(
    sorted(
        {
            *(path.name for path in PACKAGE_ROOT.glob("polymarket_live*.py")),
            *(path.name for path in PACKAGE_ROOT.glob("polymarket_autonomous*.py")),
            "polymarket_runtime_control.py",
        }
    )
)

BINANCE_PRIVATE_AUTHORITY_MODULES = frozenset(
    {
        "api",
        "autonomous",
        "execution_lifecycle",
        "execution_profiles",
        "position_lifecycle",
        "positions",
        "portfolio_risk",
        "risk_controls",
        "risk_workflows",
    }
)

PAPER_EXECUTION_AUTHORITY_MODULES = frozenset(
    {
        "binance_paper",
        "paper_execution",
        "polymarket_paper",
        "polymarket_paper_plan",
    }
)

BINANCE_PUBLIC_PREDICTOR_MODULES = (
    "polymarket_round21_binance_feed.py",
    "polymarket_round21_binance_features.py",
    "polymarket_round23_binance.py",
    "polymarket_historical_shadow_feed.py",
)

POLYMARKET_LIVE_MODEL_COMPONENTS = (
    "polymarket_round21_decision.py",
    "polymarket_round21_runtime.py",
    "polymarket_round21_session.py",
)

POLYMARKET_PRIVATE_AUTHORITY_MODULES = frozenset(
    Path(name).stem for name in POLYMARKET_AUTHORITY_MODULES
)


def _imported_modules(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[-1] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[-1])
    return frozenset(imported)


def _dynamic_imports(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        function = node.func
        is_import_module = (
            isinstance(function, ast.Attribute) and function.attr == "import_module"
        )
        is_dunder_import = (
            isinstance(function, ast.Name) and function.id == "__import__"
        )
        if is_import_module or is_dunder_import:
            imported.add(first.value.lower())
    return frozenset(imported)


def test_polymarket_authority_imports_no_binance_private_authority() -> None:
    assert REQUIRED_POLYMARKET_AUTHORITY_MODULES <= set(POLYMARKET_AUTHORITY_MODULES)
    for filename in POLYMARKET_AUTHORITY_MODULES:
        path = PACKAGE_ROOT / filename
        assert path.is_file(), (
            f"missing protected Polymarket authority module: {filename}"
        )
        imported = _imported_modules(path)
        forbidden = imported & BINANCE_PRIVATE_AUTHORITY_MODULES
        assert not forbidden, (
            f"{filename} imports Binance authority: {sorted(forbidden)}"
        )
        paper_authority = imported & PAPER_EXECUTION_AUTHORITY_MODULES
        assert not paper_authority, (
            f"{filename} imports paper execution authority: {sorted(paper_authority)}"
        )
        binance_named = sorted(name for name in imported if "binance" in name.lower())
        assert not binance_named, (
            f"{filename} directly imports Binance modules: {binance_named}"
        )
        dynamic_binance = sorted(
            name for name in _dynamic_imports(path) if "binance" in name
        )
        assert not dynamic_binance, (
            f"{filename} dynamically imports Binance modules: {dynamic_binance}"
        )


def test_binance_predictor_sidecars_import_no_polymarket_private_authority() -> None:
    for filename in BINANCE_PUBLIC_PREDICTOR_MODULES:
        path = PACKAGE_ROOT / filename
        assert path.is_file(), f"missing protected Binance predictor module: {filename}"
        forbidden = _imported_modules(path) & POLYMARKET_PRIVATE_AUTHORITY_MODULES
        assert not forbidden, (
            f"{filename} imports Polymarket authority: {sorted(forbidden)}"
        )


def test_live_model_components_import_no_private_binance_or_paper_authority() -> None:
    for filename in POLYMARKET_LIVE_MODEL_COMPONENTS:
        path = PACKAGE_ROOT / filename
        assert path.is_file(), f"missing protected live model component: {filename}"
        imported = _imported_modules(path)
        forbidden = imported & (
            BINANCE_PRIVATE_AUTHORITY_MODULES | PAPER_EXECUTION_AUTHORITY_MODULES
        )
        assert not forbidden, (
            f"{filename} imports non-Polymarket authority: {sorted(forbidden)}"
        )


def test_binance_predictor_sidecars_cannot_read_credentials_or_submit_orders() -> None:
    forbidden_tokens = (
        "x-mbx-apikey",
        "binance_api_key",
        "binance_secret",
        "api_secret",
        "listenkey",
        "/api/v3/order",
        "/fapi/v1/order",
        "create_order",
        "place_order",
        "cancel_order",
        "account_balance",
        "position_risk",
        "os.environ",
        "os.getenv",
    )
    for filename in BINANCE_PUBLIC_PREDICTOR_MODULES:
        source = (PACKAGE_ROOT / filename).read_text(encoding="utf-8").lower()
        found = tuple(token for token in forbidden_tokens if token in source)
        assert not found, f"{filename} contains private Binance capability: {found}"
