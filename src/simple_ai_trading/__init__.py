"""Core package for Simple AI Trading's BTC/ETH/SOL day-trading CLI and Windows app."""

from .process_security import harden_executable_search_path


# This must run before modules that discover or launch external executables.
harden_executable_search_path()

from .types import RiskProfile, StrategyConfig, RuntimeConfig  # noqa: E402

__all__ = ["RiskProfile", "StrategyConfig", "RuntimeConfig"]
