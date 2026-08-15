"""Delegate Round 74 model development to the installed CLI contract."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.entrypoint import main as cli_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    selected = list(sys.argv[1:] if argv is None else argv)
    has_repository = any(
        value == "--repository" or value.startswith("--repository=")
        for value in selected
    )
    if not has_repository:
        selected[:0] = ["--repository", str(REPOSITORY)]
    return cli_main(["binance-round74-develop", *selected])


if __name__ == "__main__":
    raise SystemExit(main())
