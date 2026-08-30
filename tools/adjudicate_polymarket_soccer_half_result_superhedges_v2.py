"""Representation-only v2 for retained 24-or-48-hour half-result fallbacks."""

from __future__ import annotations

from typing import Any, Mapping

from tools import adjudicate_polymarket_soccer_half_result_superhedges as base


def _validate_rules(
    main: Mapping[str, Any],
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> None:
    for role in base.ROLES:
        main_text = base._description(main[role])
        first_text = base._description(first[role])
        second_text = base._description(second[role])
        first_fallback = any(
            f"If no acceptable data is available within {hours} hours" in first_text
            for hours in (24, 48)
        )
        second_fallback = any(
            f"If no acceptable data is available within {hours} hours" in second_text
            for hours in (24, 48)
        )
        if not (
            "90 minutes of regular play plus stoppage time" in main_text
            and "first 45 minutes of regular play plus stoppage time" in first_text
            and "second half of regular play plus second-half stoppage time"
            in second_text
            and first_fallback
            and "resolve 50-50" in first_text
            and second_fallback
            and "resolve 50-50" in second_text
        ):
            raise ValueError(f"time-scope or fallback rule changed for {role}")
        if role == "draw":
            if not (
                'canceled entirely, with no make-up game, this market will resolve to "Yes"'
                in main_text
                and 'canceled entirely, with no make-up game, this market will resolve to "Yes"'
                in first_text
                and 'canceled entirely, with no make-up game, this market will resolve to "Draw"'
                in second_text
            ):
                raise ValueError("draw cancellation rule changed")
        elif not (
            'canceled entirely, with no make-up game, this market will resolve "No"'
            in main_text
            and 'canceled entirely, with no make-up game, this market will resolve "No"'
            in first_text
            and 'canceled entirely, with no make-up game, this market will resolve to "Draw"'
            in second_text
        ):
            raise ValueError(f"team cancellation rule changed for {role}")


base._validate_rules = _validate_rules


if __name__ == "__main__":
    raise SystemExit(base.main())
