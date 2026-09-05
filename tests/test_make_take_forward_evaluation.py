"""Synthetic regressions; old evaluation fixtures and results stay unchanged."""

from dataclasses import fields, replace
import importlib.util
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import simple_ai_trading.make_take_action_values as value_module
import simple_ai_trading.make_take_targets as target_module
from simple_ai_trading.make_take_evaluation import evaluate_make_take_policy
from simple_ai_trading.make_take_forward_evaluation import (
    DAY_MS,
    MakeTakeRoleInputs,
    evaluate_make_take_policy_forward,
)


def helpers():
    spec = importlib.util.spec_from_file_location(
        "legacy_evaluation_fixtures",
        Path(__file__).with_name("test_make_take_evaluation.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def six_days(start=2):
    values, base, stress = helpers()._terminal_sources()
    offsets = [(start + k * 2) * DAY_MS for k in range(3)]
    expanded_values = []
    for x in values:
        arrays = {
            f.name: np.tile(getattr(x, f.name), 3)
            for f in fields(x)
            if isinstance(getattr(x, f.name), np.ndarray)
        }
        arrays["event_index"] = np.concatenate(
            [x.event_index + k * x.event_rows for k in range(3)]
        )
        arrays["decision_time_ms"] = np.concatenate(
            [x.decision_time_ms + shift for shift in offsets]
        )
        y = replace(x, **arrays, batch_sha256="")
        expanded_values.append(
            replace(
                y, batch_sha256=value_module._sha256(value_module._batch_payload(y))
            )
        )
    expanded_targets = []
    for collection in (base, stress):
        items = []
        for x in collection:
            arrays = {
                f.name: np.tile(getattr(x, f.name), 3)
                for f in fields(x)
                if isinstance(getattr(x, f.name), np.ndarray)
            }
            arrays["terminal_time_ms"] = np.concatenate(
                [
                    np.where(x.terminal_time_ms >= 0, x.terminal_time_ms + shift, -1)
                    for shift in offsets
                ]
            )
            paths = {
                str(int(day) + shift): digest
                for shift in offsets
                for day, digest in x.day_path_sha256.items()
            }
            y = replace(
                x,
                **arrays,
                day_path_sha256=MappingProxyType(paths),
                event_rows=x.event_rows * 3,
                target_sha256="",
            )
            items.append(
                replace(
                    y,
                    target_sha256=target_module._sha256(
                        target_module._target_payload(y)
                    ),
                )
            )
        expanded_targets.append(tuple(items))
    return MakeTakeRoleInputs(
        tuple(range(start, start + 6)), tuple(expanded_values), *expanded_targets
    )


def inputs():
    h = helpers()
    policy_helper, selection, values, base, stress = h._accepted_policy()
    # The legacy fixture moves rows onto day 1 without expanding its day-path
    # map. Supply an explicit synthetic two-day source map for forward cases;
    # preserve the legacy helper and recompute this fixture's selection.
    full_targets = []
    for collection in (base, stress):
        updated = []
        for x in collection:
            y = replace(
                x,
                day_path_sha256=MappingProxyType(
                    {str(d * DAY_MS): f"{d + 100:064x}" for d in (0, 1)}
                ),
                target_sha256="",
            )
            updated.append(
                replace(
                    y,
                    target_sha256=target_module._sha256(
                        target_module._target_payload(y)
                    ),
                )
            )
        full_targets.append(tuple(updated))
    base, stress = full_targets
    selection = policy_helper.calibrate_make_take_policy(
        predictive_evaluation=policy_helper._report_with_valid_hash(),
        action_values=values,
        base_targets=base,
        stress_targets=stress,
        expected_days=(0, 1),
    )
    return policy_helper, dict(
        policy_selection=selection,
        calibration=MakeTakeRoleInputs((0, 1), values, base, stress),
        evaluation=six_days(),
        predictive_evaluation=h._evaluation_report(policy_helper),
    )


def test_legacy_fixture_proves_overlap_and_missing_day_gap():
    h = helpers()
    _, arguments = inputs()
    values, base, stress = h._terminal_sources()
    old = evaluate_make_take_policy(
        policy_selection=arguments["policy_selection"],
        predictive_evaluation=arguments["predictive_evaluation"],
        action_values=values,
        base_targets=base,
        stress_targets=stress,
        expected_days=range(6),
    )
    assert old.economic_gate_passed and old.base_metrics.closed_trades == 33
    assert {int(k) // DAY_MS for k in base[0].day_path_sha256} == {0, 1}
    with pytest.raises(ValueError, match="coverage"):
        evaluate_make_take_policy_forward(
            **(
                arguments
                | {
                    "evaluation": MakeTakeRoleInputs(
                        tuple(range(6)), values, base, stress
                    )
                }
            )
        )
    with pytest.raises(ValueError, match="separation"):
        evaluate_make_take_policy_forward(**(arguments | {"evaluation": six_days(0)}))


def test_complete_forward_roles_pass_without_profitability_qualification():
    _, arguments = inputs()
    result = evaluate_make_take_policy_forward(**arguments)
    assert result.evaluation.economic_gate_passed
    assert result.evaluation.base_metrics.closed_trades == 99
    assert (
        result.calibration_last_recorded_label_ms < result.first_evaluation_decision_ms
    )
    assert not result.qualified_edge
    assert result.evidence()["result_sha256"]
    json.dumps(result.evidence(), allow_nan=False)


@pytest.mark.parametrize(
    "days",
    [
        (2, 3, 4, 5, 6, 6),
        (True, 3, 4, 5, 6, 7),
        (2.1, 3, 4, 5, 6, 7),
        (),
        (-1, 0, 1, 2, 3, 4),
    ],
)
def test_invalid_or_nonconsecutive_days_reject(days):
    _, a = inputs()
    with pytest.raises(ValueError, match="integer role days"):
        evaluate_make_take_policy_forward(
            **(a | {"evaluation": replace(a["evaluation"], days=days)})
        )


def test_full_calibration_target_binding_rejects_replacement():
    _, a = inputs()
    c = a["calibration"]
    x = c.base_targets[0]
    y = replace(x, source_entry_sha256="a" * 64, target_sha256="")
    y = replace(
        y, target_sha256=target_module._sha256(target_module._target_payload(y))
    )
    with pytest.raises(ValueError, match="source binding"):
        evaluate_make_take_policy_forward(
            **(a | {"calibration": replace(c, base_targets=(y, *c.base_targets[1:]))})
        )


@pytest.mark.parametrize("offset,passes", [(0, False), (1, False), (-1, True)])
def test_all_calibration_labels_not_only_selected_orders_respect_boundary(
    offset, passes
):
    h, a = inputs()
    c = a["calibration"]
    x = c.base_targets[0]
    end = x.terminal_time_ms.copy()
    end[-1] = int(a["evaluation"].action_values[0].decision_time_ms.min()) + offset
    y = replace(x, terminal_time_ms=end, target_sha256="")
    y = replace(
        y, target_sha256=target_module._sha256(target_module._target_payload(y))
    )
    c = replace(c, base_targets=(y, *c.base_targets[1:]))
    selection = h.calibrate_make_take_policy(
        predictive_evaluation=h._report_with_valid_hash(),
        action_values=c.action_values,
        base_targets=c.base_targets,
        stress_targets=c.stress_targets,
        expected_days=c.days,
    )
    assert selection.accepted
    a |= {"calibration": c, "policy_selection": selection}
    if passes:
        assert evaluate_make_take_policy_forward(**a).evaluation.economic_gate_passed
    else:
        with pytest.raises(ValueError, match="label reaches"):
            evaluate_make_take_policy_forward(**a)


def test_out_of_role_decisions_reject_even_if_unselected():
    _, a = inputs()
    e = a["evaluation"]
    x = e.action_values[0]
    times = x.decision_time_ms.copy()
    times[:4] = DAY_MS + 1
    y = replace(x, decision_time_ms=times, batch_sha256="")
    y = replace(y, batch_sha256=value_module._sha256(value_module._batch_payload(y)))
    with pytest.raises(ValueError, match="outside declared"):
        evaluate_make_take_policy_forward(
            **(a | {"evaluation": replace(e, action_values=(y, *e.action_values[1:]))})
        )
