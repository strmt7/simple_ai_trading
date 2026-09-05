import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest


def test_forward_boundary_diagnostic_is_source_bound_and_reconstructs():
    root = Path(__file__).resolve().parents[1]
    artifact = json.loads(
        (root / "docs/review/2026-09-05/make-take-forward-boundary.json").read_bytes()
    )
    for path, expected in artifact["source_sha256"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location(
        "forward_boundary_fixture", root / "tests/test_make_take_forward_evaluation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, arguments = module.inputs()
    result = module.evaluate_make_take_policy_forward(**arguments)
    assert (
        result.evidence()["result_sha256"]
        == artifact["forward_complete_synthetic_case"]["evidence_sha256"]
    )
    assert result.evaluation.base_metrics.closed_trades == 99
    assert artifact["legacy_reproduction"]["actual_target_day_paths"] == [0, 1]
    assert not artifact["accepted_edge"]


def test_forward_wrapper_rejects_valid_but_unaccepted_calibration():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "rejected_forward_fixture", root / "tests/test_make_take_forward_evaluation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    helper, arguments = module.inputs()
    role = arguments["calibration"]
    selection = helper.calibrate_make_take_policy(
        predictive_evaluation=helper._report_with_valid_hash(),
        action_values=role.action_values,
        base_targets=role.base_targets,
        stress_targets=role.stress_targets,
        expected_days=role.days,
        spec=replace(
            arguments["policy_selection"].spec, minimum_calibration_closed_trades=999
        ),
    )
    assert not selection.accepted
    with pytest.raises(ValueError, match="accepted calibration"):
        module.evaluate_make_take_policy_forward(
            **(arguments | {"policy_selection": selection})
        )
