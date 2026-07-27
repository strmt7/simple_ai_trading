from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404

from simple_ai_trading.impact_absorption_event_action_policy import (
    ROUND74_ACTION_CONTEXT_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_dataset import (
    ROUND74_EVENT_DATASET_SCHEMA_VERSION,
    ROUND74_EVENT_WINDOW_REPRESENTATIONS,
)
from simple_ai_trading.impact_absorption_event_training import (
    ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION,
    ROUND74_EVENT_TRAINING_SCHEMA_VERSION,
)
from simple_ai_trading.round74_event_model_operator import (
    ROUND74_EVENT_MODEL_OPERATOR_SCHEMA_VERSION,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    REPOSITORY
    / "docs"
    / "model-research"
    / "action-value"
    / "round-074-event-sequence-model-design-v73.json"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _normalized_file_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _git_file_sha256(commit: str, relative_path: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    payload = completed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def test_round74_v73_binds_representation_through_sealed_inference() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    assert claimed == _canonical_sha256(design)
    assert design["schema_version"] == "round-074-event-sequence-model-design-v73"
    base = design["base_design"]
    assert base["normalized_lf_sha256"] == _normalized_file_sha256(
        REPOSITORY / base["path"]
    )

    source = design["source_binding"]
    for prefix in (
        "event_sequence",
        "event_dataset",
        "model_operator",
        "event_training",
        "action_policy",
        "sealed_evaluation",
        "development_operator",
        "online_latency",
    ):
        assert source[f"{prefix}_normalized_lf_sha256"] == _git_file_sha256(
            design["implementation_git_commit"],
            source[f"{prefix}_path"],
        )

    assert ROUND74_EVENT_DATASET_SCHEMA_VERSION == "round-074-event-dataset-v10"
    assert ROUND74_EVENT_MODEL_OPERATOR_SCHEMA_VERSION == (
        "round-074-event-model-operator-v4"
    )
    assert ROUND74_EVENT_TRAINING_SCHEMA_VERSION == "round-074-event-training-v15"
    assert ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION == (
        "round-074-event-pretest-policy-v14"
    )
    assert ROUND74_ACTION_CONTEXT_SCHEMA_VERSION == "round-074-action-context-v5"
    contract = design["representation_contract"]
    assert tuple(contract["supported_representations"]) == (
        ROUND74_EVENT_WINDOW_REPRESENTATIONS
    )
    assert contract["mixed_training_or_tuning_representations_rejected"] is True
    assert contract["sealed_inference_representation_mismatch_rejected"] is True
    assert design["verification"]["prospective_representation_comparison_run"] is False
    assert all(value is False for value in design["authority"].values())
