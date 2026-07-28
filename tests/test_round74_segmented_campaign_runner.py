from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading.impact_absorption_event_segmented_cohort import (
    ROUND74_SEGMENTED_COHORT_SLOT_PERIOD_NS,
    ROUND74_SEGMENTED_COHORT_START_TOLERANCE_NS,
    Round74SegmentedCohortPlan,
)
from simple_ai_trading.round74_segmented_campaign_runner import (
    ROUND74_SEGMENTED_CAMPAIGN_DATABASE_CAP_BYTES,
    ROUND74_SEGMENTED_CAMPAIGN_MINIMUM_FREE_BYTES,
    ROUND74_SEGMENTED_CAMPAIGN_SLOT_GROWTH_CAP_BYTES,
    Round74SegmentedCampaignRunnerConfig,
    inspect_round74_segmented_campaign_readiness,
    select_round74_segmented_campaign_slot,
)


def _plan() -> Round74SegmentedCohortPlan:
    return Round74SegmentedCohortPlan(
        scheduled_start_wall_ns=2_000_000_000_000_000_000,
        implementation_git_commit="c" * 40,
        prerequisite_artifact_sha256="a" * 64,
        prerequisite_window_start_wall_ns=1_999_900_000_000_000_000,
        prerequisite_window_end_wall_ns=1_999_904_000_000_000_000,
    )


def _config(root: Path) -> Round74SegmentedCampaignRunnerConfig:
    plan_path = root / "plan.json"
    plan_path.write_text(
        json.dumps(_plan().as_dict()),
        encoding="utf-8",
    )
    data = root / "data"
    data.mkdir()
    return Round74SegmentedCampaignRunnerConfig(
        repository=root,
        plan_path=plan_path,
        database_path=data / "cohort.duckdb",
        state_root=data / "cohort-state",
    )


@pytest.mark.parametrize(
    ("offset_ns", "expected_status", "expected_ordinal"),
    (
        (-1, "before_campaign", None),
        (0, "open", 0),
        (
            ROUND74_SEGMENTED_COHORT_START_TOLERANCE_NS,
            "open",
            0,
        ),
        (
            ROUND74_SEGMENTED_COHORT_START_TOLERANCE_NS + 1,
            "between_slots",
            0,
        ),
        (ROUND74_SEGMENTED_COHORT_SLOT_PERIOD_NS, "open", 1),
        (
            720 * ROUND74_SEGMENTED_COHORT_SLOT_PERIOD_NS,
            "after_campaign",
            None,
        ),
    ),
)
def test_segmented_campaign_selection_never_shifts_slots(
    offset_ns: int,
    expected_status: str,
    expected_ordinal: int | None,
) -> None:
    plan = _plan()
    selected = select_round74_segmented_campaign_slot(
        plan,
        now_wall_ns=plan.scheduled_start_wall_ns + offset_ns,
    )

    assert selected.status == expected_status
    assert selected.slot_ordinal == expected_ordinal


def test_segmented_campaign_config_freezes_resource_bounds(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.validate()

    assert config.database_cap_bytes == 64 * 1024**3
    assert config.slot_growth_cap_bytes == 512 * 1024**2
    assert config.minimum_free_bytes == 100 * 1024**3
    assert (
        config.database_cap_bytes
        == ROUND74_SEGMENTED_CAMPAIGN_DATABASE_CAP_BYTES
    )
    assert (
        config.slot_growth_cap_bytes
        == ROUND74_SEGMENTED_CAMPAIGN_SLOT_GROWTH_CAP_BYTES
    )
    assert config.minimum_free_bytes == ROUND74_SEGMENTED_CAMPAIGN_MINIMUM_FREE_BYTES

    with pytest.raises(ValueError, match="config differs"):
        Round74SegmentedCampaignRunnerConfig(
            **{**config.__dict__, "duckdb_threads": 3}
        ).validate()


def test_segmented_campaign_readiness_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    plan = _plan()
    monkeypatch.setattr(
        "simple_ai_trading.round74_segmented_campaign_runner._source_matches",
        lambda *_args: (True, ()),
    )
    monkeypatch.setattr(
        "simple_ai_trading.round74_segmented_campaign_runner."
        "_active_segmented_capture_processes",
        lambda: [],
    )
    monkeypatch.setattr(
        "simple_ai_trading.round74_segmented_campaign_runner.shutil.disk_usage",
        lambda _path: SimpleNamespace(
            free=ROUND74_SEGMENTED_CAMPAIGN_MINIMUM_FREE_BYTES
        ),
    )
    observed = plan.scheduled_start_wall_ns + 1

    ready = inspect_round74_segmented_campaign_readiness(
        config,
        now_wall_ns=observed,
    )
    assert ready["can_start_now"] is True
    assert all(ready["checks"].values())

    reservation = (
        config.state_root / "slot-000" / "reservation.json"
    )
    reservation.parent.mkdir(parents=True)
    reservation.write_text("reserved", encoding="utf-8")
    blocked = inspect_round74_segmented_campaign_readiness(
        config,
        now_wall_ns=observed,
    )
    assert blocked["can_start_now"] is False
    assert blocked["checks"]["slot_not_reserved_passed"] is False

    reservation.unlink()
    Path(f"{config.database_path}.wal").write_bytes(b"wal")
    wal_blocked = inspect_round74_segmented_campaign_readiness(
        config,
        now_wall_ns=observed,
    )
    assert wal_blocked["can_start_now"] is False
    assert wal_blocked["checks"]["wal_absent_before_slot_passed"] is False
