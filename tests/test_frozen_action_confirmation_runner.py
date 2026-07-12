from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.run_frozen_action_confirmation as runner
from tools.run_gross_architecture_screen import _canonical_sha256


def test_committed_round31_design_is_historically_hash_bound() -> None:
    path = (
        runner.ROOT
        / "docs"
        / "model-research"
        / "action-value"
        / "round-031-frozen-chronological-confirmation-design.json"
    )

    design, claimed = runner.load_frozen_confirmation_design(
        path, require_current=False
    )

    assert claimed == "1d6e8791f635be6d8d98b9f957ffffefbc211692d4f7adf07fdffb8fea667c0e"
    assert design["implementation"]["commit"] == (
        "47133de1186e3a0e54f9dfdce10fe8500bae700c"
    )
    assert design["reserved_terminal"]["date"] == "2024-03-30"


def test_committed_round31_design_refuses_replay_after_code_evolves() -> None:
    path = (
        runner.ROOT
        / "docs"
        / "model-research"
        / "action-value"
        / "round-031-frozen-chronological-confirmation-design.json"
    )

    with pytest.raises(ValueError, match="implementation"):
        runner.load_frozen_confirmation_design(path)


def _design() -> dict[str, object]:
    sealed = json.loads(runner._ROUND30_DESIGN.read_text(encoding="utf-8"))
    design: dict[str, object] = {
        "schema_version": runner.DESIGN_SCHEMA_VERSION,
        "round": 31,
        "design_revision": 1,
        "purpose": "frozen_round30_chronological_confirmation",
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
        "implementation": {
            "hash_mode": "git_blob_sha256_v1",
            "commit": "a" * 40,
            "files": [{"path": "example.py", "sha256": "b" * 64}],
        },
        "source_model": {
            "round": 30,
            "model_schema_version": "adaptive-lightgbm-hurdle-action-value-v1",
            "report_file_sha256": "a" * 64,
            "report_canonical_sha256": "b" * 64,
            "design_sha256": "c" * 64,
            "barrier_targets_sha256": "d" * 64,
            "source_manifest_fingerprint": "e" * 64,
            "corpus_certificate_sha256": "f" * 64,
            "target_contract_sha256": "1" * 64,
            "models": [
                {
                    "seed": seed,
                    "path": f"models/seed-{seed}.json",
                    "artifact_sha256": str(index) * 64,
                    "model_sha256": str(index + 3) * 64,
                }
                for index, seed in enumerate((29, 43, 71), start=1)
            ],
        },
        "availability": {
            "plan_file_sha256": "9" * 64,
            "truth_basis": "official_binance_data_vision_s3_listing",
            "coverage": [],
            "inventory_identities": [],
        },
        "governance": {
            "consumed_period_registry_sha256": runner._sha256_file(
                runner._CONSUMED_PERIODS
            ),
            "consumed_period_registry_canonical_sha256": json.loads(
                runner._CONSUMED_PERIODS.read_text(encoding="utf-8")
            )["registry_sha256"],
        },
        "feature_version": "l1-tape-causal-v8",
        "data": {
            "symbol": "BTCUSDT",
            "provider": "binance",
            "market_type": "futures",
            "required_data_types": ["bookTicker", "trades"],
            "full_history_inventory_required": True,
            "excluded_target_dates": [
                {
                    "date": "2024-02-05",
                    "reason": "consumed Round 8 date; context only",
                    "target_access_permitted": False,
                },
                {
                    "date": "2024-03-15",
                    "reason": "consumed Round 7 date; excluded from development",
                    "target_access_permitted": False,
                },
            ],
            "start_date": "2023-12-31",
            "end_date": "2024-03-29",
            "stages": {
                "confirmation": {
                    "context_start": "2023-12-31",
                    "evaluation_start": "2024-01-01",
                    "evaluation_end": "2024-02-04",
                    "next_unopened_date": "2024-02-05",
                },
                "policy": {
                    "context_start": "2024-02-05",
                    "evaluation_start": "2024-02-06",
                    "evaluation_end": "2024-03-05",
                    "next_unopened_date": "2024-03-06",
                },
                "development": {
                    "context_start": "2024-03-05",
                    "evaluation_start": "2024-03-06",
                    "evaluation_end": "2024-03-29",
                    "next_unopened_date": "2024-03-30",
                },
            },
        },
        "frozen_thresholds": {
            profile: [
                {"quantile": value, "threshold_bps": 1.0 + index}
                for index, value in enumerate((0.5, 0.7, 0.85, 0.95))
            ]
            for profile in ("conservative", "regular", "aggressive")
        },
        "runtime_resources": {
            "duckdb_memory_limit": "8GB",
            "warehouse_threads": 4,
            "compute_backend": "directml",
            "prediction_batch_size": 65_536,
            "cpu_fallback_permitted": False,
        },
        "reserved_terminal": {
            "date": "2024-03-30",
            "included_in_dataset": False,
            "access_permitted": False,
        },
        "research_basis": [
            {"title": f"Source {index}", "url": f"https://example.com/{index}"}
            for index in range(3)
        ],
    }
    for section in runner._SHARED_SECTIONS:
        design[section] = sealed[section]
    design["design_sha256"] = _canonical_sha256(design)
    return design


def test_design_loader_seals_round30_controls_and_nested_dates(tmp_path: Path) -> None:
    path = tmp_path / "design.json"
    path.write_text(json.dumps(_design()), encoding="utf-8")

    loaded, claimed = runner.load_frozen_confirmation_design(
        path, require_current=False
    )

    assert claimed == loaded["design_sha256"]
    assert loaded["data"]["stages"]["development"]["evaluation_end"] == (
        "2024-03-29"
    )
    assert loaded["reserved_terminal"]["date"] == "2024-03-30"

    loaded["execution"]["horizon_seconds"] = 300
    loaded["design_sha256"] = _canonical_sha256(
        {key: value for key, value in loaded.items() if key != "design_sha256"}
    )
    path.write_text(json.dumps(loaded), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from sealed Round 30"):
        runner.load_frozen_confirmation_design(path, require_current=False)


def test_design_loader_rejects_consumed_date_in_development(tmp_path: Path) -> None:
    design = _design()
    design["data"]["excluded_target_dates"] = design["data"][
        "excluded_target_dates"
    ][:1]
    design["design_sha256"] = _canonical_sha256(
        {key: value for key, value in design.items() if key != "design_sha256"}
    )
    path = tmp_path / "design.json"
    path.write_text(json.dumps(design), encoding="utf-8")

    with pytest.raises(ValueError, match="consumed dates are not fully excluded"):
        runner.load_frozen_confirmation_design(path, require_current=False)


def _source_report(design: dict[str, object]) -> dict[str, object]:
    return {
        "profile_results": [
            {
                "profile": profile,
                "threshold_selection": {
                    "candidates": json.loads(
                        json.dumps(design["frozen_thresholds"][profile])
                    )
                },
            }
            for profile in ("conservative", "regular", "aggressive")
        ]
    }


@pytest.mark.parametrize(
    ("survivors_by_stage", "expected_stages"),
    [
        ({"confirmation": {}}, ["confirmation"]),
        (
            {
                "confirmation": {"regular": (0.7, 2.0)},
                "policy": {},
            },
            ["confirmation", "policy"],
        ),
        (
            {
                "confirmation": {"regular": (0.7, 2.0)},
                "policy": {"regular": (0.7, 2.0)},
                "development": {"regular": (0.7, 2.0)},
            },
            ["confirmation", "policy", "development"],
        ),
    ],
)
def test_runner_never_opens_a_later_stage_before_prior_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    survivors_by_stage: dict[str, dict[str, tuple[float, float]]],
    expected_stages: list[str],
) -> None:
    design = _design()
    plan = tmp_path / "plan.json"
    plan.write_text("{}", encoding="utf-8")
    opened: list[str] = []

    monkeypatch.setattr(
        runner,
        "load_frozen_confirmation_design",
        lambda _path: (design, str(design["design_sha256"])),
    )
    monkeypatch.setattr(
        runner,
        "resolve_backend",
        lambda _requested: SimpleNamespace(
            kind="directml", device="privateuseone:0", vendor="amd"
        ),
    )
    monkeypatch.setattr(
        runner,
        "_validate_availability_plan",
        lambda *_args: {
            "truth_basis": "official_binance_data_vision_s3_listing",
            "coverage": [],
        },
    )
    monkeypatch.setattr(
        runner,
        "_load_source_models",
        lambda *_args: ([], {"report": {}, "models": []}, _source_report(design)),
    )
    monkeypatch.setattr(
        runner,
        "_ensure_causal_feature_bars",
        lambda **_kwargs: {"verified": True, "materialization_state": "reused"},
    )
    monkeypatch.setattr(
        runner,
        "_load_stage_dataset",
        lambda **kwargs: opened.append(kwargs["stage_name"])
        or {"evidence": {"stage": kwargs["stage_name"]}},
    )

    def stage_report(**kwargs):
        name = kwargs["stage_name"]
        survivors = survivors_by_stage[name]
        return (
            {
                "stage": name,
                "surviving_profiles": list(survivors),
                "trading_authority": False,
            },
            survivors,
        )

    monkeypatch.setattr(runner, "_stage_report", stage_report)

    report = runner.run_frozen_action_confirmation(
        design_path=tmp_path / "design.json",
        availability_plan_path=plan,
        source_model_root=tmp_path / "models",
        warehouse_path=tmp_path / "warehouse.duckdb",
        cache_root=tmp_path / "cache",
        output_dir=tmp_path / "output",
    )

    assert opened == expected_stages
    assert report["stage_access"] == {
        "confirmation": "confirmation" in expected_stages,
        "policy": "policy" in expected_stages,
        "development": "development" in expected_stages,
    }
    assert report["terminal_holdout_accessed"] is False
    assert report["trading_authority"] is False


def test_frozen_thresholds_must_match_round30_report_exactly() -> None:
    design = _design()
    report = _source_report(design)

    runner._validate_frozen_thresholds_against_source(design, report)

    report["profile_results"][0]["threshold_selection"]["candidates"][0][
        "threshold_bps"
    ] = 999.0
    with pytest.raises(ValueError, match="differ from Round 30"):
        runner._validate_frozen_thresholds_against_source(design, report)
