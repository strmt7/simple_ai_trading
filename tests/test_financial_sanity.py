from __future__ import annotations

from simple_ai_trading.financial_sanity import (
    build_model_financial_sanity_report,
    build_model_lab_financial_sanity_report,
)
from simple_ai_trading.model import TrainedModel


def test_model_financial_sanity_blocks_malformed_parameters() -> None:
    model = TrainedModel(
        weights=[0.0, float("inf")],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        learning_rate=2.0,
        probability_temperature=0.0,
    )

    report = build_model_financial_sanity_report(model)

    assert report.allowed is False
    assert report.block_count >= 3
    assert any(check.path == "weights" and check.status == "block" for check in report.checks)
    assert any(check.path == "learning_rate" and check.status == "block" for check in report.checks)


def test_model_lab_financial_sanity_blocks_impossible_accepted_report() -> None:
    payload = {
        "accepted_symbols": ["AAAUSDC"],
        "portfolio_risk": {
            "accepted": True,
            "portfolio_cvar_95": 0.01,
            "portfolio_max_drawdown": 0.02,
            "deployed_weight": 0.4,
            "max_pairwise_correlation": 0.1,
            "max_cluster_weight": 0.4,
        },
        "outcomes": [
            {
                "symbol": "AAAUSDC",
                "accepted": True,
                "rows": 0,
                "objective_scores": {"regular": 0.15},
                "data_coverage": {
                    "integrity_status": "ok",
                    "coverage_ratio": 1.0,
                    "gap_count": 0,
                },
                "stress_validation": {"accepted": True, "worst_max_drawdown": 0.01},
                "robustness_validation": {"accepted": True, "worst_max_drawdown": 0.02},
            }
        ],
    }

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any("rows=0" in check.detail for check in report.checks if check.status == "block")


def test_model_lab_financial_sanity_blocks_failed_market_edge_evidence() -> None:
    payload = {
        "accepted_symbols": ["AAAUSDC"],
        "portfolio_risk": {
            "accepted": True,
            "portfolio_cvar_95": 0.01,
            "portfolio_max_drawdown": 0.02,
            "deployed_weight": 0.4,
            "max_pairwise_correlation": 0.1,
            "max_cluster_weight": 0.4,
        },
        "outcomes": [
            {
                "symbol": "AAAUSDC",
                "accepted": True,
                "rows": 500,
                "objective_scores": {"regular": 0.15},
                "data_coverage": {
                    "integrity_status": "ok",
                    "coverage_ratio": 1.0,
                    "gap_count": 0,
                },
                "stress_validation": {
                    "accepted": True,
                    "worst_max_drawdown": 0.01,
                    "objectives": [
                        {
                            "objective": "regular",
                            "accepted": True,
                            "results": [
                                {
                                    "result": {
                                        "market_edge": {
                                            "accepted": False,
                                            "reason": "net_edge_pct<0.003000",
                                            "net_edge_pct": 0.001,
                                            "min_net_edge_pct": 0.003,
                                        }
                                    }
                                }
                            ],
                        }
                    ],
                },
                "robustness_validation": {
                    "accepted": True,
                    "worst_max_drawdown": 0.02,
                    "statistical_edge_accepted": True,
                },
            }
        ],
    }

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(check.label == "market edge" and check.status == "block" for check in report.checks)


def test_model_lab_financial_sanity_blocks_accepted_market_edge_with_bad_downside_risk() -> None:
    payload = {
        "accepted_symbols": ["BTCUSDT"],
        "portfolio_risk": {
            "accepted": True,
            "portfolio_cvar_95": 0.01,
            "portfolio_max_drawdown": 0.02,
            "deployed_weight": 0.4,
            "max_pairwise_correlation": 0.1,
            "max_cluster_weight": 0.4,
        },
        "outcomes": [
            {
                "symbol": "BTCUSDT",
                "accepted": True,
                "rows": 500,
                "objective_scores": {"conservative": 0.15},
                "data_coverage": {
                    "integrity_status": "ok",
                    "coverage_ratio": 1.0,
                    "gap_count": 0,
                },
                "stress_validation": {
                    "accepted": True,
                    "worst_max_drawdown": 0.01,
                    "objectives": [
                        {
                            "objective": "conservative",
                            "accepted": True,
                            "results": [
                                {
                                    "result": {
                                        "market_edge": {
                                            "accepted": True,
                                            "reason": None,
                                            "net_edge_pct": 0.01,
                                            "min_net_edge_pct": 0.002,
                                            "downside_return_risk_ratio": 0.10,
                                            "min_downside_return_risk_ratio": 0.45,
                                        }
                                    }
                                }
                            ],
                        }
                    ],
                },
                "robustness_validation": {
                    "accepted": True,
                    "worst_max_drawdown": 0.02,
                    "statistical_edge_accepted": True,
                },
            }
        ],
    }

    report = build_model_lab_financial_sanity_report(payload)

    assert report.allowed is False
    assert any(
        check.label == "market edge downside risk" and check.status == "block"
        for check in report.checks
    )
