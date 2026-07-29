"""Tuning-only final action configuration selected before sealed-test access."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re
from typing import TYPE_CHECKING

from .impact_absorption_event_action_policy import (
    ROUND74_ACTION_PROFILES,
    Round74ActionCandidateBatch,
    Round74ActionPolicySelection,
)
from .impact_absorption_event_epistemic_policy import (
    Round74EpistemicActionFilterApplication,
    Round74EpistemicActionReplayChallenge,
    apply_round74_epistemic_action_filter,
)
from .impact_absorption_event_model import Round74EventModelOutput

if TYPE_CHECKING:
    from .round74_event_development_operator import Round74DevelopmentPolicyBundle


ROUND74_FINAL_ACTION_CONFIGURATION_SCHEMA_VERSION = (
    "round-074-final-action-configuration-v1"
)
ROUND74_FINAL_ACTION_CONFIGURATION_MODES = ("baseline", "epistemic_filter")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SELECTION_CONTRACT = {
    "selection_data_role": "tuning_only",
    "baseline_quality_threshold_held_fixed": True,
    "single_final_ml_configuration": True,
    "sealed_ml_configuration_count": 1,
    "sealed_ai_overlay_configuration_count": 2,
    "sealed_qualification_configuration_count": 3,
    "post_sealed_test_selection_permitted": False,
    "automatic_runtime_or_trading_activation": False,
}
_AUTHORITY = {
    "sealed_test_accessed": False,
    "financial_edge_tested": False,
    "profitability_claim": False,
    "paper_trading": False,
    "testnet_trading": False,
    "mainnet_trading": False,
    "polymarket_trading": False,
    "order_submission": False,
    "position_management": False,
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Round74FinalActionConfiguration:
    """One development-selected ML configuration for later sealed evaluation."""

    development_bundle_sha256: str
    action_selection: Round74ActionPolicySelection
    epistemic_action_challenge: Round74EpistemicActionReplayChallenge | None
    mode: str
    decision_reason_code: str
    schema_version: str = ROUND74_FINAL_ACTION_CONFIGURATION_SCHEMA_VERSION
    sealed_test_accessed: bool = False
    trading_authority: bool = False
    profitability_claim: bool = False

    def validate(self) -> None:
        self.action_selection.validate()
        challenge = self.epistemic_action_challenge
        if challenge is not None:
            challenge.validate()
        expected_mode = (
            "epistemic_filter"
            if challenge is not None and challenge.tuning_challenge_eligible
            else "baseline"
        )
        expected_reason = (
            "tuning_eligible_epistemic_filter"
            if expected_mode == "epistemic_filter"
            else (
                "baseline_epistemic_challenge_not_eligible"
                if challenge is not None
                else "baseline_no_epistemic_challenge"
            )
        )
        if (
            self.schema_version
            != ROUND74_FINAL_ACTION_CONFIGURATION_SCHEMA_VERSION
            or not isinstance(self.development_bundle_sha256, str)
            or _SHA256.fullmatch(self.development_bundle_sha256) is None
            or self.action_selection.profile not in ROUND74_ACTION_PROFILES
            or not self.action_selection.accepted
            or self.action_selection.execution_outcome_panel_sha256 is None
            or not isinstance(self.mode, str)
            or self.mode not in ROUND74_FINAL_ACTION_CONFIGURATION_MODES
            or self.mode != expected_mode
            or not isinstance(self.decision_reason_code, str)
            or self.decision_reason_code != expected_reason
            or self.sealed_test_accessed is not False
            or self.trading_authority is not False
            or self.profitability_claim is not False
        ):
            raise ValueError("Round 74 final action configuration differs")
        if challenge is not None and (
            challenge.profile != self.action_selection.profile
            or challenge.baseline_policy_selection_sha256
            != self.action_selection.selection_sha256
            or challenge.execution_panel_sha256
            != self.action_selection.execution_outcome_panel_sha256
            or challenge.action_filter.tuning_subpartition_sha256
            != self.action_selection.tuning_subpartition_sha256
            or challenge.action_filter.probability_calibration_sha256
            != self.action_selection.probability_calibration_sha256
            or challenge.action_filter.source_batch_sha256
            != self.action_selection.target_batch_sha256
        ):
            raise ValueError("Round 74 final action configuration identity differs")

    @property
    def profile(self) -> str:
        return self.action_selection.profile

    @property
    def action_filter_sha256(self) -> str | None:
        challenge = self.epistemic_action_challenge
        return (
            challenge.action_filter.filter_sha256
            if challenge is not None and self.mode == "epistemic_filter"
            else None
        )

    @property
    def configuration_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "development_bundle_sha256": self.development_bundle_sha256,
            "profile": self.profile,
            "mode": self.mode,
            "decision_reason_code": self.decision_reason_code,
            "action_selection": self.action_selection.as_dict(),
            "epistemic_action_challenge": (
                self.epistemic_action_challenge.as_dict()
                if self.epistemic_action_challenge is not None
                else None
            ),
            "action_filter_sha256": self.action_filter_sha256,
            "selection_contract": dict(_SELECTION_CONTRACT),
            "authority": dict(_AUTHORITY),
            "sealed_test_accessed": False,
            "trading_authority": False,
            "profitability_claim": False,
        }
        if include_sha256:
            value["configuration_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74FinalActionConfiguration:
        unsigned = dict(value)
        claimed = unsigned.pop("configuration_sha256", None)
        if (
            not isinstance(claimed, str)
            or _SHA256.fullmatch(claimed) is None
            or claimed != _canonical_sha256(unsigned)
        ):
            raise ValueError("Round 74 final action configuration digest differs")
        payload = dict(unsigned)
        action_selection = payload.pop("action_selection", None)
        challenge = payload.pop("epistemic_action_challenge", None)
        action_filter_sha256 = payload.pop("action_filter_sha256", None)
        selection_contract = payload.pop("selection_contract", None)
        authority = payload.pop("authority", None)
        expected_keys = {
            "schema_version",
            "development_bundle_sha256",
            "profile",
            "mode",
            "decision_reason_code",
            "sealed_test_accessed",
            "trading_authority",
            "profitability_claim",
        }
        boolean_fields = (
            "sealed_test_accessed",
            "trading_authority",
            "profitability_claim",
        )
        string_fields = (
            "schema_version",
            "development_bundle_sha256",
            "profile",
            "mode",
            "decision_reason_code",
        )
        selection_boolean_fields = (
            "baseline_quality_threshold_held_fixed",
            "single_final_ml_configuration",
            "post_sealed_test_selection_permitted",
            "automatic_runtime_or_trading_activation",
        )
        selection_count_fields = (
            "sealed_ml_configuration_count",
            "sealed_ai_overlay_configuration_count",
            "sealed_qualification_configuration_count",
        )
        if (
            set(payload) != expected_keys
            or not isinstance(action_selection, Mapping)
            or (challenge is not None and not isinstance(challenge, Mapping))
            or not isinstance(selection_contract, Mapping)
            or selection_contract != _SELECTION_CONTRACT
            or any(
                not isinstance(selection_contract[name], bool)
                for name in selection_boolean_fields
            )
            or any(
                isinstance(selection_contract[name], bool)
                or not isinstance(selection_contract[name], int)
                for name in selection_count_fields
            )
            or not isinstance(authority, Mapping)
            or authority != _AUTHORITY
            or any(not isinstance(item, bool) for item in authority.values())
            or any(not isinstance(payload[name], str) for name in string_fields)
            or any(payload[name] is not False for name in boolean_fields)
            or (
                action_filter_sha256 is not None
                and not isinstance(action_filter_sha256, str)
            )
        ):
            raise ValueError("Round 74 final action configuration payload differs")
        selected = cls(
            schema_version=str(payload["schema_version"]),
            development_bundle_sha256=str(payload["development_bundle_sha256"]),
            action_selection=Round74ActionPolicySelection.from_dict(action_selection),
            epistemic_action_challenge=(
                Round74EpistemicActionReplayChallenge.from_dict(challenge)
                if challenge is not None
                else None
            ),
            mode=str(payload["mode"]),
            decision_reason_code=str(payload["decision_reason_code"]),
            sealed_test_accessed=payload["sealed_test_accessed"],
            trading_authority=payload["trading_authority"],
            profitability_claim=payload["profitability_claim"],
        )
        selected.validate()
        if (
            selected.profile != payload["profile"]
            or selected.action_filter_sha256 != action_filter_sha256
            or selected.as_dict() != dict(value)
        ):
            raise ValueError("Round 74 final action configuration contract differs")
        return selected


def _build_round74_final_action_configuration(
    *,
    development_bundle_sha256: str,
    action_selection: Round74ActionPolicySelection,
    epistemic_action_challenge: Round74EpistemicActionReplayChallenge | None,
) -> Round74FinalActionConfiguration:
    """Construct one deterministic pretest configuration from bound inputs."""

    mode = (
        "epistemic_filter"
        if (
            epistemic_action_challenge is not None
            and epistemic_action_challenge.tuning_challenge_eligible
        )
        else "baseline"
    )
    reason = (
        "tuning_eligible_epistemic_filter"
        if mode == "epistemic_filter"
        else (
            "baseline_epistemic_challenge_not_eligible"
            if epistemic_action_challenge is not None
            else "baseline_no_epistemic_challenge"
        )
    )
    selected = Round74FinalActionConfiguration(
        development_bundle_sha256=development_bundle_sha256,
        action_selection=action_selection,
        epistemic_action_challenge=epistemic_action_challenge,
        mode=mode,
        decision_reason_code=reason,
    )
    selected.validate()
    return selected


def select_round74_final_action_configuration(
    bundle: Round74DevelopmentPolicyBundle,
    *,
    profile: str,
) -> Round74FinalActionConfiguration:
    """Select one profile from an exact validated development bundle."""

    from .round74_event_development_operator import Round74DevelopmentPolicyBundle

    if not isinstance(bundle, Round74DevelopmentPolicyBundle):
        raise TypeError("Round 74 final action selection requires a development bundle")
    bundle.validate()
    policies = tuple(
        policy for policy in bundle.action_policies if policy.profile == profile
    )
    challenges = tuple(
        challenge
        for challenge in bundle.epistemic_action_challenges
        if challenge.profile == profile
    )
    if len(policies) != 1 or len(challenges) > 1:
        raise ValueError("Round 74 final action profile binding differs")
    return _build_round74_final_action_configuration(
        development_bundle_sha256=bundle.bundle_sha256,
        action_selection=policies[0],
        epistemic_action_challenge=challenges[0] if challenges else None,
    )


def apply_round74_final_action_configuration(
    candidates: tuple[Round74ActionCandidateBatch, ...],
    model_outputs: tuple[Round74EventModelOutput, ...],
    configuration: Round74FinalActionConfiguration,
) -> tuple[
    tuple[Round74ActionCandidateBatch, ...],
    tuple[Round74EpistemicActionFilterApplication, ...],
]:
    """Apply only the frozen target-free candidate filter, when selected."""

    configuration.validate()
    selected_candidates = tuple(candidates)
    selected_outputs = tuple(model_outputs)
    if (
        not selected_candidates
        or len(selected_candidates) != len(selected_outputs)
        or any(
            not hasattr(candidate, "validate")
            or not hasattr(candidate, "profile")
            or candidate.profile != configuration.profile
            for candidate in selected_candidates
        )
    ):
        raise ValueError("Round 74 final action candidate panel differs")
    if configuration.mode == "baseline":
        return selected_candidates, ()
    challenge = configuration.epistemic_action_challenge
    if challenge is None:
        raise ValueError("Round 74 final action filter is missing")
    filtered: list[Round74ActionCandidateBatch] = []
    applications: list[Round74EpistemicActionFilterApplication] = []
    for candidate, output in zip(
        selected_candidates,
        selected_outputs,
        strict=True,
    ):
        candidate.validate()
        output.validate(candidate.rows)
        filtered_candidate, application = apply_round74_epistemic_action_filter(
            candidate,
            output,
            challenge.action_filter,
        )
        filtered.append(filtered_candidate)
        applications.append(application)
    return tuple(filtered), tuple(applications)


__all__ = [
    "ROUND74_FINAL_ACTION_CONFIGURATION_MODES",
    "ROUND74_FINAL_ACTION_CONFIGURATION_SCHEMA_VERSION",
    "Round74FinalActionConfiguration",
    "apply_round74_final_action_configuration",
    "select_round74_final_action_configuration",
]
