"""Tests for intervention playbook utilities."""

from __future__ import annotations

import pandas as pd

from edupulse.interventions.playbook import compute_priority, infer_top_reasons, recommend_actions
from scripts.generate_interventions import max_urgency


def test_infer_top_reasons_returns_list() -> None:
    row = pd.Series({"GPA": 2.5, "Program": "A"})
    top_drivers = pd.DataFrame({"feature": ["numeric__GPA", "categorical__Program_A"]})

    reasons = infer_top_reasons(row, top_drivers, max_reasons=2)

    assert isinstance(reasons, list)
    assert reasons


def test_recommend_actions_defaults_when_no_rule_matches() -> None:
    playbook = {
        "default_actions": ["Monitor"],
        "default_owner": "General",
        "default_urgency": "low",
        "rules": [
            {"name": "Attendance", "when_any_features_contain": ["attendance"], "actions": ["Call"], "owner": "Coach", "urgency": "high"}
        ],
    }
    rec = recommend_actions(["unknown"], playbook)

    assert rec["actions"] == ["Monitor"]
    assert rec["owner"] == "General"
    assert rec["urgency"] == "low"


def test_priority_ordering() -> None:
    cfg = {"interventions": {"risk_band_priority": {"high": 1, "medium": 2, "low": 3}}}

    high_high = compute_priority("high", "high", cfg)
    medium_high = compute_priority("medium", "high", cfg)
    low_low = compute_priority("low", "low", cfg)

    assert high_high < medium_high < low_low


def test_high_risk_escalates_urgency() -> None:
    rule_urgency = "medium"
    risk_band = "high"

    risk_floor = {"high": "high", "medium": "medium", "low": "low"}.get(risk_band, rule_urgency)
    final_urgency = max_urgency(rule_urgency, risk_floor)

    assert final_urgency == "high"


def test_priority_respects_urgency_ordering() -> None:
    cfg = {"interventions": {"risk_band_priority": {"high": 1, "medium": 2, "low": 3}}}
    high_high = compute_priority("high", "high", cfg)
    high_medium = compute_priority("high", "medium", cfg)

    assert high_high < high_medium
