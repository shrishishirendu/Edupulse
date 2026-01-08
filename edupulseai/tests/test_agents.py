"""Tests for agent orchestrator pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from edupulse.agents.tools import load_student_context
from edupulse.agents.analyst import AnalystAgent
from edupulse.agents.orchestrator import InterventionOrchestrator
from edupulse.agents.planner import PlannerAgent
from edupulse.agents.writer import WriterAgent
from edupulse.agents import tools


def test_load_student_context(tmp_path: Path) -> None:
    combined = tmp_path / "combined_risk_test.csv"
    pd.DataFrame(
        {
            "student_id": [1],
            "risk_band": ["high"],
            "dropout_risk": [0.8],
            "urgency": ["high"],
            "combined_priority": [1],
            "top_reasons": ["attendance; gpa"],
            "recommended_actions": ["Call student; Notify instructor"],
            "owner": ["Coach"],
            "sentiment_score": [-0.5],
            "sentiment_label": ["negative"],
            "negative_streak": [True],
            "sudden_drop": [False],
        }
    ).to_csv(combined, index=False)

    cfg = {"fusion": {"combined_queue_path": combined}}
    ctx = load_student_context(1, cfg)

    assert ctx.student_id == "1"
    assert ctx.risk_band == "high"
    assert ctx.negative_streak is True


def test_load_student_context_string_id(tmp_path: Path) -> None:
    combined = tmp_path / "combined_risk_test.csv"
    pd.DataFrame(
        {
            "student_id": ["001"],
            "risk_band": ["medium"],
            "dropout_risk": [0.5],
            "urgency": ["medium"],
            "combined_priority": [3],
            "top_reasons": ["gpa"],
            "recommended_actions": ["Email student"],
            "owner": ["Advisor"],
            "sentiment_score": [0.1],
            "sentiment_label": ["neutral"],
            "negative_streak": [False],
            "sudden_drop": [False],
        }
    ).to_csv(combined, index=False)

    cfg = {"fusion": {"combined_queue_path": combined}}
    ctx = load_student_context("001", cfg)

    assert ctx.student_id == "1"
    assert ctx.risk_band == "medium"


def test_project_root_resolution() -> None:
    root = tools.PROJECT_ROOT
    assert (root / "configs").exists()


def test_orchestrator_outputs(tmp_path: Path) -> None:
    combined = tmp_path / "combined_risk_test.csv"
    pd.DataFrame(
        {
            "student_id": [1],
            "risk_band": ["high"],
            "dropout_risk": [0.8],
            "urgency": ["high"],
            "combined_priority": [1],
            "top_reasons": ["attendance; gpa"],
            "recommended_actions": ["Call student; Notify instructor"],
            "owner": ["Coach"],
            "sentiment_score": [-0.5],
            "sentiment_label": ["negative"],
            "negative_streak": [True],
            "sudden_drop": [False],
        }
    ).to_csv(combined, index=False)

    cfg = {"fusion": {"combined_queue_path": combined}}
    orchestrator = InterventionOrchestrator(PlannerAgent(), AnalystAgent(), WriterAgent())
    result = orchestrator.run_intervention(1, cfg)

    assert "context" in result
    assert "plan" in result
    assert "student_message" in result
    assert "staff_note" in result
    assert result["context"]["student_id"] == "1"
