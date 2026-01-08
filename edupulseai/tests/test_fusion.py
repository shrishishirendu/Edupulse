"""Tests for risk signal fusion."""

from __future__ import annotations

import pandas as pd

from scripts.fuse_risk_signals import compute_combined_priority, compute_combined_score, fuse_frames


def _cfg() -> dict:
    return {
        "fusion": {
            "student_id_column": "student_id",
            "weights": {
                "risk_band": {"high": 3, "medium": 2, "low": 1},
                "urgency": {"high": 2, "medium": 1, "low": 0},
                "sentiment_label": {"negative": 1, "neutral": 0, "positive": -0.5},
                "alert": {"negative_streak": 1.5, "sudden_drop": 1},
                "sentiment_score_multiplier": 0.5,
            },
            "priority": {"enable_escalation": True, "bump_priority_by": 1, "max_bumps": 2},
            "output": {"include_columns": []},
        },
        "sentiment": {"thresholds": {"negative": -0.2}},
    }


def test_merge_and_columns() -> None:
    cfg = _cfg()
    join_key = "student_id"
    dropout_df = pd.DataFrame(
        {
            "student_id": [1, 2],
            "risk_band": ["high", "medium"],
            "urgency": ["high", "medium"],
            "priority": [3, 4],
            "dropout_risk": [0.9, 0.4],
        }
    )
    latest_df = pd.DataFrame(
        {"student_id": [1, 2], "latest_sentiment_score": [-0.5, 0.1], "latest_sentiment_label": ["negative", "neutral"]}
    )
    alerts_df = pd.DataFrame({"student_id": [1, 2], "negative_streak": [True, False], "sudden_drop": [False, True]})

    fused = fuse_frames(dropout_df, latest_df, alerts_df, cfg, join_key, missing_key_warning=[])

    assert "combined_priority" in fused.columns
    assert "combined_score" in fused.columns
    assert {"sentiment_score", "sentiment_label", "student_id"}.issubset(fused.columns)
    assert fused.shape[0] == 2
    assert fused["latest_sentiment_score"].notna().all()
    assert fused["sentiment_score"].equals(fused["latest_sentiment_score"])
    assert fused["queue_rank"].min() == 1


def test_priority_decreases_on_alerts() -> None:
    cfg = _cfg()
    row = pd.Series(
        {
            "combined_priority": 5,
            "risk_band": "medium",
            "urgency": "medium",
            "negative_streak": True,
            "sudden_drop": False,
            "latest_sentiment_label": "negative",
            "latest_sentiment_score": -0.5,
        }
    )
    decreased = compute_combined_priority(row, cfg)
    assert decreased < 5


def test_high_urgency_sets_low_base_priority_when_missing() -> None:
    cfg = _cfg()
    join_key = "student_id"
    dropout_df = pd.DataFrame(
        {"student_id": [1], "risk_band": ["medium"], "urgency": ["high"], "dropout_risk": [0.5]}  # no priority column
    )
    latest_df = pd.DataFrame({"student_id": [1], "latest_sentiment_score": [0.0], "latest_sentiment_label": ["neutral"]})
    alerts_df = pd.DataFrame({"student_id": [1], "negative_streak": [False], "sudden_drop": [False]})

    fused = fuse_frames(dropout_df, latest_df, alerts_df, cfg, join_key, missing_key_warning=[])

    assert fused["combined_priority"].iloc[0] == 1


def test_sorting_prefers_higher_dropout_risk_when_priority_equal() -> None:
    cfg = _cfg()
    join_key = "student_id"
    dropout_df = pd.DataFrame(
        {
            "student_id": [1, 2],
            "risk_band": ["medium", "medium"],
            "urgency": ["medium", "medium"],
            "priority": [5, 5],
            "dropout_risk": [0.9, 0.2],
        }
    )
    latest_df = pd.DataFrame(
        {"student_id": [1, 2], "latest_sentiment_score": [0.0, 0.0], "latest_sentiment_label": ["neutral", "neutral"]}
    )
    alerts_df = pd.DataFrame({"student_id": [1, 2], "negative_streak": [False, False], "sudden_drop": [False, False]})

    fused = fuse_frames(dropout_df, latest_df, alerts_df, cfg, join_key, missing_key_warning=[])
    assert fused.iloc[0]["student_id"] == 1  # higher dropout_risk comes first when priority equal
    assert fused.iloc[0]["dropout_risk"] == 0.9


def test_score_blends_sentiment_and_risk() -> None:
    cfg = _cfg()
    row = pd.Series(
        {
            "risk_band": "high",
            "urgency": "high",
            "latest_sentiment_label": "negative",
            "latest_sentiment_score": -0.6,
            "negative_streak": True,
            "sudden_drop": False,
        }
    )
    score = compute_combined_score(row, cfg)
    assert score > 0
