"""Tests for sentiment scoring and aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from edupulse.sentiment.aggregate import add_sentiment, detect_alerts, load_feedback_csv, student_latest, student_rolling
from edupulse.sentiment.scoring import sentiment_label, sentiment_score
from scripts.run_sentiment import ensure_demo_data


def _cfg(tmp_path: Path) -> dict:
    return {
        "sentiment": {
            "raw_path": tmp_path / "feedback.csv",
            "id_column": "student_id",
            "text_column": "text",
            "time_column": "timestamp",
            "thresholds": {"negative": -0.2, "positive": 0.2},
            "rolling_window_days": 3,
            "alert": {"negative_streak_count": 2, "recent_days": 7, "sudden_drop_threshold": -0.3},
        }
    }


def test_sentiment_score_ordering() -> None:
    pos = sentiment_score("I love this awesome course")
    neu = sentiment_score("This course is fine")
    neg = sentiment_score("I hate this terrible course")

    assert pos > neu > neg


def test_aggregation_pipeline(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    data = pd.DataFrame(
        {
            "student_id": [1, 1, 2],
            "text": ["Great job", "Not good", "Very helpful and supportive"],
            "timestamp": ["2026-01-01T10:00:00", "2026-01-02T10:00:00", "2026-01-01T12:00:00"],
        }
    )
    path = cfg["sentiment"]["raw_path"]
    data.to_csv(path, index=False)

    df = load_feedback_csv(path, "student_id", "text", "timestamp")
    df = add_sentiment(df, cfg)
    df = student_rolling(df, cfg)
    latest = student_latest(df, cfg)
    alerts = detect_alerts(df, cfg)

    assert {"sentiment_score", "sentiment_label"}.issubset(df.columns)
    assert {"rolling_score", "rolling_delta"}.issubset(df.columns)
    assert not latest.empty
    assert {"negative_streak", "sudden_drop"}.issubset(alerts.columns)


def test_demo_mode_outputs(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    raw_path = cfg["sentiment"]["raw_path"]
    ensure_demo_data(Path(raw_path), cfg)

    df = load_feedback_csv(raw_path, "student_id", "text", "timestamp")
    df = add_sentiment(df, cfg)
    df = student_rolling(df, cfg)
    alerts = detect_alerts(df, cfg)

    assert not df.empty
    assert not alerts.empty
