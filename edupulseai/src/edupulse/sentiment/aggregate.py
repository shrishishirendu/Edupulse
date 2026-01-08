"""Sentiment aggregation utilities."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from edupulse.sentiment.scoring import sentiment_label, sentiment_score


def load_feedback_csv(path: Path | str, id_col: str, text_col: str, time_col: str) -> pd.DataFrame:
    """Load feedback CSV, parse timestamps, and drop empty text rows."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    if text_col not in df.columns:
        raise ValueError(f"text column '{text_col}' not found")
    if id_col not in df.columns:
        raise ValueError(f"id column '{id_col}' not found")
    if time_col not in df.columns:
        raise ValueError(f"time column '{time_col}' not found")

    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    return df


def add_sentiment(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Add sentiment score and label columns."""
    text_col = cfg["sentiment"]["text_column"]
    df = df.copy()
    df["sentiment_score"] = df[text_col].apply(sentiment_score)
    df["sentiment_label"] = df["sentiment_score"].apply(lambda s: sentiment_label(s, cfg))
    return df


def student_latest(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Return the latest feedback per student."""
    id_col = cfg["sentiment"]["id_column"]
    time_col = cfg["sentiment"]["time_column"]
    if df.empty:
        return df

    df_sorted = df.sort_values(by=[id_col, time_col])
    latest = df_sorted.groupby(id_col).tail(1)
    return latest.reset_index(drop=True)


def _rolling_for_student(group: pd.DataFrame, time_col: str, window_days: int) -> pd.DataFrame:
    group = group.sort_values(time_col)
    scores = group["sentiment_score"]
    times = group[time_col]
    rolling_vals: List[float] = []
    deltas: List[float] = []

    for idx, current_time in enumerate(times):
        window_start = current_time - timedelta(days=window_days)
        window_mask = (times >= window_start) & (times <= current_time)
        window_scores = scores[window_mask]
        current_mean = window_scores.mean() if not window_scores.empty else 0.0
        rolling_vals.append(current_mean)

        prev_mask = (times >= window_start - timedelta(days=window_days)) & (times < window_start)
        prev_scores = scores[prev_mask]
        prev_mean = prev_scores.mean() if not prev_scores.empty else None

        if prev_mean is None:
            deltas.append(0.0)
        else:
            deltas.append(current_mean - prev_mean)

    group = group.copy()
    group["rolling_score"] = rolling_vals
    group["rolling_delta"] = deltas
    return group


def student_rolling(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute rolling mean sentiment per student with deltas."""
    id_col = cfg["sentiment"]["id_column"]
    time_col = cfg["sentiment"]["time_column"]
    window_days = cfg["sentiment"].get("rolling_window_days", 7)

    if df.empty:
        return df

    rolled = df.groupby(id_col, group_keys=False).apply(_rolling_for_student, time_col=time_col, window_days=window_days)
    return rolled.reset_index(drop=True)


def detect_alerts(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Detect negative streaks and sudden drops per student."""
    id_col = cfg["sentiment"]["id_column"]
    time_col = cfg["sentiment"]["time_column"]
    alert_cfg = cfg["sentiment"].get("alert", {})

    negative_streak_count = alert_cfg.get("negative_streak_count", 3)
    recent_days = alert_cfg.get("recent_days", 14)
    sudden_drop_threshold = alert_cfg.get("sudden_drop_threshold", -0.3)

    alerts = []
    for student_id, group in df.groupby(id_col):
        group = group.sort_values(time_col)
        latest_time = group[time_col].max()
        recent_cutoff = latest_time - timedelta(days=recent_days)
        recent_group = group[group[time_col] >= recent_cutoff]

        neg_labels = (recent_group["sentiment_label"] == "negative").sum()
        negative_streak = neg_labels >= negative_streak_count

        recent_delta = recent_group["rolling_delta"].iloc[-1] if not recent_group.empty else 0.0
        sudden_drop = recent_delta <= sudden_drop_threshold

        alerts.append(
            {
                id_col: student_id,
                "negative_streak": bool(negative_streak),
                "sudden_drop": bool(sudden_drop),
                "latest_sentiment_score": group["sentiment_score"].iloc[-1],
                "latest_sentiment_label": group["sentiment_label"].iloc[-1],
                "latest_rolling_score": group["rolling_score"].iloc[-1] if "rolling_score" in group else None,
            }
        )

    return pd.DataFrame(alerts)
