"""Agent tool utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from edupulse.agents.types import StudentContext

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def safe_format_list(text: str) -> str:
    if not text:
        return ""
    parts = [part.strip() for part in str(text).split(";") if part.strip()]
    return ", ".join(parts)

def normalize_id(x: Any) -> str:
    s = str(x).strip()
    if s.isdigit():
        s2 = s.lstrip("0")
        return s2 if s2 != "" else "0"
    return s


def _resolve_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _resolve_combined_path(cfg: Dict[str, Any]) -> Path:
    agents_cfg = cfg.get("agents", {}) or {}
    inputs_cfg = agents_cfg.get("inputs", {}) or {}
    fusion_cfg = cfg.get("fusion") or {}
    candidate = inputs_cfg.get("combined_queue_path") or fusion_cfg.get("combined_queue_path") or Path(
        "reports"
    ) / "combined_risk_test.csv"
    return _resolve_path(candidate)


def _resolve_sentiment_roll_path(cfg: Dict[str, Any]) -> Path:
    agents_cfg = cfg.get("agents", {}) or {}
    inputs_cfg = agents_cfg.get("inputs", {}) or {}
    fusion_cfg = cfg.get("fusion") or {}
    candidate = (
        inputs_cfg.get("sentiment_rolling_path")
        or fusion_cfg.get("sentiment_rolling_path")
        or Path("reports") / "sentiment_student_rolling.csv"
    )
    return _resolve_path(candidate)


def load_student_context(student_id: str | int, cfg: Dict[str, Any]) -> StudentContext:
    combined_path = _resolve_combined_path(cfg)

    if not combined_path.exists():
        raise FileNotFoundError(f"Combined risk file not found: {combined_path}")

    df = pd.read_csv(combined_path)
    id_col = (
        (cfg.get("fusion") or {}).get("student_id_column")
        or (cfg.get("dataset") or {}).get("id_column")
        or "student_id"
    )
    if id_col not in df.columns:
        raise ValueError(f"id column '{id_col}' not found in {combined_path}")
    df[id_col] = df[id_col].map(normalize_id)
    sid = normalize_id(student_id)

    match = df[df[id_col] == sid]
    if match.empty:
        raise ValueError(f"student_id {student_id} not found in {combined_path}")
    row = match.iloc[0]

    roll_path = _resolve_sentiment_roll_path(cfg)
    sentiment_roll_score = None
    sentiment_roll_delta = None
    if roll_path.exists():
        roll_df = pd.read_csv(roll_path)
        if id_col in roll_df.columns:
            roll_df[id_col] = roll_df[id_col].map(normalize_id)
            roll_match = roll_df[roll_df[id_col] == sid]
        else:
            roll_match = roll_df.iloc[0:0]
        if not roll_match.empty:
            last = roll_match.iloc[-1]
            sentiment_roll_score = last.get("rolling_score")
            sentiment_roll_delta = last.get("rolling_delta")

    return StudentContext(
        student_id=sid,
        risk_band=safe_text(row.get("risk_band")),
        dropout_risk=float(row.get("dropout_risk", 0.0) or 0.0),
        urgency=safe_text(row.get("urgency")),
        combined_priority=float(row.get("combined_priority", 0.0) or 0.0),
        top_reasons=safe_text(row.get("top_reasons")),
        recommended_actions=safe_text(row.get("recommended_actions")),
        owner=safe_text(row.get("owner")),
        sentiment_score=row.get("sentiment_score"),
        sentiment_label=safe_text(row.get("sentiment_label")),
        negative_streak=bool(row.get("negative_streak")),
        sudden_drop=bool(row.get("sudden_drop")),
        sentiment_rolling_score=sentiment_roll_score,
        sentiment_rolling_delta=sentiment_roll_delta,
    )
