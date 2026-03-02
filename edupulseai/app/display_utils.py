"""Display-only helpers for consistent UI naming and formatting."""

from __future__ import annotations

import pandas as pd


def format_probability_4dp(value: object) -> str:
    """Return probability with exactly 4 decimals, blank when invalid/missing."""
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    return f"{float(numeric):.4f}"


def add_dropout_probability_display(
    df: pd.DataFrame,
    source_col: str = "dropout_risk",
    display_col: str = "Dropout Probability",
) -> pd.DataFrame:
    out = df.copy()
    if source_col in out.columns:
        out[display_col] = pd.to_numeric(out[source_col], errors="coerce").apply(format_probability_4dp)
    else:
        out[display_col] = ""
    return out


def rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "student_id": "Learner ID",
            "risk_band": "Retention Tier",
            "urgency": "Intervention Priority",
            "sentiment_label": "Engagement Signal",
        }
    )


def pick_present_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]
