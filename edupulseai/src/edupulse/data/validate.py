"""Input validation helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def validate_dataset(df: pd.DataFrame, required_columns: Iterable[str]) -> bool:
    """Ensure all required columns exist before analysis."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return True