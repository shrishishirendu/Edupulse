"""Feature construction utilities."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def build_feature_matrix(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Return a filtered DataFrame representing the feature matrix."""
    return df.loc[:, list(feature_columns)].copy()