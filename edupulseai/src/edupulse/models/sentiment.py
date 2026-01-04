"""Sentiment analysis placeholder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class SentimentModel:
    """Generate placeholder sentiment scores."""

    def predict(self, texts: List[str]) -> pd.DataFrame:
        """Return neutral sentiment predictions with confidence."""
        return pd.DataFrame(
            {
                "text": texts,
                "sentiment": ["neutral"] * len(texts),
                "confidence": [0.5] * len(texts),
            }
        )