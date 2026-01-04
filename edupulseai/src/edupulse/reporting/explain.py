"""Explainability utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass
class ExplanationBuilder:
    """Aggregate explanation payloads for reporting."""

    def summarize(self, explanations: Iterable[Mapping[str, float]]) -> Mapping[str, float]:
        """Return a simple aggregate importance summary."""
        summary: dict[str, float] = {}
        for expl in explanations:
            for feature, importance in expl.items():
                summary[feature] = summary.get(feature, 0.0) + importance
        return summary