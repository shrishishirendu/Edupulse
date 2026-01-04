"""Agent orchestrator implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.edupulse.models.dropout import DropoutRiskModel
from src.edupulse.models.enrollment import EnrollmentForecaster
from src.edupulse.models.sentiment import SentimentModel
from src.edupulse.reporting.explain import ExplanationBuilder
from src.edupulse.agents.tools import ToolRegistry


@dataclass
class AnalysisOrchestrator:
    """Coordinate analytical modules via tool registry."""

    tools: ToolRegistry

    def run_dropout_pipeline(self, features: Any) -> Dict[str, Any]:
        """Execute dropout analysis and summarise output."""
        model = self.tools.get_dropout_model()
        scores = model.predict(features)
        explanations = model.explain(features)
        summary = ExplanationBuilder().summarize(explanations)
        return {"scores": scores, "explanations": explanations, "summary": summary}

    def run_sentiment_pipeline(self, texts: List[str]) -> Dict[str, Any]:
        """Execute sentiment pipeline."""
        model = self.tools.get_sentiment_model()
        predictions = model.predict(texts)
        return {"sentiment": predictions}

    def run_enrollment_pipeline(self, series: Any) -> Dict[str, Any]:
        """Execute enrollment pipeline."""
        model = self.tools.get_enrollment_model()
        forecast = model.forecast(series)
        return {"forecast": forecast}