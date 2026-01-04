"""Agent tool registry."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.edupulse.models.dropout import DropoutRiskModel
from src.edupulse.models.enrollment import EnrollmentForecaster
from src.edupulse.models.sentiment import SentimentModel


@dataclass
class ToolRegistry:
    """Simple container for analytical tools accessible by agents."""

    dropout_model: DropoutRiskModel = field(default_factory=DropoutRiskModel)
    sentiment_model: SentimentModel = field(default_factory=SentimentModel)
    enrollment_model: EnrollmentForecaster = field(default_factory=EnrollmentForecaster)

    def get_dropout_model(self) -> DropoutRiskModel:
        """Return the configured dropout risk estimator."""
        return self.dropout_model

    def get_sentiment_model(self) -> SentimentModel:
        """Return the configured sentiment model."""
        return self.sentiment_model

    def get_enrollment_model(self) -> EnrollmentForecaster:
        """Return the configured enrollment forecaster."""
        return self.enrollment_model