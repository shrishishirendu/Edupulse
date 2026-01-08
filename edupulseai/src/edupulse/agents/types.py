"""Agent data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StudentContext:
    student_id: str | int
    risk_band: str
    dropout_risk: float
    urgency: str
    combined_priority: float
    top_reasons: str = ""
    recommended_actions: str = ""
    owner: str = ""
    sentiment_score: Optional[float] = None
    sentiment_label: str | None = None
    negative_streak: bool = False
    sudden_drop: bool = False
    sentiment_rolling_score: Optional[float] = None
    sentiment_rolling_delta: Optional[float] = None


@dataclass
class Plan:
    steps: List[str] = field(default_factory=list)
    escalation: List[str] = field(default_factory=list)
    checklist: List[str] = field(default_factory=list)


@dataclass
class AgentResult:
    summary: str
    details: List[str] = field(default_factory=list)
