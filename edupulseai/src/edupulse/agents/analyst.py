"""Analyst agent that summarizes context and plan."""

from __future__ import annotations

from edupulse.agents.types import AgentResult, Plan, StudentContext
from edupulse.agents.tools import safe_format_list


class AnalystAgent:
    """Create concise summaries and bullets."""

    def run(self, ctx: StudentContext, plan: Plan, cfg: dict | None = None) -> AgentResult:
        summary_parts = [
            f"Student {ctx.student_id} is {ctx.risk_band} risk (priority {ctx.combined_priority}, urgency {ctx.urgency}).",
            f"Dropout risk score: {ctx.dropout_risk:.3f}.",
        ]
        if ctx.sentiment_label:
            summary_parts.append(f"Sentiment: {ctx.sentiment_label} ({ctx.sentiment_score}).")
        if ctx.negative_streak or ctx.sudden_drop:
            summary_parts.append(f"Alerts: negative_streak={ctx.negative_streak}, sudden_drop={ctx.sudden_drop}.")

        bullets = []
        reasons = safe_format_list(ctx.top_reasons)
        if reasons:
            bullets.append(f"Top reasons: {reasons}")
        actions = safe_format_list(ctx.recommended_actions)
        if actions:
            bullets.append(f"Recommended actions: {actions}")
        if plan.escalation:
            bullets.extend(plan.escalation)

        return AgentResult(summary=" ".join(summary_parts), details=bullets)
