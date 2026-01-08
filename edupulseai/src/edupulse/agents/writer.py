"""Writer agent to generate deterministic drafts."""

from __future__ import annotations

from edupulse.agents.types import AgentResult, Plan, StudentContext
from edupulse.agents.tools import safe_format_list, safe_text


class WriterAgent:
    """Produce student and staff drafts without LLMs."""

    def __init__(self) -> None:
        self.sentiment_tones = {
            "negative": "supportive and reassuring",
            "neutral": "informative and concise",
            "positive": "encouraging and appreciative",
        }

    def run(self, ctx: StudentContext, plan: Plan, cfg: dict | None = None) -> AgentResult:
        tone = self.sentiment_tones.get(safe_text(ctx.sentiment_label).lower(), "supportive and clear")
        student_msg = self._student_message(ctx, tone)
        staff_note = self._staff_note(ctx, plan)
        return AgentResult(summary=student_msg, details=[staff_note])

    def _student_message(self, ctx: StudentContext, tone: str) -> str:
        reasons = safe_format_list(ctx.top_reasons)
        actions = safe_format_list(ctx.recommended_actions)
        return (
            f"Hi {ctx.student_id},\n\n"
            f"We're reaching out with a {tone} note. "
            f"Our latest review shows your current status as {ctx.risk_band} risk with priority {ctx.combined_priority}. "
            f"Dropout risk score: {ctx.dropout_risk:.3f}. "
            f"Sentiment: {ctx.sentiment_label} ({ctx.sentiment_score}). "
            f"{'Key factors: ' + reasons + '. ' if reasons else ''}"
            f"{'Suggested steps: ' + actions + '. ' if actions else ''}"
            f"Please let us know how we can support you."
        )

    def _staff_note(self, ctx: StudentContext, plan: Plan) -> str:
        actions = safe_format_list(ctx.recommended_actions)
        checklist = "; ".join(plan.checklist) if plan.checklist else ""
        escalation = "; ".join(plan.escalation) if plan.escalation else ""
        return (
            f"Staff note for {ctx.student_id}: "
            f"risk_band={ctx.risk_band}, urgency={ctx.urgency}, priority={ctx.combined_priority}. "
            f"Sentiment={ctx.sentiment_label} ({ctx.sentiment_score}), alerts: "
            f"negative_streak={ctx.negative_streak}, sudden_drop={ctx.sudden_drop}. "
            f"Top reasons: {safe_format_list(ctx.top_reasons)}. "
            f"Recommended actions: {actions}. "
            f"Checklist: {checklist}. Escalation: {escalation}."
        )
