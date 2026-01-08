"""Deterministic planner agent."""

from __future__ import annotations

from edupulse.agents.types import Plan, StudentContext
from edupulse.agents.tools import safe_format_list


class PlannerAgent:
    """Plan next steps for a student context."""

    def run(self, ctx: StudentContext, cfg: dict | None = None) -> Plan:
        plan = Plan()

        triage_level = self._triage(ctx)
        plan.steps.append(f"Triage level: {triage_level} (priority {ctx.combined_priority}, urgency {ctx.urgency})")

        # Required checks
        checks = ["Attendance review", "Academic performance review", "Financial/fees check", "Wellbeing check-in"]
        plan.steps.extend(checks)

        # Checklist from recommended actions
        actions = safe_format_list(ctx.recommended_actions)
        if actions:
            plan.checklist.append(f"Recommended actions: {actions}")

        # Escalation
        if ctx.negative_streak:
            plan.escalation.append("Escalate to advisor: negative sentiment streak detected.")
        if ctx.sudden_drop:
            plan.escalation.append("Escalate to coach: sudden sentiment drop detected.")
        if ctx.risk_band == "high":
            plan.escalation.append("High risk band: prioritize outreach within 24h.")

        return plan

    def _triage(self, ctx: StudentContext) -> str:
        if ctx.urgency == "high" or ctx.combined_priority <= 2:
            return "Critical"
        if ctx.urgency == "medium" or ctx.combined_priority <= 4:
            return "Elevated"
        return "Standard"
