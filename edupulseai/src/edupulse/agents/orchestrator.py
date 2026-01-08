"""Agent orchestrator for deterministic intervention generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict

from edupulse.agents.analyst import AnalystAgent
from edupulse.agents.planner import PlannerAgent
from edupulse.agents.tools import load_student_context
from edupulse.agents.types import AgentResult, Plan, StudentContext
from edupulse.agents.writer import WriterAgent
from edupulse.agents.writer_llm import LLMWriterAgent, build_llm_client


@dataclass
class InterventionOrchestrator:
    """Coordinate planner, analyst, and writer agents."""

    planner: PlannerAgent
    analyst: AnalystAgent
    writer: WriterAgent

    def run_intervention(self, student_id: str | int, cfg: Dict[str, Any]) -> Dict[str, Any]:
        ctx: StudentContext = load_student_context(student_id, cfg)
        plan: Plan = self.planner.run(ctx, cfg)
        analyst_result: AgentResult = self.analyst.run(ctx, plan, cfg)
        writer_result: AgentResult = self.writer.run(ctx, plan, cfg)

        audit_log = [
            "Loaded student context",
            "Planner generated steps",
            "Analyst summarized context",
            "Writer generated drafts",
        ]

        return {
            "context": asdict(ctx),
            "plan": asdict(plan),
            "analyst_summary": analyst_result.summary,
            "analyst_details": analyst_result.details,
            "student_message": writer_result.summary,
            "staff_note": writer_result.details[0] if writer_result.details else "",
            "audit_log": audit_log,
        }


def build_writer_agent(cfg: Dict[str, Any]) -> WriterAgent | LLMWriterAgent:
    writer_cfg = cfg.get("writer") or {}
    mode = writer_cfg.get("mode", "template").lower()
    if mode == "llm":
        client = build_llm_client(cfg)
        return LLMWriterAgent(client=client, fallback=WriterAgent())
    return WriterAgent()
