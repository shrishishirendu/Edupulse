"""LLM-backed writer with strict parsing and fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from edupulse.agents.types import AgentResult, Plan, StudentContext
from edupulse.agents.tools import safe_format_list, safe_text
from edupulse.agents.writer import WriterAgent
from edupulse.llm.clients import BaseLLMClient, OllamaClient, OpenAICompatibleClient


def _build_prompt(ctx: StudentContext) -> str:
    reasons = safe_format_list(ctx.top_reasons)
    actions = safe_format_list(ctx.recommended_actions)
    matched = safe_format_list(getattr(ctx, "matched_rules", ""))
    return (
        "Write two sections exactly in this format:\n"
        "STUDENT_MESSAGE:\n"
        "<one short message, 60-120 words, supportive tone, no blame, no diagnosis>\n\n"
        "STAFF_NOTE:\n"
        "<one staff note, 80-140 words, includes: risk summary, key drivers, 2-3 actions>\n\n"
        "Inputs:\n"
        f"- student_id: {ctx.student_id}\n"
        f"- risk_band: {ctx.risk_band}\n"
        f"- urgency: {ctx.urgency}\n"
        f"- dropout_risk: {ctx.dropout_risk}\n"
        f"- sentiment_label: {ctx.sentiment_label}\n"
        f"- negative_streak: {ctx.negative_streak}\n"
        f"- sudden_drop: {ctx.sudden_drop}\n"
        f"- top_reasons: {reasons}\n"
        f"- recommended_actions: {actions}\n"
        f"- owner: {ctx.owner}\n"
        f"- matched_rules: {matched}\n"
        "Output exactly the two sections as specified."
    )


def _parse_response(text: str) -> tuple[str, str] | None:
    if not text:
        return None
    marker_student = "STUDENT_MESSAGE:"
    marker_staff = "STAFF_NOTE:"
    lower = text.lower()
    if marker_student.lower() not in lower or marker_staff.lower() not in lower:
        return None
    # Simple split
    parts = text.split(marker_student, 1)
    if len(parts) < 2:
        return None
    rest = parts[1]
    staff_parts = rest.split(marker_staff, 1)
    if len(staff_parts) < 2:
        return None
    student_msg = staff_parts[0].strip()
    staff_note = staff_parts[1].strip()
    if not student_msg or not staff_note:
        return None
    return student_msg, staff_note


@dataclass
class LLMWriterAgent:
    client: BaseLLMClient
    fallback: WriterAgent

    def run(self, ctx: StudentContext, plan: Plan, cfg: Dict[str, Any]) -> AgentResult:
        prompt = _build_prompt(ctx)
        try:
            raw = self.client.generate(prompt)
            parsed = _parse_response(raw)
            if not parsed:
                raise ValueError("LLM output could not be parsed.")
            student_msg, staff_note = parsed
            return AgentResult(summary=student_msg, details=[staff_note])
        except Exception:
            # Fallback to deterministic template
            return self.fallback.run(ctx, plan, cfg)


def build_llm_client(cfg: Dict[str, Any]) -> BaseLLMClient:
    writer_cfg = cfg.get("writer") or {}
    provider = writer_cfg.get("provider", "ollama")
    if provider == "ollama":
        return OllamaClient(
            model=writer_cfg.get("model", "llama3.1:8b"),
            endpoint=writer_cfg.get("endpoint", "http://localhost:11434"),
            temperature=writer_cfg.get("temperature", 0.0),
            max_tokens=writer_cfg.get("max_tokens"),
        )
    if provider in {"lmstudio", "openai_compatible"}:
        return OpenAICompatibleClient(
            model=writer_cfg.get("model", "local-model"),
            endpoint=writer_cfg.get("endpoint", "http://127.0.0.1:1234/v1"),
            temperature=writer_cfg.get("temperature", 0.2),
            max_tokens=writer_cfg.get("max_tokens", 350),
            timeout_seconds=writer_cfg.get("timeout_seconds", 30),
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")
