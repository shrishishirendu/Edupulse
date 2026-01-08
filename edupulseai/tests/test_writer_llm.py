"""Tests for LLM-backed writer."""

from __future__ import annotations

from dataclasses import dataclass

from edupulse.agents.types import AgentResult, Plan, StudentContext
from edupulse.agents.writer import WriterAgent
from edupulse.agents.writer_llm import LLMWriterAgent, build_llm_client
from edupulse.llm import clients as llm_clients


@dataclass
class FakeClient:
    response: str

    def generate(self, prompt: str) -> str:
        return self.response


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _context() -> StudentContext:
    return StudentContext(
        student_id="1",
        risk_band="high",
        dropout_risk=0.8,
        urgency="high",
        combined_priority=1,
        top_reasons="attendance; gpa",
        recommended_actions="Call student; Notify instructor",
        owner="Coach",
        sentiment_label="negative",
        sentiment_score=-0.5,
        negative_streak=True,
        sudden_drop=False,
    )


def test_llm_writer_parses_two_sections() -> None:
    fake_output = "STUDENT_MESSAGE:\nHello student.\n\nSTAFF_NOTE:\nThis is a staff note."
    agent = LLMWriterAgent(client=FakeClient(fake_output), fallback=WriterAgent())
    result: AgentResult = agent.run(_context(), Plan(), {})

    assert "Hello student" in result.summary
    assert "staff note" in result.details[0]


def test_llm_writer_falls_back_on_bad_output() -> None:
    fake_output = "BAD OUTPUT"
    agent = LLMWriterAgent(client=FakeClient(fake_output), fallback=WriterAgent())
    result: AgentResult = agent.run(_context(), Plan(), {})

    # Template writer includes student_id in output
    assert "1" in result.summary


def test_build_llm_client_accepts_lmstudio(monkeypatch) -> None:
    def fake_post(url: str, json: dict, timeout: int) -> FakeResponse:
        assert url == "http://127.0.0.1:1234/v1/chat/completions"
        assert json["model"] == "local-model"
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "STUDENT_MESSAGE:\nHello.\n\nSTAFF_NOTE:\nStaff note."
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(llm_clients.requests, "post", fake_post)
    cfg = {
        "writer": {
            "provider": "lmstudio",
            "endpoint": "http://127.0.0.1:1234/v1",
            "model": "local-model",
        }
    }
    client = build_llm_client(cfg)
    agent = LLMWriterAgent(client=client, fallback=WriterAgent())
    result: AgentResult = agent.run(_context(), Plan(), cfg)

    assert "Hello" in result.summary
    assert "Staff note" in result.details[0]


def test_llm_writer_falls_back_on_missing_sections(monkeypatch) -> None:
    def fake_post(url: str, json: dict, timeout: int) -> FakeResponse:
        return FakeResponse({"choices": [{"message": {"content": "No sections here"}}]})

    monkeypatch.setattr(llm_clients.requests, "post", fake_post)
    cfg = {
        "writer": {
            "provider": "openai_compatible",
            "endpoint": "http://127.0.0.1:1234/v1",
        }
    }
    client = build_llm_client(cfg)
    agent = LLMWriterAgent(client=client, fallback=WriterAgent())
    result: AgentResult = agent.run(_context(), Plan(), cfg)

    assert "1" in result.summary
