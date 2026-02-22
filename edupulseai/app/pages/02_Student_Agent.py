"""Student Agent page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from theme import apply_theme, render_sidebar_branding

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"

import sys as _sys

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(SRC_DIR))
from edupulse.agents.analyst import AnalystAgent
from edupulse.agents.orchestrator import InterventionOrchestrator
from edupulse.agents.planner import PlannerAgent
from edupulse.agents.writer import WriterAgent
from edupulse.agents.orchestrator import build_writer_agent


@st.cache_data
def load_config(config_path: Path) -> dict:
    path = config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@st.cache_data
def load_queue(cfg: dict) -> pd.DataFrame:
    agents_cfg = cfg.get("agents", {}) or {}
    inputs_cfg = agents_cfg.get("inputs", {}) or {}
    queue_path = inputs_cfg.get("combined_queue_path") or Path("reports") / "combined_risk_test.csv"
    queue_path = Path(queue_path)
    if not queue_path.is_absolute():
        queue_path = PROJECT_ROOT / queue_path
    if not queue_path.exists():
        st.error(f"Queue not found at {queue_path}")
        return pd.DataFrame()
    df = pd.read_csv(queue_path)
    if "queue_rank" in df.columns:
        df = df.sort_values(by="queue_rank")
    return df


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _first_available(row: pd.Series | dict, keys: list[str], default: str = "") -> str:
    for key in keys:
        if hasattr(row, "get"):
            value = _safe_text(row.get(key))
        else:
            value = ""
        if value:
            return value
    return default


def _normalize_band(value: str) -> str:
    v = _safe_text(value).lower()
    if v == "high":
        return "High"
    if v == "low":
        return "Low"
    return "Medium"


def _normalize_urgency(value: str) -> str:
    v = _safe_text(value).lower()
    if v == "high":
        return "High"
    if v == "low":
        return "Low"
    return "Medium"


def _normalize_sentiment(value: str) -> str:
    v = _safe_text(value).lower()
    if "negative" in v:
        return "Negative"
    if "positive" in v:
        return "Positive"
    return "Neutral"


def _compose_signal_summary(row: pd.Series | dict) -> str:
    indicators: list[str] = []
    if _first_available(row, ["attendance_trend", "attendance_change", "attendance_rate"]):
        indicators.append("attendance trend")
    if _first_available(row, ["engagement_trend", "engagement_score", "lms_activity"]):
        indicators.append("engagement pattern")
    if _first_available(row, ["missed_assignments", "missed_tasks", "overdue_assignments"]):
        indicators.append("assignment completion")
    if _first_available(row, ["last_login"]):
        indicators.append("learning platform activity")
    if _first_available(row, ["sentiment_label", "sentiment_display"]):
        indicators.append("recent feedback tone")
    if not indicators:
        return "multiple early-warning indicators"
    if len(indicators) == 1:
        return indicators[0]
    return ", ".join(indicators[:-1]) + f", and {indicators[-1]}"


def build_staff_note(student_row: pd.Series | dict) -> str:
    student_id = _first_available(student_row, ["student_id"], "Student")
    risk_band = _normalize_band(_first_available(student_row, ["risk_band"], "Medium"))
    urgency = _normalize_urgency(_first_available(student_row, ["urgency"], "Medium"))
    sentiment = _normalize_sentiment(_first_available(student_row, ["sentiment_label", "sentiment_display"], "Neutral"))
    signals = _compose_signal_summary(student_row)

    if risk_band == "High" and urgency == "High":
        actions = [
            "Initiate same-day outreach by phone/SMS and confirm student safety and availability.",
            "Escalate to advisor/case manager for coordinated support within 24 hours.",
            "Schedule a focused recovery meeting (attendance, coursework, and immediate blockers).",
            "Activate tutor/peer mentor support and set a short 7-day check-in plan.",
        ]
    elif risk_band == "Low" and urgency == "Low":
        actions = [
            "Send a positive progress check-in and reinforce current momentum.",
            "Confirm near-term goals for attendance and coursework continuity.",
            "Encourage use of optional support channels (office hours/study group).",
        ]
    else:
        actions = [
            "Complete a supportive check-in within 48 hours to identify immediate blockers.",
            "Agree on a practical one-week study and attendance plan.",
            "Connect the student to tutoring or academic skills support as needed.",
            "Set a follow-up review date and confirm accountability checkpoints.",
        ]

    if sentiment == "Negative":
        actions.append("Use a wellbeing-first approach and offer counselling/wellbeing referral where appropriate.")
    elif sentiment == "Positive":
        actions.append("Reinforce strengths and convert momentum into consistent weekly habits.")

    prompts = [
        "What is the biggest challenge affecting your progress this week?",
        "Which support option would help you move forward fastest right now?",
        "What is one commitment you can make before our next check-in?",
    ]
    tracking = [
        "Record outreach outcome, agreed actions, and owner.",
        "Monitor attendance/engagement movement before next review.",
        "Update intervention status and escalate if no response or further decline.",
    ]

    return "\n".join(
        [
            "Intervention Brief - Priority Summary",
            "",
            "Why flagged:",
            f"- {student_id} is currently classified as {risk_band} risk with {urgency} urgency based on {signals}.",
            "- This case is surfaced for timely, supportive intervention planning.",
            "",
            "Recommended Actions (next 48 hours):",
            *[f"- {item}" for item in actions[:5]],
            "",
            "Suggested Conversation Prompts:",
            *[f"- {item}" for item in prompts],
            "",
            "Follow-up & Tracking:",
            *[f"- {item}" for item in tracking],
            "",
            "Use professional judgment; this is decision support.",
        ]
    )


def build_student_message(student_row: pd.Series | dict) -> str:
    name = _first_available(student_row, ["first_name", "student_name"], "")
    student_id = _first_available(student_row, ["student_id"], "")
    risk_band = _normalize_band(_first_available(student_row, ["risk_band"], "Medium"))
    urgency = _normalize_urgency(_first_available(student_row, ["urgency"], "Medium"))
    sentiment = _normalize_sentiment(_first_available(student_row, ["sentiment_label", "sentiment_display"], "Neutral"))

    greeting = f"Hi {name}," if name else (f"Hi Student {student_id}," if student_id else "Hi there,")
    immediate_line = (
        "Let's take a few immediate steps today so you can get back in control quickly."
        if urgency == "High"
        else "A few focused steps now can keep your progress on track."
    )

    if risk_band == "High" and urgency == "High":
        steps = [
            "Reply to this message and confirm one time today or tomorrow for a short support call.",
            "List the top 1-2 blockers you are facing right now.",
            "Complete one pending academic task today, even if it is small.",
        ]
    elif risk_band == "Low" and urgency == "Low":
        steps = [
            "Keep your current study rhythm and protect your learning schedule.",
            "Complete your next planned task and note any doubts for class/office hours.",
            "Check in with a teacher or study group this week to stay ahead.",
        ]
    else:
        steps = [
            "Choose one priority task and complete it today.",
            "Set a 30-minute study block for the next two days.",
            "Reach out to a tutor/teacher if you feel stuck on any topic.",
        ]

    if sentiment == "Negative":
        support_line = "If things feel heavy right now, you can also connect with wellbeing support; asking for help is a strength."
    elif sentiment == "Positive":
        support_line = "Your recent momentum is valuable; let's build on it with small, consistent actions."
    else:
        support_line = "You do not have to do this alone; support is available from your teacher, tutor, and student success team."

    return "\n".join(
        [
            greeting,
            "",
            "You are not alone in this. We believe in your ability to recover momentum and move forward with confidence.",
            immediate_line,
            "",
            "Your course matters because it builds practical skills, future opportunities, and the confidence to take your next step in study or career.",
            "Every small improvement now can create meaningful long-term impact.",
            "",
            "Your next 3 steps (today):",
            *[f"- {item}" for item in steps],
            "",
            support_line,
            "If helpful, reply to this message or book a short check-in with your advisor so we can support you early.",
        ]
    )


def _render_message_card(title: str, body: str, border_color: str) -> None:
    body_html = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    st.markdown(
        f"""
        <div style="
            background:#1f2937;
            border:1px solid #374151;
            border-left:4px solid {border_color};
            border-radius:12px;
            padding:16px 18px;
            margin-top:8px;
            margin-bottom:8px;
        ">
            <div style="font-weight:700; margin-bottom:10px;">{title}</div>
            <div style="line-height:1.55;">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_theme()
    render_sidebar_branding()
    st.title("Student Agent")
    cfg = load_config(DEFAULT_CONFIG)
    queue_df = load_queue(cfg)
    if queue_df.empty:
        return

    student_ids = queue_df["student_id"].tolist()
    selected_id = st.selectbox("Select student", student_ids)
    writer_mode = (cfg.get("writer") or {}).get("mode", "template")
    st.caption(f"Writer mode: {writer_mode}")

    if st.button("Generate Intervention Plan"):
        orchestrator = InterventionOrchestrator(PlannerAgent(), AnalystAgent(), build_writer_agent(cfg))
        result = orchestrator.run_intervention(str(selected_id), cfg)

        ctx = result.get("context", {})
        row_match = queue_df[queue_df["student_id"].astype(str) == str(selected_id)]
        row_data = row_match.iloc[0].to_dict() if not row_match.empty else {}
        merged_context = {**row_data, **ctx}

        student_msg = build_student_message(merged_context)
        staff_note = build_staff_note(merged_context)

        st.subheader("Risk Summary")
        st.write(
            f"Risk band: {ctx.get('risk_band')}, Priority: {ctx.get('combined_priority')}, "
            f"Urgency: {ctx.get('urgency')}, Dropout risk: {ctx.get('dropout_risk')}"
        )
        st.write(f"Sentiment: {ctx.get('sentiment_label')} ({ctx.get('sentiment_score')})")
        st.write(f"Top reasons: {ctx.get('top_reasons')}")
        st.write(f"Recommended actions: {ctx.get('recommended_actions')}")
        st.write(f"Owner: {ctx.get('owner')}")

        tabs = st.tabs(["Student Message", "Staff Note"])
        with tabs[0]:
            _render_message_card("Student Success Message", student_msg, "#60a5fa")
        with tabs[1]:
            _render_message_card("Advisor Staff Note", staff_note, "#f59e0b")

        st.subheader("Analyst Details")
        for detail in result.get("analyst_details", []):
            st.markdown(f"- {detail}")

        st.subheader("Audit Log")
        for item in result.get("audit_log", []):
            st.markdown(f"- {item}")


if __name__ == "__main__":
    main()
