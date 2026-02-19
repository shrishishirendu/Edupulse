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
        st.subheader("Risk Summary")
        st.write(
            f"Risk band: {ctx.get('risk_band')}, Priority: {ctx.get('combined_priority')}, "
            f"Urgency: {ctx.get('urgency')}, Dropout risk: {ctx.get('dropout_risk')}"
        )
        st.write(f"Sentiment: {ctx.get('sentiment_label')} ({ctx.get('sentiment_score')})")
        st.write(f"Top reasons: {ctx.get('top_reasons')}")
        st.write(f"Recommended actions: {ctx.get('recommended_actions')}")
        st.write(f"Owner: {ctx.get('owner')}")

        st.subheader("Student Message")
        st.info(result.get("student_message", ""))

        st.subheader("Staff Note")
        st.warning(result.get("staff_note", ""))

        st.subheader("Analyst Details")
        for detail in result.get("analyst_details", []):
            st.markdown(f"- {detail}")

        st.subheader("Audit Log")
        for item in result.get("audit_log", []):
            st.markdown(f"- {item}")


if __name__ == "__main__":
    main()
