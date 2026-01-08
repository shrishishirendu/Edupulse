"""Daily Counselor Pack page."""

from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

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


def make_markdown(date_str: str, students: list[dict]) -> str:
    lines = [f"# Daily Counselor Pack – {date_str}"]
    for idx, student in enumerate(students, start=1):
        ctx = student.get("context", {})
        lines.append(f"## Rank {idx} – Student {ctx.get('student_id', 'N/A')}")
        lines.append(
            f"- Risk: band={ctx.get('risk_band')}, priority={ctx.get('combined_priority')}, urgency={ctx.get('urgency')}"
        )
        lines.append(f"- Top reasons: {ctx.get('top_reasons', '')}")
        lines.append(f"- Recommended actions: {ctx.get('recommended_actions', '')}")
        lines.append("### Student message")
        lines.append(student.get("student_message", ""))
        lines.append("### Staff note")
        lines.append(student.get("staff_note", ""))
    return "\n\n".join(lines)


def main() -> None:
    st.title("Daily Counselor Pack")
    cfg = load_config(DEFAULT_CONFIG)
    queue_df = load_queue(cfg)
    if queue_df.empty:
        return

    top_n = st.number_input("Top N", min_value=1, max_value=len(queue_df), value=min(10, len(queue_df)), step=1)

    if st.button("Generate Daily Counselor Pack"):
        top_df = queue_df.head(int(top_n))
        orchestrator = InterventionOrchestrator(PlannerAgent(), AnalystAgent(), WriterAgent())

        students_outputs = []
        for _, row in top_df.iterrows():
            sid = row.get("student_id")
            result = orchestrator.run_intervention(str(sid), cfg)
            students_outputs.append(result)

        today_str = date.today().isoformat()
        md_content = make_markdown(today_str, students_outputs)
        st.success(f"Generated pack for {len(students_outputs)} students.")
        st.markdown(md_content)

        md_bytes = md_content.encode("utf-8")
        st.download_button(
            label="Download Markdown",
            data=md_bytes,
            file_name=f"daily_pack_{today_str}.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
