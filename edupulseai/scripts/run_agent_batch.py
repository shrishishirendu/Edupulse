"""Run intervention orchestrator for top N students from the combined queue."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edupulse.agents.analyst import AnalystAgent
from edupulse.agents.orchestrator import InterventionOrchestrator, build_writer_agent
from edupulse.agents.planner import PlannerAgent
from edupulse.agents.writer import WriterAgent

DEFAULT_CONFIG_PATH = Path("configs") / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch intervention agent for top-N students.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config YAML.")
    parser.add_argument("--top_n", type=int, default=10, help="Number of students to process (default: 10).")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping.")
    return cfg


def load_queue(cfg: Dict[str, Any]) -> pd.DataFrame:
    agents_cfg = cfg.get("agents", {}) or {}
    inputs_cfg = agents_cfg.get("inputs", {}) or {}
    combined_path = inputs_cfg.get("combined_queue_path") or Path("reports") / "combined_risk_test.csv"
    combined_path = Path(combined_path)
    if not combined_path.is_absolute():
        combined_path = PROJECT_ROOT / combined_path
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined queue not found at {combined_path}")

    df = pd.read_csv(combined_path)
    if "queue_rank" not in df.columns:
        raise ValueError("Combined queue must include 'queue_rank'.")
    df = df.sort_values(by="queue_rank")
    return df


def make_markdown(date_str: str, students: List[Dict[str, Any]]) -> str:
    lines = [f"# Daily Counselor Pack – {date_str}"]
    for idx, student in enumerate(students, start=1):
        ctx = student.get("context", {})
        lines.append(f"## Rank {idx} – Student {ctx.get('student_id', 'N/A')}")
        lines.append(f"- Risk: band={ctx.get('risk_band')}, priority={ctx.get('combined_priority')}, urgency={ctx.get('urgency')}")
        lines.append(f"- Top reasons: {ctx.get('top_reasons', '')}")
        lines.append(f"- Recommended actions: {ctx.get('recommended_actions', '')}")
        lines.append("### Student message")
        lines.append(student.get("student_message", ""))
        lines.append("### Staff note")
        lines.append(student.get("staff_note", ""))
    return "\n\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(cfg_path)
    queue_df = load_queue(cfg)
    top_df = queue_df.head(args.top_n)

    orchestrator = InterventionOrchestrator(PlannerAgent(), AnalystAgent(), build_writer_agent(cfg))

    students_outputs: List[Dict[str, Any]] = []
    for _, row in top_df.iterrows():
        sid = row.get("student_id")
        result = orchestrator.run_intervention(str(sid), cfg)
        students_outputs.append(result)

    today_str = date.today().isoformat()
    output_dir = PROJECT_ROOT / "reports" / "agent_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"daily_pack_{today_str}.json"
    md_path = output_dir / f"daily_pack_{today_str}.md"

    payload = {"date": today_str, "top_n": len(students_outputs), "students": students_outputs}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(make_markdown(today_str, students_outputs), encoding="utf-8")

    print(f"Processed {len(students_outputs)} students.")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
