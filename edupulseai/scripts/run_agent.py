"""Run the intervention orchestrator for a given student."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

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
    parser = argparse.ArgumentParser(description="Run EduPulse intervention agent.")
    parser.add_argument("--student_id", required=True, help="Student identifier to process.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config YAML.")
    parser.add_argument("--mode", choices=["intervention"], default="intervention", help="Mode to run (intervention).")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping.")
    return cfg


def write_outputs(student_id: str, payload: Dict[str, Any]) -> None:
    output_dir = PROJECT_ROOT / "reports" / "agent_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"student_{student_id}_{timestamp}.json"
    md_path = output_dir / f"student_{student_id}_{timestamp}.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        f"# Student {student_id} Intervention",
        "## Context",
        json.dumps(payload.get("context", {}), indent=2),
        "## Plan",
        json.dumps(payload.get("plan", {}), indent=2),
        "## Analyst Summary",
        payload.get("analyst_summary", ""),
        "## Analyst Details",
        "\n".join(f"- {d}" for d in payload.get("analyst_details", []) or []),
        "## Student Message",
        payload.get("student_message", ""),
        "## Staff Note",
        payload.get("staff_note", ""),
        "## Audit Log",
        "\n".join(f"- {item}" for item in payload.get("audit_log", []) or []),
    ]
    md_path.write_text("\n\n".join(md_lines), encoding="utf-8")

    print(f"Wrote outputs to {json_path} and {md_path}")


def main() -> None:
    args = parse_args()
    cfg_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(cfg_path)

    orchestrator = InterventionOrchestrator(
        planner=PlannerAgent(),
        analyst=AnalystAgent(),
        writer=build_writer_agent(cfg),
    )

    result = orchestrator.run_intervention(args.student_id, cfg)
    write_outputs(str(args.student_id), result)


if __name__ == "__main__":
    main()
