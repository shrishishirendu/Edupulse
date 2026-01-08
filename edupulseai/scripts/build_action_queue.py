"""Build a simple action queue artifact from fused risk signals."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
INPUT_PATH = REPORTS_DIR / "combined_risk_test.csv"
OUTPUT_PATH = REPORTS_DIR / "action_queue_test.csv"

COLUMNS = [
    "student_id",
    "combined_priority",
    "combined_score",
    "risk_band",
    "dropout_risk",
    "urgency",
    "sentiment_label",
    "sentiment_score",
    "negative_streak",
    "sudden_drop",
    "top_reasons",
    "recommended_actions",
    "owner",
]


def build_action_queue() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        print(f"Input not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)
    missing = [col for col in COLUMNS if col not in df.columns]
    if missing:
        print(f"Warning: missing columns {missing}; filling with blanks/defaults.")
        for col in missing:
            if col in ("negative_streak", "sudden_drop"):
                df[col] = False
            elif col == "sentiment_score":
                df[col] = pd.NA
            else:
                df[col] = ""

    queue = df[COLUMNS].copy()
    queue = queue.sort_values(by=["combined_priority", "combined_score"], ascending=[True, False])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(OUTPUT_PATH, index=False)

    preview_cols = ["student_id", "combined_priority", "risk_band", "urgency", "sentiment_label"]
    print("Top 20 preview:")
    print(queue[preview_cols].head(20).to_string(index=False))
    return queue


def main() -> None:
    build_action_queue()


if __name__ == "__main__":
    main()
