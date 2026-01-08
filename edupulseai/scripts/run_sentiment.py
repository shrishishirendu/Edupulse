"""Run lexicon-based sentiment scoring and aggregation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edupulse.sentiment.aggregate import (
    add_sentiment,
    detect_alerts,
    load_feedback_csv,
    student_latest,
    student_rolling,
)


DEFAULT_CONFIG_PATH = Path("configs") / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lexicon-based sentiment pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config YAML.")
    parser.add_argument("--demo", action="store_true", help="Generate a tiny demo dataset if raw_path is missing.")
    return parser.parse_args()


def load_config(cfg_path: Path) -> Dict[str, Any]:
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Configuration file must be a mapping.")
    return cfg


def ensure_demo_data(path: Path, cfg: Dict[str, Any]) -> None:
    """Create a small demo dataset if requested."""
    sentiment_cfg = cfg.get("sentiment") or {}
    id_col = sentiment_cfg.get("id_column", "student_id")
    text_col = sentiment_cfg.get("text_column", "text")
    time_col = sentiment_cfg.get("time_column", "timestamp")

    data = [
        {id_col: 1, text_col: "I really enjoy this course, very helpful content!", time_col: "2026-01-01T09:00:00"},
        {id_col: 1, text_col: "Not happy with the latest assignment feedback.", time_col: "2026-01-05T10:00:00"},
        {id_col: 2, text_col: "The instructor is extremely supportive.", time_col: "2026-01-02T14:00:00"},
        {id_col: 2, text_col: "I am frustrated and confused about the project.", time_col: "2026-01-06T15:00:00"},
        {id_col: 3, text_col: "Content is fine, nothing special.", time_col: "2026-01-03T12:00:00"},
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)
    print(f"Demo sentiment data written to {path}")


def main() -> None:
    args = parse_args()
    cfg_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(cfg_path)
    sentiment_cfg = cfg.get("sentiment") or {}

    raw_path = Path(sentiment_cfg.get("raw_path", ""))
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path

    if not raw_path.exists():
        if args.demo:
            ensure_demo_data(raw_path, cfg)
        else:
            print(f"Sentiment raw data not found at {raw_path}. Use --demo to generate sample data.", file=sys.stderr)
            sys.exit(1)

    id_col = sentiment_cfg.get("id_column", "id")
    text_col = sentiment_cfg.get("text_column", "text")
    time_col = sentiment_cfg.get("time_column", "timestamp")

    df = load_feedback_csv(raw_path, id_col=id_col, text_col=text_col, time_col=time_col)
    df = add_sentiment(df, cfg)
    df = student_rolling(df, cfg)
    latest = student_latest(df, cfg)
    alerts = detect_alerts(df, cfg)

    output_dir = sentiment_cfg.get("output_dir", "reports")
    output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = output_dir / "sentiment_events.csv"
    latest_path = output_dir / "sentiment_student_latest.csv"
    rolling_path = output_dir / "sentiment_student_rolling.csv"
    alerts_path = output_dir / "sentiment_alerts.csv"

    df.to_csv(events_path, index=False)
    latest.to_csv(latest_path, index=False)
    df[[id_col, time_col, "rolling_score", "rolling_delta"]].to_csv(rolling_path, index=False)
    alerts.to_csv(alerts_path, index=False)

    print("Sentiment pipeline complete.")
    print(f"Label counts: {df['sentiment_label'].value_counts().to_dict()}")
    rolling_means = df.groupby(id_col)["rolling_score"].mean().sort_values().head(10)
    print("Top 10 students with most negative rolling score:")
    print(rolling_means.to_dict())


if __name__ == "__main__":
    main()
