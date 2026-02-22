"""Fuse dropout interventions with sentiment signals into a ranked queue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse dropout and sentiment risk signals.")
    parser.add_argument("--config", type=Path, default=Path("configs") / "config.yaml", help="Path to config YAML.")
    parser.add_argument("--split", choices=["test", "val", "both"], default="both", help="Which splits to fuse.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("Missing dependency: PyYAML. Add PyYAML to requirements.txt and redeploy.")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping.")
    return cfg


def safe_bool(x) -> bool:
    """Handle pd.NA/None/float bool conversions safely."""
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    return bool(x)


def max_urgency(a: str, b: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    a_score = order.get(str(a).lower(), -1)
    b_score = order.get(str(b).lower(), -1)
    return a if a_score >= b_score else b


def compute_combined_score(row: pd.Series, cfg: Dict[str, Any]) -> float:
    fusion_cfg = cfg.get("fusion", {})
    weights = fusion_cfg.get("weights", {})
    risk_w = weights.get("risk_band", {"high": 3.0, "medium": 2.0, "low": 1.0})
    urg_w = weights.get("urgency", {"high": 2.0, "medium": 1.0, "low": 0.0})
    sent_w = weights.get("sentiment_label", {"negative": 1.0, "neutral": 0.0, "positive": -0.5})
    alert_w = weights.get("alert", {"negative_streak": 1.5, "sudden_drop": 1.0})
    score_mult = weights.get("sentiment_score_multiplier", 0.5)

    risk_band = str(row.get("risk_band")).lower()
    urgency = str(row.get("urgency")).lower()
    sent_label = str(row.get("latest_sentiment_label")).lower()
    sent_score = row.get("latest_sentiment_score")

    base = risk_w.get(risk_band, 0.0) + urg_w.get(urgency, 0.0) + sent_w.get(sent_label, 0.0)
    if safe_bool(row.get("negative_streak")):
        base += alert_w.get("negative_streak", 0.0)
    if safe_bool(row.get("sudden_drop")):
        base += alert_w.get("sudden_drop", 0.0)

    if sent_score is not None and pd.notnull(sent_score):
        base += -float(sent_score) * score_mult

    return float(base)


def compute_combined_priority(row: pd.Series, cfg: Dict[str, Any]) -> float:
    fusion_cfg = cfg.get("fusion", {})
    priority_cfg = fusion_cfg.get("priority", {})
    enable_escalation = priority_cfg.get("enable_escalation", True)
    bump_by = priority_cfg.get("bump_priority_by", 1)
    max_bumps = priority_cfg.get("max_bumps", 2)

    base_priority = row.get("combined_priority", row.get("priority", 999))
    bumps = 0
    if enable_escalation:
        if safe_bool(row.get("negative_streak")):
            bumps += 1
        if safe_bool(row.get("sudden_drop")):
            bumps += 1
        sent_score = row.get("latest_sentiment_score")
        sent_label = str(row.get("latest_sentiment_label")).lower()
        neg_thresh = (cfg.get("sentiment") or {}).get("thresholds", {}).get("negative", -0.4)
        if sent_label == "negative" and sent_score is not None and sent_score <= neg_thresh:
            bumps += 1
    bumps = min(bumps, max_bumps)

    combined_priority = max(1, float(base_priority) - bumps * bump_by)
    return combined_priority


def fuse_frames(
    dropout_df: pd.DataFrame,
    latest_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    cfg: Dict[str, Any],
    join_key: str,
    missing_key_warning: List[str],
) -> pd.DataFrame:
    if join_key not in dropout_df.columns:
        missing_key_warning.append(f"WARNING: join key '{join_key}' missing in dropout interventions; sentiment merge skipped.")
        combined = dropout_df.copy()
        for col in ["latest_sentiment_score", "latest_sentiment_label", "negative_streak", "sudden_drop"]:
            if col not in combined.columns:
                combined[col] = pd.NA
    else:
        sentiment_df = latest_df.merge(alerts_df, on=join_key, how="outer", suffixes=("", "_alert"))
        combined = dropout_df.merge(sentiment_df, on=join_key, how="left")

    for col in ["negative_streak", "sudden_drop"]:
        if col not in combined.columns:
            combined[col] = False
        combined[col] = combined[col].fillna(False)

    if "student_id" not in combined.columns:
        combined["student_id"] = ""

    if "sentiment_score" not in combined.columns:
        combined["sentiment_score"] = combined["latest_sentiment_score"] if "latest_sentiment_score" in combined.columns else pd.NA
    if "sentiment_label" not in combined.columns:
        combined["sentiment_label"] = combined["latest_sentiment_label"] if "latest_sentiment_label" in combined.columns else "unknown"

    combined["sentiment_label"] = combined["sentiment_label"].fillna("unknown")

    # Base priority
    if "priority" in combined.columns:
        base = pd.to_numeric(combined["priority"], errors="coerce")
    else:
        base = pd.Series([pd.NA] * len(combined))

    urgency_map = {"high": 1, "medium": 5, "low": 10}
    risk_map = {"high": 2, "medium": 6, "low": 10}

    derived = combined["urgency"].str.lower().map(urgency_map)
    fallback = combined["risk_band"].str.lower().map(risk_map)

    base_filled = base.fillna(derived).fillna(fallback).fillna(10)
    combined["combined_priority"] = pd.to_numeric(base_filled, errors="coerce").fillna(10)

    combined["combined_score"] = pd.to_numeric(
        combined.apply(lambda row: compute_combined_score(row, cfg), axis=1), errors="coerce"
    ).fillna(0)
    combined["combined_priority"] = pd.to_numeric(
        combined.apply(lambda row: compute_combined_priority(row, cfg), axis=1), errors="coerce"
    ).fillna(base_filled).fillna(10)

    # Sorting keys
    urgency_order = {"high": 0, "medium": 1, "low": 2}
    combined["_urg_order"] = combined["urgency"].str.lower().map(urgency_order).fillna(3)
    combined["_sent_sort"] = combined["sentiment_score"].fillna(999)

    if "dropout_risk" not in combined.columns:
        combined["dropout_risk"] = 0.0
    else:
        combined["dropout_risk"] = pd.to_numeric(combined["dropout_risk"], errors="coerce").fillna(0.0)

    if "combined_score" not in combined.columns:
        combined["combined_score"] = 0.0
    else:
        combined["combined_score"] = pd.to_numeric(combined["combined_score"], errors="coerce").fillna(0.0)

    combined = combined.sort_values(
        by=[
            "combined_priority",
            "_urg_order",
            "dropout_risk",
            "negative_streak",
            "sudden_drop",
            "_sent_sort",
            "combined_score",
        ],
        ascending=[True, True, False, False, False, True, False],
    )
    combined = combined.drop(columns=["_urg_order", "_sent_sort"])
    combined["queue_rank"] = range(1, len(combined) + 1)

    include_cols = (cfg.get("fusion") or {}).get("output", {}).get("include_columns")
    if include_cols:
        cols = [col for col in include_cols if col in combined.columns]
        # Ensure essential computed columns retained
        for col in ["combined_score", "combined_priority", "sentiment_score", "sentiment_label", "queue_rank"]:
            if col not in cols and col in combined.columns:
                cols.append(col)
        combined = combined[cols]

    combined = combined.sort_values(by=["combined_priority", "combined_score"], ascending=[True, False])
    return combined


def load_reports(split: str, join_key: str, reports_dir: Path) -> Dict[str, pd.DataFrame]:
    splits = ["val", "test"] if split == "both" else [split]
    dropout_frames = {}
    for sp in splits:
        path = reports_dir / f"dropout_interventions_{sp}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Dropout interventions not found for split '{sp}' at {path}")
        dropout_frames[sp] = pd.read_csv(path)

    sentiment_latest_path = reports_dir / "sentiment_student_latest.csv"
    sentiment_alerts_path = reports_dir / "sentiment_alerts.csv"
    if not sentiment_latest_path.exists() or not sentiment_alerts_path.exists():
        raise FileNotFoundError("Sentiment reports not found; run run_sentiment.py first.")

    latest_df = pd.read_csv(sentiment_latest_path)
    alerts_df = pd.read_csv(sentiment_alerts_path)
    return {"dropout": dropout_frames, "latest": latest_df, "alerts": alerts_df}


def print_summary(df: pd.DataFrame) -> None:
    merged_count = df["latest_sentiment_score"].notna().sum() if "latest_sentiment_score" in df.columns else 0
    print(f"Merged rows with sentiment: {merged_count}/{len(df)}")
    print(f"combined_priority min/max: {df['combined_priority'].min()} / {df['combined_priority'].max()}")
    print("Top combined_priority counts:")
    print(df["combined_priority"].value_counts().head(10).to_dict())
    print("Top 10 by queue rank:")
    preview_cols = [
        "queue_rank",
        "student_id",
        "combined_priority",
        "urgency",
        "dropout_risk",
        "sentiment_label",
        "negative_streak",
        "sudden_drop",
    ]
    print(
        df.sort_values(by=["queue_rank"], ascending=True)
        .head(10)[[col for col in preview_cols if col in df.columns]]
        .to_string(index=False)
    )
    alert_cols = [col for col in ["negative_streak", "sudden_drop"] if col in df.columns]
    if alert_cols:
        alert_counts = {col: int(df[col].fillna(False).astype(bool).sum()) for col in alert_cols}
        print(f"Alert counts: {alert_counts}")


def main() -> None:
    args = parse_args()
    cfg_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(cfg_path)
    fusion_cfg = cfg.get("fusion") or {}
    join_key = fusion_cfg.get("student_id_column") or (cfg.get("dataset") or {}).get("id_column")
    if not join_key:
        print("Fusion join key not configured (fusion.student_id_column).", file=sys.stderr)
        sys.exit(1)

    reports_dir = PROJECT_ROOT / "reports"
    try:
        reports = load_reports(args.split, join_key, reports_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    dropout_frames = reports["dropout"]
    latest_df = reports["latest"]
    alerts_df = reports["alerts"]
    missing_key_warning: List[str] = []

    for split, dropout_df in dropout_frames.items():
        fused = fuse_frames(dropout_df, latest_df, alerts_df, cfg, join_key, missing_key_warning)
        output_path = reports_dir / f"combined_risk_{split}.csv"
        fused.to_csv(output_path, index=False)
        print(f"Wrote {output_path}")
        print_summary(fused)

    for warning in missing_key_warning:
        print(warning, file=sys.stderr)


if __name__ == "__main__":
    main()
