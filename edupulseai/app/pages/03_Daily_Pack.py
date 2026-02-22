"""Daily Counsellor Pack page."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from sentiment_utils import ensure_sentiment_labels, load_latest_feedback
from theme import apply_theme, render_sidebar_branding

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


@st.cache_data
def load_config(config_path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Add PyYAML to requirements.txt.")
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


@st.cache_data
def load_feedback(cfg: dict) -> pd.DataFrame:
    sentiment_cfg = cfg.get("sentiment", {}) or {}
    feedback_path = Path(sentiment_cfg.get("raw_path", "data/raw/student_feedback.csv"))
    if not feedback_path.is_absolute():
        feedback_path = PROJECT_ROOT / feedback_path
    return load_latest_feedback(feedback_path)


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _get_first(row: pd.Series, candidates: list[str], default: str = "") -> str:
    for col in candidates:
        if col in row.index:
            val = _safe_text(row.get(col))
            if val:
                return val
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
    low = _safe_text(value).lower()
    if "negative" in low:
        return "🔴 Negative"
    if "positive" in low:
        return "🟢 Positive"
    return "🟡 Neutral"


def _primary_signal(row: pd.Series) -> str:
    checks = [
        (["attendance_trend", "attendance_change", "attendance_rate"], "Attendance trend change"),
        (["engagement_trend", "engagement_score", "lms_activity"], "Engagement drop signal"),
        (["missed_assignments", "missed_tasks", "overdue_assignments"], "Missed assignment pressure"),
        (["last_login"], "Reduced recent platform activity"),
        (["negative_streak", "sudden_drop"], "Recent wellbeing/behaviour concern"),
        (["sentiment_label", "sentiment_display"], "Recent feedback tone shift"),
    ]
    for cols, phrase in checks:
        if any(col in row.index and _safe_text(row.get(col)) for col in cols):
            return phrase
    return "Multi-factor early-warning signal"


def _action_type(row: pd.Series) -> str:
    band = _normalize_band(_safe_text(row.get("risk_band", "Medium")))
    urgency = _normalize_urgency(_safe_text(row.get("urgency", "Medium")))
    sentiment = _normalize_sentiment(_safe_text(row.get("sentiment_label", row.get("sentiment_display", ""))))

    if band == "High" and urgency == "High":
        return "Call"
    if "Negative" in sentiment:
        return "Wellbeing Referral"
    if band == "High" or urgency == "High":
        return "Academic Review"
    if band == "Medium" or urgency == "Medium":
        return "Call"
    return "Monitor"


def _prepare_for_sort(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    urgency_rank = {"high": 3, "medium": 2, "low": 1}
    risk_rank = {"high": 3, "medium": 2, "low": 1}

    if "urgency" in out.columns:
        out["_urg_rank"] = out["urgency"].astype(str).str.lower().map(urgency_rank).fillna(0)
    else:
        out["_urg_rank"] = 0

    if "risk_band" in out.columns:
        out["_risk_rank"] = out["risk_band"].astype(str).str.lower().map(risk_rank).fillna(0)
    else:
        out["_risk_rank"] = 0

    sort_cols = ["_urg_rank", "_risk_rank"]
    ascending = [False, False]
    if "queue_rank" in out.columns:
        sort_cols.append("queue_rank")
        ascending.append(True)

    return out.sort_values(by=sort_cols, ascending=ascending)


def _render_case_card(row: pd.Series, accent_color: str) -> None:
    student_label = _get_first(row, ["student_name", "first_name", "name"], "Student")
    student_id = _get_first(row, ["student_id"], "N/A")
    risk = _normalize_band(_safe_text(row.get("risk_band", "Medium")))
    urgency = _normalize_urgency(_safe_text(row.get("urgency", "Medium")))
    sentiment = _normalize_sentiment(_safe_text(row.get("sentiment_label", row.get("sentiment_display", ""))))
    signal = _primary_signal(row)
    action = _action_type(row)

    st.markdown(
        f"""
        <div style="background:#111827;border:1px solid #374151;border-left:4px solid {accent_color};border-radius:12px;padding:12px 14px;margin-bottom:10px;">
            <div style="font-weight:700;">{student_label} <span style="color:#9ca3af;font-weight:500;">({student_id})</span></div>
            <div style="margin-top:6px;color:#d1d5db;">Risk Band: <b>{risk}</b> &nbsp;|&nbsp; Urgency: <b>{urgency}</b> &nbsp;|&nbsp; Sentiment: <b>{sentiment}</b></div>
            <div style="margin-top:6px;color:#d1d5db;">Primary Signal: {signal}</div>
            <div style="margin-top:6px;color:#d1d5db;">Suggested Action Type: <b>{action}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _tier_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    risk = df["risk_band"].astype(str).str.lower() if "risk_band" in df.columns else pd.Series(["medium"] * len(df), index=df.index)
    urg = df["urgency"].astype(str).str.lower() if "urgency" in df.columns else pd.Series(["medium"] * len(df), index=df.index)

    tier1 = (risk == "high") & (urg == "high")
    tier3 = (risk == "low") & (urg == "low")
    tier2 = ((risk == "medium") | (urg == "medium")) & (~tier1) & (~tier3)
    return tier1, tier2, tier3


def _render_tier_tab(df: pd.DataFrame, mask: pd.Series, accent_color: str) -> None:
    tier_df = _prepare_for_sort(df[mask].copy()).head(5)
    if tier_df.empty:
        st.caption("No students in this tier today.")
        return
    for _, row in tier_df.iterrows():
        _render_case_card(row, accent_color)


def main() -> None:
    apply_theme()
    render_sidebar_branding()

    cfg = load_config(DEFAULT_CONFIG)
    queue_df = load_queue(cfg)
    if queue_df.empty:
        return

    feedback_df = load_feedback(cfg)
    queue_df = ensure_sentiment_labels(queue_df, student_id_col="student_id", feedback_df=feedback_df)

    today_str = date.today().strftime("%B %d, %Y")
    st.title(f"Counsellor Daily Pack — {today_str}")

    tier1_mask, tier2_mask, tier3_mask = _tier_masks(queue_df)
    tabs = st.tabs([
        "Tier 1 – Immediate Intervention",
        "Tier 2 – Monitor & Check-in",
        "Tier 3 – Stable Observation",
    ])

    with tabs[0]:
        _render_tier_tab(queue_df, tier1_mask, "#ef4444")
    with tabs[1]:
        _render_tier_tab(queue_df, tier2_mask, "#f59e0b")
    with tabs[2]:
        _render_tier_tab(queue_df, tier3_mask, "#22c55e")


if __name__ == "__main__":
    main()
