"""Action Queue page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from theme import apply_theme, render_sidebar_branding

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


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


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    risk_opts = sorted(df["risk_band"].dropna().unique())
    urg_opts = sorted(df["urgency"].dropna().unique())
    sent_opts = sorted(df["sentiment_label"].dropna().unique())

    st.sidebar.header("Filters")
    selected_risk = st.sidebar.multiselect("Risk band", risk_opts, default=risk_opts)
    selected_urg = st.sidebar.multiselect("Urgency", urg_opts, default=urg_opts)
    selected_sent = st.sidebar.multiselect("Sentiment", sent_opts, default=sent_opts)
    only_alerts = st.sidebar.checkbox("Only with alerts (negative streak or sudden drop)", value=False)

    mask = (
        df["risk_band"].isin(selected_risk)
        & df["urgency"].isin(selected_urg)
        & df["sentiment_label"].isin(selected_sent)
    )
    if only_alerts and {"negative_streak", "sudden_drop"}.issubset(df.columns):
        mask &= (df["negative_streak"].fillna(False) | df["sudden_drop"].fillna(False))

    return df[mask]


def render_detail(row: pd.Series) -> None:
    st.subheader(f"Student {row.get('student_id', 'N/A')}")
    st.markdown(f"**Queue rank:** {row.get('queue_rank', 'N/A')} | **Priority:** {row.get('combined_priority', 'N/A')}")
    st.markdown(
        f"Risk band: {row.get('risk_band', 'N/A')} | Urgency: {row.get('urgency', 'N/A')} | "
        f"Sentiment: {row.get('sentiment_label', 'N/A')} ({row.get('sentiment_score', 'N/A')})"
    )
    st.markdown(f"Dropout risk: {row.get('dropout_risk', 'N/A')}")
    st.markdown(f"Top reasons: {row.get('top_reasons', '')}")
    st.markdown(f"Recommended actions: {row.get('recommended_actions', '')}")
    st.markdown(f"Owner: {row.get('owner', '')}")
    st.markdown(
        f"Alerts - Negative streak: {row.get('negative_streak', False)}, Sudden drop: {row.get('sudden_drop', False)}"
    )


def main() -> None:
    apply_theme()
    render_sidebar_branding()
    st.title("Action Queue")
    cfg = load_config(DEFAULT_CONFIG)
    df = load_queue(cfg)
    if df.empty:
        return

    df_filtered = filter_df(df)
    display_cols = ["queue_rank", "student_id", "risk_band", "urgency", "dropout_risk", "sentiment_label"]
    st.dataframe(df_filtered[display_cols], use_container_width=True)

    if not df_filtered.empty:
        st.markdown("Select a row to view details (first row shown by default).")
        selected_id = st.selectbox("Student", df_filtered["student_id"].tolist())
        row = df_filtered[df_filtered["student_id"] == selected_id].iloc[0]
        render_detail(row)


if __name__ == "__main__":
    main()
