"""Action Queue page."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None

from sentiment_utils import ensure_sentiment_labels, load_latest_feedback
from theme import apply_theme, render_sidebar_branding

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


@st.cache_data
def load_config(config_path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("Missing dependency: PyYAML. Add PyYAML to requirements.txt and redeploy.")
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


def _normalize_level(value: object, default: str) -> str:
    text = _safe_text(value).lower()
    if text == "high":
        return "High"
    if text == "low":
        return "Low"
    if text == "medium":
        return "Medium"
    return default


def _sentiment_name(value: object) -> str:
    low = _safe_text(value).lower()
    if "negative" in low:
        return "Negative"
    if "positive" in low:
        return "Positive"
    return "Neutral"


def sentiment_style(value: object) -> str:
    text = _safe_text(value).lower()
    if "negative" in text:
        return "color: #ff6b6b; font-weight: 600;"
    if "neutral" in text:
        return "color: #f6d365; font-weight: 600;"
    if "positive" in text:
        return "color: #66d9a3; font-weight: 600;"
    return ""


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    risk_opts = sorted(df["risk_band"].dropna().unique()) if "risk_band" in df.columns else []
    urg_opts = sorted(df["urgency"].dropna().unique()) if "urgency" in df.columns else []
    sent_opts = sorted(df["sentiment_label"].dropna().unique()) if "sentiment_label" in df.columns else []

    st.sidebar.header("Filters")
    selected_risk = st.sidebar.multiselect("Risk band", risk_opts, default=risk_opts)
    selected_urg = st.sidebar.multiselect("Urgency", urg_opts, default=urg_opts)
    selected_sent = st.sidebar.multiselect("Sentiment", sent_opts, default=sent_opts)
    only_alerts = st.sidebar.checkbox("Only with alerts (negative streak or sudden drop)", value=False)

    mask = pd.Series([True] * len(df), index=df.index)
    if "risk_band" in df.columns:
        mask &= df["risk_band"].isin(selected_risk)
    if "urgency" in df.columns:
        mask &= df["urgency"].isin(selected_urg)
    if "sentiment_label" in df.columns:
        mask &= df["sentiment_label"].isin(selected_sent)
    if only_alerts and {"negative_streak", "sudden_drop"}.issubset(df.columns):
        mask &= (df["negative_streak"].fillna(False) | df["sudden_drop"].fillna(False))

    return df[mask]


def _has_value(row: pd.Series, columns: list[str]) -> bool:
    return any(col in row.index and _safe_text(row.get(col)) for col in columns)


def build_key_signals(row: pd.Series) -> list[str]:
    signal_types: list[str] = []
    if _has_value(row, ["attendance_trend", "attendance_change", "attendance_rate"]):
        signal_types.append("attendance")
    if _has_value(row, ["engagement_change", "engagement_trend", "engagement_score", "lms_activity"]):
        signal_types.append("engagement")
    if _has_value(row, ["missed_assignments", "missed_tasks", "overdue_assignments"]):
        signal_types.append("submissions")
    if _has_value(row, ["sentiment_label", "sentiment_score", "negative_streak", "sudden_drop"]):
        signal_types.append("sentiment")
    if _has_value(row, ["last_login", "inactivity_days"]):
        signal_types.append("inactivity")
    if _has_value(row, ["anomaly", "anomaly_score", "recent_anomalies", "alerts"]):
        signal_types.append("anomaly")
    return signal_types[:4]


def build_intervention_summary(row: pd.Series) -> str:
    try:
        sid = _safe_text(row.get("student_id")) or _safe_text(getattr(row, "name", "unknown"))
        h = int(hashlib.md5(sid.encode()).hexdigest()[:8], 16)

        def pick(options: list[str], offset: int = 0, fallback: str = "") -> str:
            if not options:
                return fallback
            return options[(h + offset) % len(options)]

        risk = _normalize_level(row.get("risk_band"), "Medium")
        urgency = _normalize_level(row.get("urgency"), "Medium")
        sentiment = _sentiment_name(row.get("sentiment_label"))

        summary_openers = {
            "high_high": [
                "This case requires prompt intervention coordination.",
                "Immediate advisor action is recommended for this student.",
                "This profile indicates elevated near-term support urgency.",
                "Priority escalation is advised to reduce interruption risk.",
                "Rapid outreach is warranted based on current warning signals.",
                "This student should be treated as an immediate intervention case.",
                "An accelerated support response is recommended at this stage.",
                "Current indicators justify same-cycle intervention planning.",
            ],
            "mid": [
                "This case calls for structured support in the next intervention cycle.",
                "The student shows emerging pressure that merits timely follow-up.",
                "A focused check-in is recommended to prevent further drift.",
                "Early intervention is advised to stabilize engagement.",
                "This profile benefits from proactive, near-term advisor support.",
                "Moderate risk signals suggest a guided support response.",
                "A targeted follow-up plan should be initiated promptly.",
                "This case warrants caution and practical support planning.",
            ],
            "low_low": [
                "This student appears stable with preventive monitoring advised.",
                "Current indicators support a low-intensity observation approach.",
                "The case is suitable for routine follow-up and reinforcement.",
                "No immediate escalation is indicated; maintain structured monitoring.",
                "This profile aligns with stable observation and light-touch support.",
                "Continue continuity-focused monitoring for this student.",
                "A preventive check-in approach is appropriate at this stage.",
                "Signals suggest stability with ongoing progress reinforcement.",
            ],
        }

        if risk == "High" and urgency == "High":
            headline = pick(summary_openers["high_high"], offset=1)
        elif risk == "Low" and urgency == "Low":
            headline = pick(summary_openers["low_low"], offset=1)
        else:
            headline = pick(summary_openers["mid"], offset=1)

        signal_types = build_key_signals(row)
        signal_phrase_banks = {
            "attendance": [
                "Attendance trend reflects reduced consistency in recent periods.",
                "Recent attendance behaviour indicates weaker continuity.",
                "Attendance movement suggests emerging participation instability.",
                "Attendance-related signals point to reduced class continuity.",
                "Observed attendance pattern indicates decline in regularity.",
                "Attendance indicators show a softening engagement pattern.",
                "Attendance shifts suggest potential continuity risk.",
                "Class attendance trajectory appears less stable than expected.",
            ],
            "engagement": [
                "Engagement activity has softened relative to baseline.",
                "Recent participation markers show a downward movement.",
                "Engagement behaviour indicates lower active involvement.",
                "Platform engagement suggests reduced academic interaction.",
                "Engagement trend points to declining study participation.",
                "Recent engagement signals indicate weaker learner activation.",
                "In-course participation appears below recent norm.",
                "Behavioural engagement cues suggest reduced momentum.",
            ],
            "submissions": [
                "Submission patterns indicate rising completion pressure.",
                "Missed or delayed tasks suggest execution risk.",
                "Assignment completion behaviour reflects workflow strain.",
                "Task submission consistency has weakened in recent activity.",
                "Coursework delivery markers indicate elevated backlog risk.",
                "Incomplete assignment signals suggest reduced task follow-through.",
                "Submission timing patterns indicate potential academic slippage.",
                "Assessment completion behaviour points to schedule pressure.",
            ],
            "sentiment": [
                "Feedback tone indicates potential motivational pressure.",
                "Recent sentiment markers suggest emotional strain around study workload.",
                "Behavioural tone signals indicate lower confidence or morale.",
                "Sentiment cues suggest elevated stress in learning engagement.",
                "Recent feedback profile indicates caution for wellbeing-linked risk.",
                "Tone patterns suggest support may be needed to restore confidence.",
                "Feedback signals indicate possible pressure affecting persistence.",
                "Sentiment movement suggests a need for supportive contact.",
            ],
            "inactivity": [
                "Recent inactivity suggests reduced study momentum.",
                "Platform inactivity window indicates a continuity gap.",
                "Recency-of-engagement signals point to participation slowdown.",
                "Login activity pattern suggests lower recent academic touchpoints.",
                "Inactivity markers indicate reduced near-term course contact.",
                "Recent activity gaps suggest weaker routine engagement.",
                "Interaction cadence has slowed, signalling continuity risk.",
                "Lower recent activity suggests attention drift from coursework.",
            ],
            "anomaly": [
                "Recent anomaly indicators suggest behavioural deviation from expected pattern.",
                "Anomalous signal activity warrants additional advisor review.",
                "Behavioural anomaly cues indicate unusual short-term movement.",
                "Detected anomalies suggest an out-of-pattern engagement shift.",
                "Recent deviations in risk signals justify closer follow-up.",
                "Anomaly pattern indicates non-routine behavioural change.",
                "Outlier indicators suggest potential disruption in learning behaviour.",
                "Unusual signal movement was detected in recent periods.",
            ],
        }

        signal_bullets: list[str] = []
        for idx, signal_type in enumerate(signal_types):
            options = signal_phrase_banks.get(signal_type, [])
            line = pick(options, offset=10 + idx, fallback="")
            if _safe_text(line):
                signal_bullets.append(line)
        signals_block = "\n".join([f"• {s}" for s in signal_bullets[:4]])
        if not signals_block:
            signals_block = "• No strong signals available yet — monitoring recommended."

        signal_name_map = {
            "attendance": "attendance continuity",
            "engagement": "engagement behaviour",
            "submissions": "submission consistency",
            "sentiment": "feedback tone",
            "inactivity": "recent activity cadence",
            "anomaly": "behavioural anomaly movement",
        }
        signal_focus = ", ".join(signal_name_map[s] for s in signal_types[:2] if s in signal_name_map)
        if not signal_focus:
            signal_focus = "overall risk signals"

        recent_shift = _has_value(row, ["attendance_change", "rolling_delta", "sentiment_rolling_delta", "sudden_drop", "recent_anomalies"])
        persistent_pattern = _has_value(row, ["negative_streak", "historical_dropout_indicators", "dropout_history"])
        recent_clauses = [
            "The pattern appears to reflect a recent shift rather than a long-standing trajectory.",
            "Current evidence suggests a short-horizon change in behaviour.",
            "Signal timing indicates a near-term movement requiring quick stabilization.",
            "Recent movement across indicators points to a fresh transition phase.",
            "The observed profile is consistent with a newly emerging risk shift.",
            "Signal chronology suggests a recent change in learning behaviour.",
        ]
        persistent_clauses = [
            "Indicators also suggest persistence across multiple review periods.",
            "The pattern appears sustained rather than isolated.",
            "Signals indicate a recurring behavioural profile over time.",
            "Evidence points to a continuing pattern across recent cycles.",
            "Observed risk behaviour appears persistent across periods.",
            "The profile suggests continuity of risk markers beyond a single event.",
        ]
        neutral_clauses = [
            "Available indicators support a cautious interpretation pending further review.",
            "Current data suggests monitoring plus structured follow-up.",
            "Signals are directionally clear but should be validated in next cycle.",
            "The present profile supports targeted intervention without delay.",
            "Pattern strength is moderate and warrants practical action planning.",
            "Indicators suggest proactive support while monitoring trend continuation.",
        ]

        interp_templates = {
            "high_high": [
                "The combination of {focus} indicates elevated near-term interruption risk. Sentiment is {sentiment}, strengthening the case for immediate outreach.",
                "Converging signals in {focus} suggest acute retention pressure. Sentiment currently reads {sentiment}, so escalation should be immediate.",
                "Current movement in {focus} points to high short-term risk. With {sentiment} sentiment context, direct advisor action is recommended now.",
                "The profile across {focus} indicates high-intensity risk in the immediate window. Sentiment is {sentiment}, reinforcing rapid intervention.",
                "Signal convergence around {focus} suggests an urgent stability risk. Sentiment remains {sentiment}, supporting same-cycle escalation.",
                "Risk indicators in {focus} show a critical intervention window. Sentiment appears {sentiment}, indicating need for prompt contact.",
            ],
            "mid": [
                "The pattern across {focus} suggests growing pressure that can still be stabilized with timely support. Sentiment is {sentiment}.",
                "Signals in {focus} indicate an emerging risk profile that benefits from structured follow-up. Sentiment context is {sentiment}.",
                "Current evidence from {focus} points to moderate disengagement risk. Sentiment is {sentiment}, so practical guidance is advised.",
                "The profile in {focus} reflects developing strain rather than acute collapse. Sentiment appears {sentiment}, supporting guided intervention.",
                "Observed movement in {focus} suggests caution and active support planning. Sentiment currently reads {sentiment}.",
                "Indicators around {focus} show medium-intensity risk requiring proactive check-in. Sentiment remains {sentiment}.",
            ],
            "low_low": [
                "Signals in {focus} suggest manageable variation with no immediate escalation need. Sentiment is {sentiment}.",
                "The profile around {focus} appears stable, with preventive reinforcement appropriate. Sentiment remains {sentiment}.",
                "Current movement in {focus} is limited and suitable for monitoring-led support. Sentiment is {sentiment}.",
                "Indicators across {focus} point to low-intensity risk conditions. Sentiment context is {sentiment}.",
                "The observed profile in {focus} supports routine follow-up and continuity checks. Sentiment appears {sentiment}.",
                "Risk signals in {focus} remain comparatively stable. Sentiment is {sentiment}, consistent with light-touch monitoring.",
            ],
        }

        if risk == "High" and urgency == "High":
            interp_main = pick(interp_templates["high_high"], offset=20).format(focus=signal_focus, sentiment=sentiment.lower())
        elif risk == "Low" and urgency == "Low":
            interp_main = pick(interp_templates["low_low"], offset=20).format(focus=signal_focus, sentiment=sentiment.lower())
        else:
            interp_main = pick(interp_templates["mid"], offset=20).format(focus=signal_focus, sentiment=sentiment.lower())

        if persistent_pattern:
            interp_tail = pick(persistent_clauses, offset=30)
        elif recent_shift:
            interp_tail = pick(recent_clauses, offset=30)
        else:
            interp_tail = pick(neutral_clauses, offset=30)
        interpretation = f"{interp_main} {interp_tail}"

        contact_actions = [
            "Schedule a direct check-in call within 24 hours and confirm response.",
            "Initiate advisor outreach and lock a brief support conversation window.",
            "Contact the student for a focused check-in and document agreed next step.",
            "Arrange an immediate support touchpoint and confirm participation.",
            "Run a short advisor call to identify blockers and immediate commitments.",
            "Set up direct contact and verify communication channel reliability.",
            "Prioritize personal outreach and confirm check-in completion.",
            "Complete a same-cycle contact step with a clear follow-up timestamp.",
        ]
        academic_actions = [
            "Review the academic plan and define one high-impact completion target.",
            "Run a focused course progress review and set a one-week study checkpoint.",
            "Align coursework priorities and confirm a practical recovery sequence.",
            "Conduct an academic support review with actionable study milestones.",
            "Confirm pending course requirements and set immediate completion tasks.",
            "Set a targeted learning plan with a measurable next checkpoint.",
            "Prioritize backlog items and establish a short execution plan.",
            "Coordinate academic support resources and define weekly outcomes.",
        ]
        wellbeing_actions = [
            "Include a wellbeing check and offer referral support if pressure persists.",
            "Initiate psychological safety check-in and confirm support pathways.",
            "Provide access to wellbeing services and document acceptance of support.",
            "Add a wellbeing referral option and monitor emotional-risk response.",
            "Use a wellbeing-first follow-up and confirm safeguarding escalation route.",
            "Offer counsellor referral and track sentiment response in next review.",
            "Complete wellbeing screening step and capture follow-up ownership.",
            "Activate wellbeing support touchpoint alongside academic intervention.",
        ]
        monitoring_actions = [
            "Track weekly indicator movement and log update in the next review cycle.",
            "Maintain weekly monitoring and confirm status progression checkpoint.",
            "Run a scheduled follow-up review to verify signal stabilization.",
            "Monitor trend continuation and adjust support intensity if needed.",
            "Confirm next-cycle review and validate signal direction changes.",
            "Keep case under structured observation with documented weekly checks.",
            "Use routine monitoring cadence and escalate only on deterioration.",
            "Review indicator consistency in the next planned advisor cycle.",
        ]

        action_1 = pick(contact_actions, offset=40, fallback="Schedule a check-in")
        action_2 = pick(academic_actions, offset=50, fallback="Review recent engagement/attendance")
        if sentiment == "Negative" or urgency == "High":
            action_3 = pick(wellbeing_actions, offset=60, fallback="Confirm support needs and next steps")
        else:
            action_3 = pick(monitoring_actions, offset=60, fallback="Confirm support needs and next steps")
        actions = [a for a in [action_1, action_2, action_3] if _safe_text(a)]

        actions_block = "\n".join([f"• {a}" for a in actions[:3]])
        if not actions_block:
            actions_block = "\n".join(
                [
                    "• Schedule a check-in",
                    "• Review recent engagement/attendance",
                    "• Confirm support needs and next steps",
                ]
            )

        return "\n".join(
            [
                "Intervention Summary:",
                headline,
                "",
                "Key Signals Identified:",
                signals_block,
                "",
                "Interpretation:",
                interpretation,
                "",
                "Recommended Immediate Actions:",
                actions_block,
            ]
        )
    except Exception:
        return "\n".join(
            [
                "Intervention Summary:",
                "This student requires structured follow-up support.",
                "",
                "Key Signals Identified:",
                "• No strong signals available yet — monitoring recommended.",
                "",
                "Interpretation:",
                "Current indicators are incomplete, so a precautionary support check is advised.",
                "",
                "Recommended Immediate Actions:",
                "• Schedule a check-in",
                "• Review recent engagement/attendance",
                "• Confirm support needs and next steps",
            ]
        )


def render_detail(row: pd.Series) -> None:
    st.subheader(f"Student {_safe_text(row.get('student_id')) or 'N/A'}")
    st.markdown(
        f"Risk band: {_safe_text(row.get('risk_band')) or 'N/A'} | "
        f"Urgency: {_safe_text(row.get('urgency')) or 'N/A'} | "
        f"Sentiment: {_safe_text(row.get('sentiment_label')) or 'N/A'}"
    )
    st.markdown(build_intervention_summary(row))


def render_risk_urgency_info(df: pd.DataFrame) -> None:
    def has_any(cols: list[str]) -> bool:
        return any(col in df.columns for col in cols)

    risk_items: list[str] = []
    if has_any(["attendance_rate", "attendance_trend", "attendance_change"]):
        risk_items.append("Attendance trends")
    if has_any(["academic_performance", "grades", "grade", "gpa"]):
        risk_items.append("Academic performance")
    if has_any(["engagement_score", "lms_activity", "engagement_trend"]):
        risk_items.append("Engagement patterns")
    if has_any(["historical_dropout_indicators", "dropout_history", "dropout_risk"]):
        risk_items.append("Historical dropout signals")
    if not risk_items:
        risk_items = [
            "Attendance trends",
            "Academic performance",
            "Engagement patterns",
            "Historical dropout signals",
        ]

    urgency_items: list[str] = []
    if has_any(["recent_attendance_drop", "attendance_change", "attendance_trend"]):
        urgency_items.append("Sudden attendance decline")
    if has_any(["missed_assignments", "missed_tasks", "overdue_assignments"]):
        urgency_items.append("Missed or overdue assignments")
    if has_any(["negative_feedback_signals", "sentiment_label", "sentiment_score"]):
        urgency_items.append("Recent negative behavioural or feedback indicators")
    if has_any(["critical_deadline", "upcoming_deadlines", "deadline_days_left"]):
        urgency_items.append("Time-critical academic milestones")
    if not urgency_items:
        urgency_items = [
            "Sudden attendance decline",
            "Missed or overdue assignments",
            "Recent negative behavioural or feedback indicators",
            "Time-critical academic milestones",
        ]

    with st.expander("How Risk & Urgency Are Determined"):
        st.markdown("**Risk Band**")
        st.markdown(
            "Risk Band reflects a student's overall likelihood of disengagement or dropout based on historical "
            "academic and engagement patterns."
        )
        st.markdown("Risk Band is derived from key longitudinal indicators such as:")
        for item in risk_items:
            st.markdown(f"- {item}")

        st.markdown("**Urgency**")
        st.markdown(
            "Urgency indicates how quickly intervention is required based on recent changes or time-sensitive signals."
        )
        st.markdown("Urgency is triggered by short-term signals such as:")
        for item in urgency_items:
            st.markdown(f"- {item}")


def main() -> None:
    apply_theme()
    render_sidebar_branding()
    st.title("Action Queue")
    cfg = load_config(DEFAULT_CONFIG)
    df = load_queue(cfg)
    if df.empty:
        return

    feedback_df = load_feedback(cfg)
    df = ensure_sentiment_labels(df, student_id_col="student_id", feedback_df=feedback_df)
    render_risk_urgency_info(df)

    df_filtered = filter_df(df)
    display_cols = ["student_id", "risk_band", "urgency", "sentiment_label"]
    cols_present = [col for col in display_cols if col in df_filtered.columns]
    styled_df = df_filtered[cols_present].style.applymap(sentiment_style, subset=["sentiment_label"])
    st.dataframe(styled_df, use_container_width=True)

    if not df_filtered.empty and "student_id" in df_filtered.columns:
        st.markdown("Select a row to view details (first row shown by default).")
        selected_id = st.selectbox("Student", df_filtered["student_id"].tolist())
        row = df_filtered[df_filtered["student_id"] == selected_id].iloc[0]
        render_detail(row)


if __name__ == "__main__":
    main()
