"""Home page for AxiGuide Streamlit app."""

from __future__ import annotations

import streamlit as st

from theme import apply_theme, render_sidebar_branding


def format_kpi(value: object, default: int) -> str:
    if value is None:
        value = default
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    st.set_page_config(page_title="AxiGuide", page_icon="\U0001F393", layout="wide")
    apply_theme()
    render_sidebar_branding()

    st.markdown(
        """
        <div class="ax-hero">
            <div>
                <div class="ax-hero-title">AxiGuide</div>
                <div class="ax-hero-subtitle">
                    Intelligent Early Warning &amp; Intervention Platform for<br />
                    Student Success
                </div>
            </div>
            <div class="ax-badge">Powered by iSoft</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ax-divider"></div>', unsafe_allow_html=True)

    kpi_students = format_kpi(st.session_state.get("students_monitored"), 1284)
    kpi_risk = format_kpi(st.session_state.get("high_risk_flagged"), 97)
    kpi_actions = format_kpi(st.session_state.get("actions_due_today"), 23)

    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        st.markdown(
            f"""
            <div class="ax-card ax-kpi">
                <div class="ax-kpi-label">Students Monitored</div>
                <div class="ax-kpi-value">{kpi_students}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_cols[1]:
        st.markdown(
            f"""
            <div class="ax-card ax-kpi">
                <div class="ax-kpi-label">High-Risk Flagged</div>
                <div class="ax-kpi-value">{kpi_risk}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_cols[2]:
        st.markdown(
            f"""
            <div class="ax-card ax-kpi">
                <div class="ax-kpi-label">Actions Due Today</div>
                <div class="ax-kpi-value">{kpi_actions}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="ax-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ax-section-title">Quick Actions</div>', unsafe_allow_html=True)

    action_cols = st.columns(3)
    with action_cols[0]:
        st.markdown(
            """
            <div class="ax-card">
                <strong>Review flagged students</strong>
                <div class="ax-action-desc">
                    Open the Action Queue to triage and assign interventions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Go to Action Queue", key="qa_action_queue"):
            if hasattr(st, "switch_page"):
                st.switch_page("pages/01_Action_Queue.py")

    with action_cols[1]:
        st.markdown(
            """
            <div class="ax-card">
                <strong>Launch Student Agent</strong>
                <div class="ax-action-desc">
                    Ask the agent for insights on a student profile.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Student Agent", key="qa_student_agent"):
            if hasattr(st, "switch_page"):
                st.switch_page("pages/02_Student_Agent.py")

    with action_cols[2]:
        st.markdown(
            """
            <div class="ax-card">
                <strong>Prepare Daily Pack</strong>
                <div class="ax-action-desc">
                    Generate a daily briefing for your success team.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View Daily Pack", key="qa_daily_pack"):
            if hasattr(st, "switch_page"):
                st.switch_page("pages/03_Daily_Pack.py")

    st.markdown('<div class="ax-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ax-section-title">How We Prioritise Students</div>',
        unsafe_allow_html=True,
    )

    prioritise_cols = st.columns(2)
    with prioritise_cols[0]:
        st.markdown(
            """
            <div class="ax-card">
                <strong>Risk Band</strong>
                <div class="ax-action-desc">
                    Risk Band reflects a student's overall likelihood of academic disengagement or dropout based on historical and behavioural patterns. It helps identify who may need structured, long-term support.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with prioritise_cols[1]:
        st.markdown(
            """
            <div class="ax-card">
                <strong>Urgency</strong>
                <div class="ax-action-desc">
                    Urgency indicates how quickly intervention is required based on recent changes, anomalies, or time-sensitive signals. It ensures critical cases are acted on immediately.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="ax-action-desc" style="margin-top:0.5rem;">
            <strong>
                Together, Risk Band defines severity, while Urgency defines immediacy — enabling smarter, faster intervention decisions.
            </strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ax-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ax-section-title">How AxiGuide Helps</div>', unsafe_allow_html=True)

    help_cols = st.columns(3)
    with help_cols[0]:
        st.markdown(
            """
            <div class="ax-card">
                <strong>Detect risk signals</strong>
                <div class="ax-action-desc">
                    Identify early indicators across attendance, grades, and engagement.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with help_cols[1]:
        st.markdown(
            """
            <div class="ax-card">
                <strong>Prioritise interventions</strong>
                <div class="ax-action-desc">
                    Focus resources on students who need timely, targeted support.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with help_cols[2]:
        st.markdown(
            """
            <div class="ax-card">
                <strong>Track outcomes</strong>
                <div class="ax-action-desc">
                    Monitor impact and close the loop on student success plans.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

