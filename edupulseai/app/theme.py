"""Shared AxiGuide theme helpers."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def apply_theme() -> None:
    """Inject AxiGuide theme CSS for all pages."""
    st.markdown(
        """
        <style>
        :root {
            --ax-bg: #2B0B10;
            --ax-surface: #3A1017;
            --ax-primary: #7B1E2B;
            --ax-accent: #C0392B;
            --ax-text: #FFFFFF;
            --ax-muted: #E6D6D8;
            --ax-border: rgba(192, 57, 43, 0.35);
        }
        .stApp {
            background: linear-gradient(180deg, var(--ax-bg) 0%, var(--ax-surface) 100%);
            color: var(--ax-text);
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: var(--ax-text);
        }
        .stApp p, .stApp li, .stApp span, .stApp label {
            color: var(--ax-muted);
        }
        [data-testid="stSidebar"] {
            background-color: var(--ax-bg);
        }
        [data-testid="stSidebar"] * {
            color: var(--ax-text);
        }
        [data-testid="stSidebarNav"] a {
            color: var(--ax-text) !important;
            padding: 0.35rem 0.5rem;
            border-radius: 8px;
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background-color: var(--ax-primary);
            color: var(--ax-text) !important;
        }
        section[data-testid="stSidebar"] h2 {
            display: none !important;
        }
        .ax-hero {
            background-color: var(--ax-surface);
            border: 1px solid var(--ax-border);
            border-radius: 18px;
            padding: 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 24px;
            flex-wrap: wrap;
        }
        .ax-hero-title {
            font-size: 52px;
            line-height: 1.05;
            font-weight: 700;
            color: var(--ax-text);
        }
        .ax-hero-subtitle {
            font-size: 19px;
            line-height: 1.4;
            color: var(--ax-muted);
            margin-top: 8px;
            max-width: 560px;
        }
        .ax-badge {
            background-color: var(--ax-primary);
            border: 1px solid var(--ax-accent);
            color: var(--ax-text);
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 12px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .ax-card {
            background-color: var(--ax-surface);
            border: 1px solid var(--ax-border);
            border-radius: 16px;
            padding: 18px;
            height: 100%;
        }
        .ax-kpi-label {
            font-size: 12px;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--ax-muted);
        }
        .ax-kpi-value {
            font-size: 28px;
            font-weight: 700;
            color: var(--ax-text);
            margin-top: 6px;
        }
        .ax-section-title {
            font-size: 22px;
            font-weight: 600;
            color: var(--ax-text);
            margin: 0 0 8px 0;
        }
        .ax-divider {
            height: 1px;
            background: var(--ax-border);
            margin: 20px 0;
        }
        .ax-action-desc {
            color: var(--ax-muted);
            margin-top: 6px;
        }
        .stButton > button {
            width: 100%;
            background-color: var(--ax-accent);
            border: 1px solid var(--ax-accent);
            color: var(--ax-text);
            border-radius: 10px;
            padding: 0.5rem 0.75rem;
            font-weight: 600;
        }
        .stButton > button:hover {
            background-color: var(--ax-primary);
            border-color: var(--ax-primary);
        }
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stSelectbox"] div[data-baseweb="select"],
        [data-testid="stMultiSelect"] div[data-baseweb="select"],
        [data-testid="stNumberInput"] input,
        [data-testid="stDateInput"] input {
            background-color: var(--ax-surface);
            color: var(--ax-text);
            border-color: var(--ax-border);
        }
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background-color: var(--ax-surface);
            border: 1px solid var(--ax-border);
            border-radius: 12px;
        }
        [data-testid="stMetric"] {
            background-color: var(--ax-surface);
            border: 1px solid var(--ax-border);
            border-radius: 12px;
            padding: 12px;
        }
        [data-testid="stExpander"] {
            background-color: var(--ax-surface);
            border: 1px solid var(--ax-border);
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_branding() -> None:
    logo_path = Path("app/assets/isoft_logo.png")
    st.sidebar.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
    if logo_path.is_file():
        st.sidebar.image(str(logo_path), width=150)
    else:
        st.sidebar.markdown("iSoft logo missing: `app/assets/isoft_logo.png`")
    st.sidebar.markdown("---")
