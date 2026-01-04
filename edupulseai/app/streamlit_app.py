"""Streamlit MVP interface for EduPulse AI."""

from __future__ import annotations

import streamlit as st


def main() -> None:
    """Render the MVP placeholder UI."""
    st.set_page_config(page_title="EduPulse AI")
    st.title("EduPulse AI Dashboard")
    st.sidebar.header("Navigation")
    st.sidebar.info("Placeholder for dataset upload and analysis controls.")
    st.success("MVP scaffold ready")


if __name__ == "__main__":
    main()