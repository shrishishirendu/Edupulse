"""Home page for EduPulse AI Streamlit app."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


@st.cache_data
def load_config(config_path: Path) -> dict:
    path = config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    st.set_page_config(page_title="EduPulse AI – Student Intervention System", layout="wide")
    st.title("EduPulse AI – Student Intervention System")
    st.sidebar.title("Navigation")

    cfg = load_config(DEFAULT_CONFIG)
    st.success("Configuration loaded.")
    st.json(cfg, expanded=False)
    st.markdown("Use the sidebar to navigate to Action Queue, Student Agent, or Daily Pack pages.")


if __name__ == "__main__":
    main()
