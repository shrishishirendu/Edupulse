"""Smoke tests for EduPulse package."""

from __future__ import annotations

from src import edupulse


def test_package_version() -> None:
    """Ensure package exposes a version constant."""
    assert edupulse.__version__ == "0.1.0"