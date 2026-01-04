"""Tests for dropout prediction reports and explanations."""

from __future__ import annotations

import pandas as pd

from edupulse.models.dropout import explain_top_drivers, make_prediction_report, train_model


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GPA": [2.5, 3.2, 1.8, 2.9, 3.5, 2.1],
            "AttendanceRate": [0.8, 0.95, 0.6, 0.85, 0.9, 0.55],
            "Program": ["A", "B", "A", "C", "B", "C"],
            "Target": ["Dropout", "Enrolled", "Graduate", "Dropout", "Enrolled", "Graduate"],
        }
    )


def _config() -> dict:
    return {
        "dataset": {
            "target_column": "Target",
            "numeric_columns": ["GPA", "AttendanceRate"],
            "categorical_columns": ["Program"],
            "ignore_columns": [],
        },
        "modeling": {
            "dropout_positive_labels": ["Dropout"],
            "dropout_negative_labels": ["Enrolled", "Graduate"],
            "random_state": 0,
            "decision_threshold": 0.5,
            "risk_thresholds": {"high": 0.75, "medium": 0.5},
        },
    }


def test_prediction_report_contains_expected_columns_and_bands() -> None:
    df = _sample_df()
    cfg = _config()

    model = train_model(df, cfg)
    report = make_prediction_report(model, df, cfg)

    expected_columns = ["Target", "y_true", "y_pred", "dropout_risk", "risk_band"]
    for col in expected_columns:
        assert col in report.columns

    unique_bands = set(report["risk_band"].unique())
    assert unique_bands.issubset({"low", "medium", "high"})


def test_explain_top_drivers_returns_non_empty_for_logreg_pipeline() -> None:
    df = _sample_df()
    cfg = _config()

    model = train_model(df, cfg)
    top_drivers = explain_top_drivers(model, cfg, top_k=3)

    assert not top_drivers.empty
    assert set(["feature", "coefficient", "direction", "abs_value"]).issubset(top_drivers.columns)
