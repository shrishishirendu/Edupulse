"""Tests for dropout baseline training."""

from __future__ import annotations

import pandas as pd

from edupulse.models.dropout import make_binary_target, train_model


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GPA": [2.5, 3.2, 1.8, 2.9, 3.5, 2.1],
            "AttendanceRate": [0.8, 0.95, 0.6, 0.85, 0.9, 0.55],
            "Program": ["A", "B", "A", "C", "B", "C"],
            "Target": ["Dropout", "Enrolled", "Graduate", "Dropout", "Enrolled", "Graduate"],
        }
    )


def test_train_model_predict_proba() -> None:
    df = _sample_dataframe()
    cfg = {
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
        },
    }

    y = make_binary_target(df, cfg)
    model = train_model(df, cfg)

    assert hasattr(model, "predict_proba")
    proba = model.predict_proba(df)
    preds = model.predict(df)

    assert proba.shape[0] == len(df)
    assert set(preds).issubset({0, 1})
    assert set(y.unique()) == {0, 1}


def test_infers_features_when_not_provided() -> None:
    df = _sample_dataframe()
    df["IgnoreMe"] = ["x"] * len(df)

    cfg = {
        "dataset": {
            "target_column": "Target",
            "numeric_columns": [],
            "categorical_columns": [],
            "ignore_columns": ["IgnoreMe"],
        },
        "modeling": {
            "dropout_positive_labels": ["Dropout"],
            "dropout_negative_labels": ["Enrolled", "Graduate"],
            "random_state": 123,
        },
    }

    model = train_model(df, cfg)
    assert hasattr(model, "predict_proba")

    proba = model.predict_proba(df)
    assert proba.shape == (len(df), 2)
