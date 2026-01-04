"""Dropout baseline model utilities for binary early-warning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_PROCESSED_DIR = Path("data") / "processed"


def load_processed_splits(processed_dir: str | Path = DEFAULT_PROCESSED_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test CSVs from the processed directory."""
    base = Path(processed_dir)
    train_path = base / "train.csv"
    val_path = base / "val.csv"
    test_path = base / "test.csv"

    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing processed split: {path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


def _dataset_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dataset = cfg.get("dataset") or {}
    if not isinstance(dataset, dict):
        raise ValueError("config['dataset'] must be a mapping.")
    return dataset


def _target_column(cfg: Dict[str, Any]) -> str:
    target = _dataset_cfg(cfg).get("target_column")
    if not target:
        raise ValueError("Config missing dataset.target_column")
    return str(target)


def _ignore_columns(cfg: Dict[str, Any]) -> List[str]:
    ignore = _dataset_cfg(cfg).get("ignore_columns") or []
    return [str(col) for col in ignore]


def _labels(cfg: Dict[str, Any]) -> Tuple[Sequence[Any], Sequence[Any]]:
    modeling_cfg = cfg.get("modeling") or {}
    positive = modeling_cfg.get("dropout_positive_labels") or _dataset_cfg(cfg).get("target_positive_values") or ["Dropout"]
    negative = modeling_cfg.get("dropout_negative_labels") or _dataset_cfg(cfg).get("target_negative_values") or [
        "Enrolled",
        "Graduate",
    ]
    return positive, negative


def _resolve_feature_columns(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    dataset_cfg = _dataset_cfg(cfg)
    target_column = _target_column(cfg)
    ignore_columns = _ignore_columns(cfg)

    def _clean_columns(columns: Sequence[str]) -> List[str]:
        return [str(col) for col in columns if col not in {target_column, *ignore_columns}]

    numeric_cols = _clean_columns(dataset_cfg.get("numeric_columns") or [])
    categorical_cols = _clean_columns(dataset_cfg.get("categorical_columns") or [])

    remaining_columns: Iterable[str] = (col for col in df.columns if col not in {target_column, *ignore_columns})

    if not numeric_cols and not categorical_cols:
        inferred_numeric: List[str] = []
        inferred_categorical: List[str] = []
        for col in remaining_columns:
            if is_numeric_dtype(df[col]):
                inferred_numeric.append(col)
            else:
                inferred_categorical.append(col)
        numeric_cols = inferred_numeric
        categorical_cols = inferred_categorical
    elif not numeric_cols:
        numeric_cols = [col for col in remaining_columns if is_numeric_dtype(df[col]) and col not in categorical_cols]
    elif not categorical_cols:
        categorical_cols = [col for col in remaining_columns if not is_numeric_dtype(df[col]) and col not in numeric_cols]

    categorical_cols = [col for col in categorical_cols if col not in numeric_cols]

    missing = [col for col in [*numeric_cols, *categorical_cols] if col not in df.columns]
    if missing:
        raise ValueError(f"Configured feature columns not found in DataFrame: {missing}")

    if not numeric_cols and not categorical_cols:
        raise ValueError("No feature columns available after applying ignore/target filters.")
    return numeric_cols, categorical_cols


def make_binary_target(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """Convert multi-class Target to binary dropout label (1=Dropout, 0=Enrolled/Graduate)."""
    target_column = _target_column(cfg)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    positives, negatives = _labels(cfg)
    target_series = df[target_column]

    positive_mask = target_series.isin(positives)
    negative_mask = target_series.isin(negatives)
    known_mask = positive_mask | negative_mask

    if not known_mask.all():
        unknown_values = target_series[~known_mask].unique()
        raise ValueError(f"Unexpected target labels encountered: {unknown_values}")

    binary_target = positive_mask.astype(int)
    binary_target.name = target_column
    return binary_target


def build_preprocessor(
    cfg: Dict[str, Any],
    numeric_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
) -> ColumnTransformer:
    """Create preprocessing transformer with one-hot encoding for categorical features."""
    dataset_cfg = _dataset_cfg(cfg)
    numeric_cols = list(numeric_columns or dataset_cfg.get("numeric_columns") or [])
    categorical_cols = list(categorical_columns or dataset_cfg.get("categorical_columns") or [])

    if not numeric_cols and not categorical_cols:
        raise ValueError("At least one feature column is required to build the preprocessor.")

    transformers = []
    if categorical_cols:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
    if numeric_cols:
        transformers.append(("numeric", Pipeline([("scaler", StandardScaler())]), numeric_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_model(train_df: pd.DataFrame, cfg: Dict[str, Any]) -> Pipeline:
    """Fit the dropout baseline model and return the trained pipeline."""
    numeric_cols, categorical_cols = _resolve_feature_columns(train_df, cfg)
    preprocessor = build_preprocessor(cfg, numeric_columns=numeric_cols, categorical_columns=categorical_cols)

    modeling_cfg = cfg.get("modeling") or {}
    classifier = LogisticRegression(
        max_iter=5000,
        solver="saga",
        class_weight="balanced",
        random_state=modeling_cfg.get("random_state", None),
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    y_train = make_binary_target(train_df, cfg)
    pipeline.fit(train_df, y_train)
    return pipeline


def evaluate(model: Pipeline, df: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
    """Compute evaluation metrics for the given model and dataset."""
    metrics: Dict[str, float] = {}
    y_pred = model.predict(df)

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    if hasattr(model, "predict_proba"):
        try:
            y_scores = model.predict_proba(df)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


def get_risk_scores(model: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Return dropout probability scores from a predict_proba-capable model."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support predict_proba; cannot compute risk scores.")
    scores = model.predict_proba(df)[:, 1]
    return pd.Series(scores, index=df.index, name="dropout_risk")


def assign_risk_band(p: float, thresholds: Dict[str, float]) -> str:
    """Assign risk band given probability and configured thresholds."""
    high = thresholds.get("high")
    medium = thresholds.get("medium")
    if high is None or medium is None:
        raise ValueError("Risk thresholds must include 'high' and 'medium' keys.")
    if p >= high:
        return "high"
    if p >= medium:
        return "medium"
    return "low"


def make_prediction_report(model: Pipeline, df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Build a prediction report with risk scores, bands, and binary labels."""
    dataset_cfg = _dataset_cfg(cfg)
    modeling_cfg = cfg.get("modeling") or {}

    target_column = _target_column(cfg)
    id_column = dataset_cfg.get("id_column")

    decision_threshold = modeling_cfg.get("decision_threshold", 0.5)
    risk_thresholds = modeling_cfg.get("risk_thresholds") or {"high": 0.75, "medium": 0.5}

    y_true = make_binary_target(df, cfg)
    proba = get_risk_scores(model, df)
    y_pred = (proba >= decision_threshold).astype(int)
    risk_bands = proba.apply(lambda p: assign_risk_band(p, risk_thresholds))

    data = {
        target_column: df[target_column],
        "y_true": y_true,
        "y_pred": y_pred,
        "dropout_risk": proba,
        "risk_band": risk_bands,
    }

    columns = []
    if id_column and id_column in df.columns:
        columns.append(id_column)
        data[id_column] = df[id_column]

    columns.extend([target_column, "y_true", "y_pred", "dropout_risk", "risk_band"])
    report = pd.DataFrame(data, columns=columns)
    return report


def explain_top_drivers(model: Pipeline, cfg: Dict[str, Any], top_k: int = 15) -> pd.DataFrame:
    """Return top positive/negative drivers from a trained LogisticRegression pipeline."""
    if not isinstance(model, Pipeline):
        raise ValueError("Model must be a sklearn Pipeline with preprocessor and classifier steps.")

    preprocessor = model.named_steps.get("preprocessor")
    classifier = model.named_steps.get("classifier")

    if preprocessor is None or classifier is None:
        raise ValueError("Pipeline must include 'preprocessor' and 'classifier' steps.")
    if not hasattr(preprocessor, "get_feature_names_out"):
        raise ValueError("Preprocessor must support get_feature_names_out to derive feature names.")
    if not isinstance(classifier, LogisticRegression):
        raise ValueError("Classifier must be an instance of LogisticRegression.")
    if not hasattr(classifier, "coef_"):
        raise ValueError("Classifier is not fitted; coefficients unavailable.")

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    drivers = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )
    drivers["abs_value"] = drivers["coefficient"].abs()
    drivers["direction"] = drivers["coefficient"].apply(
        lambda coef: "increases_dropout_risk" if coef > 0 else "decreases_dropout_risk"
    )

    positive = drivers[drivers["coefficient"] > 0].nlargest(top_k, "abs_value")
    negative = drivers[drivers["coefficient"] < 0].nlargest(top_k, "abs_value")

    combined = pd.concat([positive, negative]).sort_values("abs_value", ascending=False)
    return combined.reset_index(drop=True)


@dataclass
class DropoutRiskModel:
    """Lightweight wrapper providing predict/predict_proba on a trained pipeline."""

    model: Pipeline | None = None

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict dropout risk scores (probabilities if available)."""
        if self.model is None:
            raise ValueError("Model is not trained; call 'fit' first or load a trained pipeline.")
        if hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba(features)[:, 1]
            return pd.Series(scores, index=features.index, name="dropout_risk")
        predictions = self.model.predict(features)
        return pd.Series(predictions, index=features.index, name="dropout_label")

    def predict_proba(self, features: pd.DataFrame):
        """Predict dropout probabilities using the wrapped pipeline."""
        if self.model is None:
            raise ValueError("Model is not trained; call 'fit' first or load a trained pipeline.")
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Wrapped model does not support predict_proba.")
        return self.model.predict_proba(features)

    def fit(self, df: pd.DataFrame, cfg: Dict[str, Any]) -> "DropoutRiskModel":
        """Train the internal model using the provided DataFrame and config."""
        self.model = train_model(df, cfg)
        return self

    def explain(self, features: pd.DataFrame) -> List[Dict[str, float]]:
        """Return simple probability-based explanations for compatibility."""
        if self.model is None:
            raise ValueError("Model is not trained; call 'fit' first or load a trained pipeline.")
        if hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba(features)[:, 1]
            return [{"dropout_probability": float(score)} for score in scores]
        predictions = self.model.predict(features)
        return [{"predicted_label": float(value)} for value in predictions]
