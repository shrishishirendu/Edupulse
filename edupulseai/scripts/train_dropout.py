"""Train and evaluate the dropout baseline model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edupulse.models.dropout import (
    evaluate,
    explain_top_drivers,
    load_processed_splits,
    make_binary_target,
    make_prediction_report,
    train_model,
)

DEFAULT_CONFIG_PATH = Path("configs") / "config.yaml"
ARTIFACTS_DIR = Path("models") / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the dropout baseline model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML (default: configs/config.yaml).",
    )
    return parser.parse_args()


def resolve_config_path(config_path: Path) -> Path:
    cfg_path = config_path
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    return cfg_path


def load_config(config_path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("Missing dependency: PyYAML. Add PyYAML to requirements.txt and redeploy.")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping.")
    return data


def print_metrics(label: str, metrics: Dict[str, float]) -> None:
    pretty = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items() if value == value)
    print(f"{label} -> {pretty}")


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    train_df, val_df, test_df = load_processed_splits(processed_dir)

    dataset_cfg = config.get("dataset") or {}
    id_column = dataset_cfg.get("id_column")
    if id_column:
        for df in (val_df, test_df):
            if id_column not in df.columns:
                df[id_column] = range(1, len(df) + 1)
        val_ids = val_df[id_column].reset_index(drop=True)
        test_ids = test_df[id_column].reset_index(drop=True)
    else:
        val_ids = None
        test_ids = None

    model = train_model(train_df, config)

    y_val = make_binary_target(val_df, config)
    val_metrics = evaluate(model, val_df, y_val)

    y_test = make_binary_target(test_df, config)
    test_metrics = evaluate(model, test_df, y_test)

    artifacts_dir = PROJECT_ROOT / ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "dropout_model.joblib"
    metrics_path = artifacts_dir / "dropout_metrics.json"

    joblib.dump(model, model_path)
    metrics_payload = {"val": val_metrics, "test": test_metrics}
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    val_report = make_prediction_report(model, val_df, config)
    test_report = make_prediction_report(model, test_df, config)

    if id_column and id_column not in val_report.columns and val_ids is not None:
        val_report.insert(0, id_column, val_ids)
    if id_column and id_column not in test_report.columns and test_ids is not None:
        test_report.insert(0, id_column, test_ids)

    val_report.to_csv(reports_dir / "dropout_val_predictions.csv", index=False)
    test_report.to_csv(reports_dir / "dropout_test_predictions.csv", index=False)

    top_drivers = explain_top_drivers(model, config)
    top_drivers.to_csv(reports_dir / "dropout_top_drivers.csv", index=False)

    print("Dropout baseline training complete.")
    print_metrics("Validation", val_metrics)
    print_metrics("Test", test_metrics)

    band_counts = test_report["risk_band"].value_counts()
    low_count = int(band_counts.get("low", 0))
    medium_count = int(band_counts.get("medium", 0))
    high_count = int(band_counts.get("high", 0))
    print(f"Test risk bands -> low={low_count}, medium={medium_count}, high={high_count}")

    increasing = top_drivers[top_drivers["direction"] == "increases_dropout_risk"].head(5)
    inc_features = ", ".join(increasing["feature"].tolist())
    print(f"Top increasing-risk features: {inc_features}")


if __name__ == "__main__":
    main()
