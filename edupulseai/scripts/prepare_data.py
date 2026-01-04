"""Prepare stratified train/val/test splits for EduPulse AI datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edupulse.data.load import load_csv

DEFAULT_CONFIG_PATH = Path("configs") / "config.yaml"
SPLIT_SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare stratified dataset splits.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML (default: configs/config.yaml).",
    )
    return parser.parse_args()


def resolve_paths(config_path: Path | None, base_dir: Path | None) -> Tuple[Path, Path]:
    """Resolve base directory and config path, respecting absolute paths."""
    root_dir = Path(base_dir) if base_dir else PROJECT_ROOT
    cfg_path = config_path or DEFAULT_CONFIG_PATH
    cfg_path = cfg_path if cfg_path.is_absolute() else root_dir / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    return root_dir, cfg_path


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration from disk."""
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping.")
    return config


def dataset_settings(config: Dict[str, Any]) -> Tuple[Path, str]:
    """Extract dataset path and target column from the config."""
    dataset_cfg = config.get("dataset") or {}
    dataset_path = dataset_cfg.get("path")
    target_column = dataset_cfg.get("target_column")
    if not dataset_path:
        raise ValueError("Config missing dataset.path")
    if not target_column:
        raise ValueError("Config missing dataset.target_column")
    return Path(dataset_path), str(target_column)


def validate_target(df: pd.DataFrame, target_column: str) -> pd.Series:
    """Ensure the target column is present and usable for stratification."""
    if df.empty:
        raise ValueError("Dataset is empty; cannot create splits.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    target_series = df[target_column]
    if target_series.isnull().any():
        raise ValueError(f"Target column '{target_column}' contains null values; cannot stratify.")
    if target_series.nunique() < 2:
        raise ValueError(f"Target column '{target_column}' must contain at least two classes.")
    return target_series


def stratified_splits(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test splits with a 70/15/15 ratio."""
    target_series = validate_target(df, target_column)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=target_series,
        random_state=SPLIT_SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[target_column],
        random_state=SPLIT_SEED,
    )
    return train_df, val_df, test_df


def distribution(series: pd.Series) -> Dict[str, Dict[str, float | int]]:
    """Compute counts and proportions for a target series."""
    counts = series.value_counts(dropna=False)
    total = float(series.shape[0]) or 1.0
    return {
        "counts": {str(key): int(value) for key, value in counts.items()},
        "proportions": {str(key): float(value / total) for key, value in counts.items()},
    }


def write_metadata(
    processed_dir: Path,
    dataset_path: Path,
    target_column: str,
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Write metadata.json describing splits and target distribution."""
    metadata = {
        "source_path": str(dataset_path.resolve()),
        "target_column": target_column,
        "row_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "target_distribution": {
            "overall": distribution(full_df[target_column]),
            "train": distribution(train_df[target_column]),
            "val": distribution(val_df[target_column]),
            "test": distribution(test_df[target_column]),
        },
    }
    metadata_path = processed_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def prepare_data(config_path: Path | None = None, base_dir: Path | None = None) -> None:
    """Prepare data splits and metadata from the given configuration."""
    root_dir, resolved_config = resolve_paths(config_path, base_dir)
    config = load_config(resolved_config)
    dataset_path, target_column = dataset_settings(config)
    dataset_path = dataset_path if dataset_path.is_absolute() else root_dir / dataset_path

    processed_dir = root_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(dataset_path)
    train_df, val_df, test_df = stratified_splits(df, target_column)

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    write_metadata(processed_dir, dataset_path, target_column, df, train_df, val_df, test_df)

    print(
        f"Prepared dataset from {dataset_path} -> "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)} "
        f"(saved in {processed_dir})"
    )


def main() -> None:
    args = parse_args()
    prepare_data(config_path=args.config)


if __name__ == "__main__":
    main()
