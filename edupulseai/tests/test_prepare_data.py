"""Tests for the prepare_data script."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.prepare_data import prepare_data


def test_prepare_data_creates_splits_and_metadata(tmp_path: Path) -> None:
    base_dir = tmp_path
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = raw_dir / "students.csv"
    df = pd.DataFrame(
        {
            "student_id": list(range(20)),
            "dropout": [0, 1] * 10,
            "feature": list(range(20)),
        }
    )
    df.to_csv(dataset_path, index=False)

    config_dir = base_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        f"dataset:\n  path: {dataset_path}\n  target_column: dropout\n",
        encoding="utf-8",
    )

    prepare_data(config_path=config_path, base_dir=base_dir)

    processed_dir = base_dir / "data" / "processed"
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    assert train_path.exists()
    assert val_path.exists()
    assert test_path.exists()

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    assert len(train_df) == 14
    assert len(val_df) == 3
    assert len(test_df) == 3

    metadata_path = processed_dir / "metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["row_counts"] == {"train": 14, "val": 3, "test": 3}
    assert metadata["target_column"] == "dropout"
    assert metadata["source_path"] == str(dataset_path.resolve())
    assert metadata["target_distribution"]["overall"]["counts"] == {"0": 10, "1": 10}
