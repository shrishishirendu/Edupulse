"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

LoadFormat = Literal["csv", "parquet"]


def _validate_path(path: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset not found at: {resolved}")
    return resolved


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV dataset from disk with a helpful missing-file error."""
    resolved = _validate_path(path)
    return pd.read_csv(resolved)


def load_dataset(path: Path, fmt: LoadFormat = "csv") -> pd.DataFrame:
    """Load a dataset from disk into a DataFrame."""
    if fmt == "csv":
        return load_csv(path)
    if fmt == "parquet":
        resolved = _validate_path(path)
        return pd.read_parquet(resolved)
    msg = f"Unsupported format: {fmt}"
    raise ValueError(msg)
