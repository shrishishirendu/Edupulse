"""Configuration models for EduPulse AI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class DataPaths(BaseModel):
    """Paths to structured datasets."""

    raw: Path = Field(default=Path("data/raw"))
    processed: Path = Field(default=Path("data/processed"))


class ModelPaths(BaseModel):
    """Locations for serialized model assets."""

    artifacts: Path = Field(default=Path("models/artifacts"))


class AppConfig(BaseModel):
    """Top-level application configuration."""

    name: str = Field(default="EduPulse AI")
    data: DataPaths = Field(default_factory=DataPaths)
    models: ModelPaths = Field(default_factory=ModelPaths)
    environment: Optional[str] = None


DEFAULT_CONFIG = AppConfig()