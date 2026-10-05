"""Application configuration with pydantic v2 validation.

Loaded from config.yaml (or config.example.yaml) in the CWD, then validated.
Any key can be overridden via environment variable:  SHELF_<SECTION>_<KEY>=value
(e.g. SHELF_ML_DEVICE=cuda).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings  # optional: falls back gracefully


# ---------------------------------------------------------------------------
# Sub-config sections
# ---------------------------------------------------------------------------

class AppSection(BaseModel):
    name: str = Field(default="Retail Shelf Monitor")
    debug: bool = Field(default=False)
    models_dir: str = Field(default="models")

    @field_validator("models_dir", mode="before")
    @classmethod
    def from_env(cls, v: str) -> str:
        return os.environ.get("SHELF_MODELS_DIR", v)


class DatabaseConfig(BaseModel):
    backend: Literal["sqlite", "postgres"] = Field(default="sqlite")
    # SQLite
    sqlite_path: str = Field(default="outputs/shelf_monitor.db")
    # PostgreSQL (optional)
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="retail_shelf_monitoring")
    user: str = Field(default="postgres")
    password: str = Field(default="postgres")

    @property
    def url(self) -> str:
        if self.backend == "sqlite":
            path = Path(self.sqlite_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite+aiosqlite:///{path}"
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class AlertsConfig(BaseModel):
    backend: Literal["memory", "redis"] = Field(default="memory")
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: Optional[str] = Field(default=None)


class MLConfig(BaseModel):
    """Object detection model configuration."""
    model_path: Optional[str] = Field(default=None, description="Custom .pt or .onnx path")
    inference_engine: Literal["ultralytics", "onnxruntime"] = Field(default="ultralytics")
    confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    iou: float = Field(default=0.45, ge=0.0, le=1.0)
    imgsz: int = Field(default=640)
    max_det: int = Field(default=300)
    device: str = Field(default="auto", description="'auto', 'cpu', 'cuda', 'cuda:0'")
    class_names: list[str] = Field(default_factory=lambda: ["product"])


class SKUConfig(BaseModel):
    """SKU recognition configuration."""
    embedding_model_path: Optional[str] = Field(default=None)
    index_path: Optional[str] = Field(default=None)
    labels_path: Optional[str] = Field(default=None)
    top_k: int = Field(default=3)
    similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    embedding_dim: int = Field(default=256)
    input_size: int = Field(default=224)
    re_id_every_k_frames: int = Field(default=10)


class GridConfig(BaseModel):
    clustering_method: Literal["dbscan", "kmeans"] = Field(default="dbscan")
    eps: float = Field(default=15.0, gt=0.0)
    min_samples: int = Field(default=2, ge=1)
    position_tolerance: int = Field(default=1, ge=0)


class AlignerConfig(BaseModel):
    single_shelf_mode: bool = Field(default=True, description="Skip alignment, use fixed shelf_id")
    fixed_shelf_id: str = Field(default="shelf_1")
    reference_dir: str = Field(default="data/reference_shelves")
    feature_type: Literal["orb", "sift"] = Field(default="orb")
    max_features: int = Field(default=5000)
    lowe_ratio: float = Field(default=0.75)
    ransac_reproj: float = Field(default=5.0)
    min_alignment_confidence: float = Field(default=0.3)


class StreamConfig(BaseModel):
    keyframe_interval: int = Field(default=15, ge=1)
    diff_threshold: float = Field(default=0.05, gt=0.0)
    single_shelf_mode: bool = Field(default=True)


class TrackingConfig(BaseModel):
    max_age: int = Field(default=30)
    min_hits: int = Field(default=3)
    iou_threshold: float = Field(default=0.3)


class ConsensusConfig(BaseModel):
    min_consecutive_frames: int = Field(default=3, ge=1)
    clear_after_frames: int = Field(default=2, ge=1)
    state_timeout_sec: int = Field(default=300)


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    format: Literal["text", "json"] = Field(default="text")
    file_path: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    app: AppSection = Field(default_factory=AppSection)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    sku: SKUConfig = Field(default_factory=SKUConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    aligner: AlignerConfig = Field(default_factory=AlignerConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    consensus: ConsensusConfig = Field(default_factory=ConsensusConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with p.open() as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "AppConfig":
        """Load config.yaml if present, else fall back to defaults."""
        try:
            return cls.from_yaml(path)
        except FileNotFoundError:
            return cls()

    @property
    def models_dir(self) -> Path:
        return Path(self.app.models_dir)
