"""Model Registry — auto-loads trained models from the `models/` directory.

Contract filenames (all at models/ root):
  product_detector.pt       - YOLO11 fine-tuned weights (required)
  product_detector.onnx     - ONNX export (preferred for CPU inference)
  sku_embedding.pt          - EmbeddingNet state_dict
  sku_embedding.onnx        - ONNX export of embedding (optional)
  sku_index.faiss           - FAISS flat index
  sku_labels.json           - {idx -> {sku_id, name}} mapping
  model_manifest.json       - metadata validated by this module

After the developer unzips shelf_models.zip into models/, NO code changes are needed.
"""
from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from ..frameworks.exceptions import ModelCorruptError, ModelNotFoundError
from ..frameworks.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic manifest schema
# ---------------------------------------------------------------------------


class DetectorManifest(BaseModel):
    file: str = "product_detector.pt"
    onnx_file: str = "product_detector.onnx"
    imgsz: int = 640
    conf_default: float = 0.35
    iou_default: float = 0.45
    max_det: int = 300
    class_names: list[str] = Field(default_factory=lambda: ["product"])


class EmbeddingManifest(BaseModel):
    file: str = "sku_embedding.pt"
    onnx_file: str = "sku_embedding.onnx"
    input_size: int = 224
    dim: int = 256
    mean: list[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225])
    backbone: str = "mobilenet_v3_large"


class IndexManifest(BaseModel):
    file: str = "sku_index.faiss"
    labels_file: str = "sku_labels.json"
    metric: str = "inner_product"
    dim: int = 256
    num_vectors: int = 0
    similarity_threshold: float = 0.6


class MetricsManifest(BaseModel):
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    top1: float = 0.0
    top5: float = 0.0


class TrainingManifest(BaseModel):
    epochs_stage1: int = 0
    epochs_stage2: int = 0
    dataset_notes: str = ""


class ModelManifest(BaseModel):
    schema_version: str = "1.0"
    created_at: str = ""
    synthetic: bool = False
    detector: DetectorManifest = Field(default_factory=DetectorManifest)
    embedding: EmbeddingManifest = Field(default_factory=EmbeddingManifest)
    index: IndexManifest = Field(default_factory=IndexManifest)
    metrics: MetricsManifest = Field(default_factory=MetricsManifest)
    training: TrainingManifest = Field(default_factory=TrainingManifest)


# ---------------------------------------------------------------------------
# Operation mode
# ---------------------------------------------------------------------------


class ModelMode(str, Enum):
    BASELINE = "BASELINE"          # Only stock yolo11n.pt (COCO weights)
    POSITION_ONLY = "POSITION-ONLY"  # Detector OK, no embedding/index
    FULL = "FULL"                  # All models present


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BANNER_WIDTH = 72

CONTRACT_FILES = {
    "product_detector.pt",
    "product_detector.onnx",
    "sku_embedding.pt",
    "sku_embedding.onnx",
    "sku_index.faiss",
    "sku_labels.json",
    "model_manifest.json",
}


class ModelRegistry:
    """Resolves, validates, and provides paths to all model artifacts."""

    def __init__(self, models_dir: str | Path = "models", prefer_onnx: bool = True) -> None:
        self.models_dir = Path(models_dir)
        self.prefer_onnx = prefer_onnx
        self.manifest: Optional[ModelManifest] = None
        self.mode: ModelMode = ModelMode.BASELINE
        self.sku_labels: Dict[str, Dict[str, str]] = {}
        self._status_lines: list[str] = []

        self._discover()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detector_path(self) -> Path:
        """Return the preferred detector path (ONNX if present, else .pt)."""
        if self.prefer_onnx:
            onnx = self.models_dir / "product_detector.onnx"
            if onnx.exists():
                return onnx
        pt = self.models_dir / "product_detector.pt"
        if pt.exists():
            return pt
        # Fallback: stock YOLO weights (downloaded by ultralytics on first use)
        return Path("yolo11n.pt")

    def embedding_path(self) -> Optional[Path]:
        if self.mode != ModelMode.FULL:
            return None
        if self.prefer_onnx:
            onnx = self.models_dir / "sku_embedding.onnx"
            if onnx.exists():
                return onnx
        pt = self.models_dir / "sku_embedding.pt"
        return pt if pt.exists() else None

    def index_path(self) -> Optional[Path]:
        if self.mode != ModelMode.FULL:
            return None
        p = self.models_dir / "sku_index.faiss"
        return p if p.exists() else None

    def labels_path(self) -> Optional[Path]:
        if self.mode != ModelMode.FULL:
            return None
        p = self.models_dir / "sku_labels.json"
        return p if p.exists() else None

    def print_status(self) -> None:
        """Print a summary banner to stdout."""
        print("=" * _BANNER_WIDTH)
        print(f"  Model Registry  —  mode: {self.mode.value}")
        print("=" * _BANNER_WIDTH)
        for line in self._status_lines:
            print(line)
        print("=" * _BANNER_WIDTH)

    def status_table(self) -> list[dict]:
        """Return status as a list of dicts for the check-models CLI."""
        rows = []
        for fname in sorted(CONTRACT_FILES):
            p = self.models_dir / fname
            rows.append(
                {
                    "file": fname,
                    "found": p.exists(),
                    "size": f"{p.stat().st_size // 1024} KB" if p.exists() else "-",
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _discover(self) -> None:
        manifest_path = self.models_dir / "model_manifest.json"
        detector_pt = self.models_dir / "product_detector.pt"
        detector_onnx = self.models_dir / "product_detector.onnx"
        embedding_pt = self.models_dir / "sku_embedding.pt"
        embedding_onnx = self.models_dir / "sku_embedding.onnx"
        index_f = self.models_dir / "sku_index.faiss"
        labels_f = self.models_dir / "sku_labels.json"

        has_detector = detector_pt.exists() or detector_onnx.exists()
        has_embedding = embedding_pt.exists() or embedding_onnx.exists()
        has_index = index_f.exists()
        has_labels = labels_f.exists()

        # Load manifest
        if manifest_path.exists():
            try:
                with manifest_path.open() as f:
                    raw = json.load(f)
                self.manifest = ModelManifest.model_validate(raw)
                self._status_lines.append(f"  manifest      : OK (v{self.manifest.schema_version})")
            except Exception as exc:
                self._status_lines.append(f"  manifest      : CORRUPT ({exc})")
        else:
            self._status_lines.append("  manifest      : MISSING")

        # Determine mode
        if has_detector and has_embedding and has_index and has_labels:
            self.mode = ModelMode.FULL
            # Validate dim consistency
            if self.manifest:
                self._validate_dim_consistency()
        elif has_detector:
            self.mode = ModelMode.POSITION_ONLY
        else:
            self.mode = ModelMode.BASELINE

        # Load SKU labels
        if has_labels:
            try:
                with labels_f.open() as f:
                    self.sku_labels = json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to load sku_labels.json: {exc}")

        # Log status
        self._status_lines += [
            f"  detector      : {'OK' if has_detector else 'MISSING (using COCO yolo11n)'}",
            f"  embedding     : {'OK' if has_embedding else 'MISSING'}",
            f"  FAISS index   : {'OK' if has_index else 'MISSING'}",
            f"  SKU labels    : {'OK (%d SKUs)' % len(self.sku_labels) if has_labels else 'MISSING'}",
            f"  ─── MODE: {self.mode.value} ───",
        ]

        if self.mode == ModelMode.BASELINE:
            logger.warning(
                "⚠ BASELINE mode: no custom detector found. "
                "Using stock yolo11n.pt. Accuracy on dense shelves will be limited. "
                "Run on Colab/Kaggle to train and copy models/ into place."
            )
        elif self.mode == ModelMode.POSITION_ONLY:
            logger.warning(
                "⚠ POSITION-ONLY mode: detector OK but no embedding/index. "
                "SKU recognition disabled; MISPLACED alerts will not fire."
            )
        else:
            logger.info("✓ FULL mode: all models loaded.")

    def _validate_dim_consistency(self) -> None:
        """Check that embedding dim matches FAISS index dim."""
        if self.manifest is None:
            return
        emb_dim = self.manifest.embedding.dim
        idx_dim = self.manifest.index.dim
        if emb_dim != idx_dim:
            raise ModelCorruptError(
                f"Embedding dim ({emb_dim}) ≠ FAISS index dim ({idx_dim}). "
                "Rebuild the index with the current embedding model."
            )

    # ------------------------------------------------------------------
    # Getters with fallbacks
    # ------------------------------------------------------------------

    def get_imgsz(self) -> int:
        if self.manifest:
            return self.manifest.detector.imgsz
        return 640

    def get_conf(self) -> float:
        if self.manifest:
            return self.manifest.detector.conf_default
        return 0.35

    def get_iou(self) -> float:
        if self.manifest:
            return self.manifest.detector.iou_default
        return 0.45

    def get_max_det(self) -> int:
        if self.manifest:
            return self.manifest.detector.max_det
        return 300

    def get_class_names(self) -> list[str]:
        if self.manifest:
            return self.manifest.detector.class_names
        return ["product"]

    def get_embedding_dim(self) -> int:
        if self.manifest:
            return self.manifest.embedding.dim
        return 256

    def get_similarity_threshold(self) -> float:
        if self.manifest:
            return self.manifest.index.similarity_threshold
        return 0.6

    def get_mean_std(self) -> Tuple[list, list]:
        if self.manifest:
            return self.manifest.embedding.mean, self.manifest.embedding.std
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def sku_id_for_index(self, idx: int) -> str:
        """Return the sku_id string for a FAISS index row, or 'unknown_sku'."""
        entry = self.sku_labels.get(str(idx))
        if entry is None:
            return "unknown_sku"
        if isinstance(entry, dict):
            return entry.get("sku_id", f"sku_{idx}")
        return str(entry)
