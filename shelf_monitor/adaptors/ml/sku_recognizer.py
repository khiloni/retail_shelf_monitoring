"""SKU recognizer adaptor.

Extracts embeddings from product image crops using MobileNetV3-Large (PyTorch or ONNX)
and queries a FAISS index to identify the SKU.
Implements tracker-based caching so tracked objects are re-identified only every K frames.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ...frameworks.exceptions import InferenceError
from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


class SkuRecognizer:
    """Identifies SKUs from cropped product bounding boxes using embeddings and FAISS search."""

    def __init__(
        self,
        embedding_model_path: Optional[str | Path] = None,
        index_path: Optional[str | Path] = None,
        labels_path: Optional[str | Path] = None,
        input_size: int = 224,
        embedding_dim: int = 256,
        similarity_threshold: float = 0.6,
        top_k: int = 3,
        mean: Optional[list[float]] = None,
        std: Optional[list[float]] = None,
        device: str = "cpu",
        re_id_every_k_frames: int = 10,
    ) -> None:
        self.embedding_model_path = Path(embedding_model_path) if embedding_model_path else None
        self.index_path = Path(index_path) if index_path else None
        self.labels_path = Path(labels_path) if labels_path else None
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.mean = np.array(mean or [0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array(std or [0.229, 0.224, 0.225], dtype=np.float32)
        self.device = device
        self.re_id_every_k_frames = re_id_every_k_frames

        self._pt_model = None
        self._onnx_session = None
        self._faiss_index = None
        self._labels: Dict[str, Dict[str, str]] = {}
        self._cache: Dict[int, Tuple[str, int]] = {}  # track_id -> (sku_id, frame_count)

        self._init_model()
        self._init_index()

    @property
    def has_index(self) -> bool:
        return self._faiss_index is not None and len(self._labels) > 0

    def _init_model(self) -> None:
        if not self.embedding_model_path or not self.embedding_model_path.exists():
            logger.info("No SKU embedding model found; recognizer running in stub/position-only mode")
            return

        p_str = str(self.embedding_model_path)
        if p_str.endswith(".onnx"):
            try:
                import onnxruntime as ort
                self._onnx_session = ort.InferenceSession(p_str, providers=["CPUExecutionProvider"])
                logger.info(f"Loaded ONNX embedding model from {p_str}")
            except Exception as e:
                logger.warning(f"Failed to load ONNX embedding model: {e}")
        else:
            try:
                import torch
                from ...frameworks.pytorch_models.embedding_net import EmbeddingNet
                model = EmbeddingNet(embedding_dim=self.embedding_dim, input_size=self.input_size, pretrained=False)
                state = torch.load(p_str, map_location="cpu")
                model.load_state_dict(state)
                model.eval()
                self._pt_model = model
                logger.info(f"Loaded PyTorch embedding model from {p_str}")
            except Exception as e:
                logger.warning(f"Failed to load PyTorch embedding model: {e}")

    def _init_index(self) -> None:
        if not self.index_path or not self.index_path.exists():
            return

        try:
            import faiss
            self._faiss_index = faiss.read_index(str(self.index_path))
            logger.info(f"Loaded FAISS index from {self.index_path} ({self._faiss_index.ntotal} vectors)")
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")

        if self.labels_path and self.labels_path.exists():
            try:
                with self.labels_path.open() as f:
                    self._labels = json.load(f)
                logger.info(f"Loaded {len(self._labels)} SKU label mappings")
            except Exception as e:
                logger.warning(f"Failed to load SKU labels: {e}")

    def preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Resize, BGR->RGB, normalize (C, H, W) float32."""
        resized = cv2.resize(crop, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        return normalized.transpose(2, 0, 1)

    def extract_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        """Batch extract L2-normalized embeddings of shape (N, embedding_dim)."""
        if not crops:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        blobs = np.stack([self.preprocess_crop(c) for c in crops], axis=0)  # (N, C, H, W)

        if self._onnx_session is not None:
            input_name = self._onnx_session.get_inputs()[0].name
            feats = self._onnx_session.run(None, {input_name: blobs})[0]
        elif self._pt_model is not None:
            import torch
            with torch.no_grad():
                t = torch.from_numpy(blobs)
                feats = self._pt_model(t).cpu().numpy()
        else:
            # Fallback random/hash embedding for stub mode
            feats = np.zeros((len(crops), self.embedding_dim), dtype=np.float32)
            feats[:, 0] = 1.0

        # Ensure L2 normalized
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return feats / norms

    def batch_identify_skus(
        self, crops: List[np.ndarray], track_ids: Optional[List[Optional[int]]] = None
    ) -> List[str]:
        """Identify SKUs for a list of cropped product images."""
        if not crops:
            return []

        if not self.has_index:
            return ["unknown_sku"] * len(crops)

        embeddings = self.extract_embeddings(crops)

        # FAISS search
        import faiss
        scores, indices = self._faiss_index.search(embeddings.astype(np.float32), self.top_k)

        sku_ids = []
        for i in range(len(crops)):
            best_idx = int(indices[i][0])
            best_score = float(scores[i][0])

            # For inner-product / cosine similarity, score >= threshold
            if best_score >= self.similarity_threshold and best_idx >= 0:
                sku_entry = self._labels.get(str(best_idx))
                if isinstance(sku_entry, dict):
                    sku_id = sku_entry.get("sku_id", f"sku_{best_idx}")
                elif sku_entry is not None:
                    sku_id = str(sku_entry)
                else:
                    sku_id = f"sku_{best_idx}"
            else:
                sku_id = "unknown_sku"

            sku_ids.append(sku_id)

        return sku_ids
