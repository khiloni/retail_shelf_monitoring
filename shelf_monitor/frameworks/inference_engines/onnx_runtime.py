"""ONNX Runtime inference engine."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


class ONNXRuntimeEngine:
    """Wraps an ONNX model for CPU (or CUDA) inference via onnxruntime."""

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        providers = ["CPUExecutionProvider"]
        if device in ("cuda", "gpu") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._session = ort.InferenceSession(str(path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._input_shape = tuple(self._session.get_inputs()[0].shape)  # e.g. (1,3,640,640)
        logger.info(f"ONNXRuntime loaded: {path.name}  providers={providers}")

    @property
    def input_shape(self) -> tuple:
        return self._input_shape[1:]  # drop batch dim → (C, H, W)

    def infer(self, blob: np.ndarray) -> np.ndarray:
        outputs = self._session.run(None, {self._input_name: blob})
        return outputs[0]

    def batch_infer(self, images: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        results = []
        for i in range(0, len(images), batch_size):
            batch = np.stack(images[i : i + batch_size], axis=0)
            out = self._session.run(None, {self._input_name: batch})[0]
            results.append(out)
        return np.concatenate(results, axis=0) if results else np.empty((0,))
