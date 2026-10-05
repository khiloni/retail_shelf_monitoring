"""Optional TensorRT inference engine (extra dependency, not installed by default)."""
from __future__ import annotations

from typing import Any

import numpy as np

from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


class TensorRTInferenceEngine:
    """Thin wrapper around TensorRT — install tensorrt extra to use."""

    def __init__(self, engine_path: str, **kwargs: Any) -> None:
        try:
            import tensorrt  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "TensorRT is optional. Install with: pip install shelf-monitor[tensorrt] "
                "(or your platform-specific TensorRT wheel)."
            ) from exc
        self.engine_path = engine_path
        self._session = None
        logger.warning("TensorRT engine stub loaded; implement binding for your TRT version.")

    def predict(self, image: np.ndarray) -> Any:
        raise NotImplementedError(
            "TensorRT inference is optional. Use onnx_runtime (default) or ultralytics PyTorch."
        )
