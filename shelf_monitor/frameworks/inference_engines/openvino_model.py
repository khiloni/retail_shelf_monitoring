"""Stub for OpenVINO inference engine (optional dependency)."""
from __future__ import annotations

from pathlib import Path

from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


class OpenVINOEngine:
    """Wraps OpenVINO compiled model (optional extra)."""

    def __init__(self, model_path: str | Path, device: str = "CPU") -> None:
        try:
            from openvino.runtime import Core  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenVINO is not installed. Install with: pip install openvino"
            ) from exc

        path = Path(model_path)
        core = Core()
        model = core.read_model(str(path))
        self._compiled = core.compile_model(model, device)
        self._input = self._compiled.input(0)
        logger.info(f"OpenVINO engine loaded: {path.name} on {device}")

    @property
    def input_shape(self) -> tuple:
        return tuple(self._input.shape)[1:]  # (C, H, W)

    def infer(self, blob):
        import numpy as np
        result = self._compiled([blob])
        return list(result.values())[0]
