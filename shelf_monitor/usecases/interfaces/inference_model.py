"""Abstract interface for inference models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class InferenceModel(ABC):
    """Adapter interface that every inference engine must implement."""

    @property
    @abstractmethod
    def input_shape(self) -> tuple:
        """Return (C, H, W)."""

    @abstractmethod
    def infer(self, blob: np.ndarray) -> np.ndarray:
        """Run inference on a single pre-processed blob (1, C, H, W)."""

    def batch_infer(self, images: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Default: loop over single calls.  Override for true batching."""
        results = [self.infer(img[None]) for img in images]
        return np.concatenate(results, axis=0) if results else np.empty((0,))
