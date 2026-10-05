"""Abstract tracker interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ...entities.detection import Detection


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Detection]: ...

    @abstractmethod
    def predict(self) -> List[Detection]: ...

    @abstractmethod
    def reset(self) -> None: ...
