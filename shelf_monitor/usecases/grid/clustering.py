"""DBSCAN / KMeans clustering of detections into shelf rows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ClusterItem:
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    sku_id: str = "unknown_sku"
    confidence: float = 1.0
    track_id: Optional[int] = None


class RowClusterer:
    """Clusters items by y-centre into shelf rows."""

    def __init__(
        self,
        method: str = "dbscan",
        eps: float = 15.0,
        min_samples: int = 2,
        n_clusters: Optional[int] = None,
    ) -> None:
        self.method = method
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters = n_clusters

    def cluster_by_y_coordinate(
        self, items: List[ClusterItem]
    ) -> List[List[ClusterItem]]:
        """Return lists of items grouped into rows, sorted top-to-bottom."""
        if not items:
            return []

        y_coords = np.array([item.center[1] for item in items]).reshape(-1, 1)

        if self.method == "dbscan":
            labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(y_coords)
        elif self.method == "kmeans":
            k = self.n_clusters or max(1, len(items) // 4)
            k = min(k, len(items))
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(y_coords)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        # Group items by cluster label; discard DBSCAN noise (label == -1)
        clusters: dict[int, list[ClusterItem]] = {}
        for item, label in zip(items, labels):
            if label == -1:
                continue  # noise
            clusters.setdefault(label, []).append(item)

        if not clusters:
            # All items were noise — treat all as one row
            return [items]

        # Sort clusters by their mean y-coordinate (top to bottom)
        sorted_rows = sorted(
            clusters.values(),
            key=lambda row: np.mean([i.center[1] for i in row]),
        )
        return sorted_rows


class ItemSorter:
    """Sort items within a row left-to-right and assign sequential indices."""

    @staticmethod
    def assign_indices(items: List[ClusterItem]) -> List[Tuple[int, ClusterItem]]:
        sorted_items = sorted(items, key=lambda it: it.center[0])
        return list(enumerate(sorted_items))
