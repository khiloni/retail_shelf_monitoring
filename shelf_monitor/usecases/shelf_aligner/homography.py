"""Homography estimator (RANSAC) with validity checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ...frameworks.logging_config import get_logger
from .feature_matcher import MatchResult

logger = get_logger(__name__)


@dataclass
class HomographyResult:
    is_valid: bool
    matrix: Optional[np.ndarray] = None
    inlier_ratio: float = 0.0
    num_inliers: int = 0
    condition_number: float = 0.0


class HomographyEstimator:
    """Estimates a homography from feature matches and validates it."""

    def __init__(
        self,
        ransac_reproj_threshold: float = 5.0,
        min_inlier_ratio: float = 0.3,
        min_inliers: int = 10,
        max_iterations: int = 2000,
    ) -> None:
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.min_inliers = min_inliers
        self.max_iterations = max_iterations

    def estimate_homography(self, match_result: MatchResult) -> HomographyResult:
        if match_result.num_matches < 4:
            return HomographyResult(is_valid=False)

        H, mask = cv2.findHomography(
            match_result.query_points,
            match_result.ref_points,
            cv2.RANSAC,
            self.ransac_reproj_threshold,
            maxIters=self.max_iterations,
        )

        if H is None or mask is None:
            return HomographyResult(is_valid=False)

        num_inliers = int(mask.sum())
        inlier_ratio = num_inliers / max(1, match_result.num_matches)

        if num_inliers < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            return HomographyResult(is_valid=False, inlier_ratio=inlier_ratio)

        if not self._is_homography_valid(H):
            return HomographyResult(is_valid=False, inlier_ratio=inlier_ratio)

        cond = np.linalg.cond(H)

        return HomographyResult(
            is_valid=True,
            matrix=H,
            inlier_ratio=inlier_ratio,
            num_inliers=num_inliers,
            condition_number=float(cond),
        )

    def warp_image(
        self, image: np.ndarray, H: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        return cv2.warpPerspective(image, H, target_size)

    def _is_homography_valid(self, H: np.ndarray) -> bool:
        """Validate homography via determinant, perspective terms, and condition number."""
        det = np.linalg.det(H[:2, :2])
        if not (0.1 <= abs(det) <= 10.0):
            return False
        if abs(H[2, 0]) > 0.002 or abs(H[2, 1]) > 0.002:
            return False
        cond = np.linalg.cond(H)
        if cond > 10.0:
            return False
        return True
