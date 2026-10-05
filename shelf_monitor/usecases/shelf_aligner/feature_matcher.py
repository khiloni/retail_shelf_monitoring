"""ORB/SIFT feature matcher for shelf alignment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MatchResult:
    num_matches: int
    good_matches: List
    query_points: np.ndarray
    ref_points: np.ndarray
    ref_keypoints: List
    query_keypoints: List


class FeatureMatcher:
    """Detects ORB or SIFT keypoints and matches them with Lowe ratio test."""

    def __init__(
        self,
        feature_type: str = "orb",
        max_features: int = 5000,
        lowe_ratio: float = 0.75,
        min_matches: int = 10,
    ) -> None:
        self.lowe_ratio = lowe_ratio
        self.min_matches = min_matches

        if feature_type.lower() == "sift":
            self._detector = cv2.SIFT_create(nfeatures=max_features)
            self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            self._detector = cv2.ORB_create(nfeatures=max_features)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def precompute_reference_features(
        self, images: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        """Pre-compute and cache keypoints/descriptors for reference shelf images."""
        ref_features: Dict[str, Dict] = {}
        for shelf_id, img in images.items():
            kps, descs = self._detect_and_compute(img)
            if descs is not None and len(kps) >= self.min_matches:
                ref_features[shelf_id] = {
                    "image": img,
                    "keypoints": kps,
                    "descriptors": descs,
                }
                logger.debug(f"Precomputed {len(kps)} keypoints for shelf {shelf_id}")
        return ref_features

    def match_features(
        self,
        query_image: np.ndarray,
        ref_image: np.ndarray,
        ref_keypoints: List,
        ref_descriptors: np.ndarray,
    ) -> MatchResult:
        q_kps, q_descs = self._detect_and_compute(query_image)

        if q_descs is None or ref_descriptors is None:
            return MatchResult(
                num_matches=0,
                good_matches=[],
                query_points=np.empty((0, 2)),
                ref_points=np.empty((0, 2)),
                ref_keypoints=ref_keypoints,
                query_keypoints=q_kps,
            )

        matches_list = self._matcher.knnMatch(q_descs, ref_descriptors, k=2)
        good = [
            m
            for pair in matches_list
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < self.lowe_ratio * n.distance
        ]

        if len(good) < self.min_matches:
            return MatchResult(
                num_matches=len(good),
                good_matches=good,
                query_points=np.empty((0, 2)),
                ref_points=np.empty((0, 2)),
                ref_keypoints=ref_keypoints,
                query_keypoints=q_kps,
            )

        q_pts = np.float32([q_kps[m.queryIdx].pt for m in good])
        r_pts = np.float32([ref_keypoints[m.trainIdx].pt for m in good])

        return MatchResult(
            num_matches=len(good),
            good_matches=good,
            query_points=q_pts,
            ref_points=r_pts,
            ref_keypoints=ref_keypoints,
            query_keypoints=q_kps,
        )

    def _detect_and_compute(
        self, image: np.ndarray
    ) -> Tuple[List, Optional[np.ndarray]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        kps, descs = self._detector.detectAndCompute(gray, None)
        return kps, descs
