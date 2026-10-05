"""Shelf alignment use case — matches keyframes against stored reference shelf images."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from ...entities.frame import Frame
from ...frameworks.logging_config import get_logger
from .feature_matcher import FeatureMatcher
from .homography import HomographyEstimator

logger = get_logger(__name__)


class ShelfAligner:
    """Aligns video frames against stored reference shelf images using ORB/SIFT + RANSAC."""

    def __init__(
        self,
        reference_dir: str | Path,
        feature_matcher: FeatureMatcher,
        homography_estimator: HomographyEstimator,
        min_alignment_confidence: float = 0.3,
        single_shelf_mode: bool = False,
        fixed_shelf_id: str = "shelf_1",
    ) -> None:
        self.reference_dir = Path(reference_dir)
        self.feature_matcher = feature_matcher
        self.homography_estimator = homography_estimator
        self.min_confidence = min_alignment_confidence
        self.single_shelf_mode = single_shelf_mode
        self.fixed_shelf_id = fixed_shelf_id

        self.reference_features: Dict[str, Dict] = {}
        if not self.single_shelf_mode:
            self.load_reference_shelves()

    def load_reference_shelves(self) -> None:
        """Load reference images from reference_dir and precompute features."""
        if not self.reference_dir.exists() or not self.reference_dir.is_dir():
            logger.warning(f"Reference directory not found: {self.reference_dir}")
            return

        images = {}
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for img_path in self.reference_dir.glob(ext):
                shelf_id = img_path.stem
                img = cv2.imread(str(img_path))
                if img is not None:
                    images[shelf_id] = img
                else:
                    logger.warning(f"Failed to read reference image: {img_path}")

        if images:
            self.reference_features = self.feature_matcher.precompute_reference_features(images)
            logger.info(f"Loaded and indexed {len(self.reference_features)} reference shelves")

    def align_to_best_reference(self, frame: Frame) -> Frame:
        """Match frame against references, update shelf_id and alignment metrics."""
        if self.single_shelf_mode:
            frame.shelf_id = self.fixed_shelf_id
            frame.alignment_confidence = 1.0
            frame.inlier_ratio = 1.0
            return frame

        if not self.reference_features:
            frame.shelf_id = self.fixed_shelf_id
            frame.alignment_confidence = 0.5
            frame.inlier_ratio = 0.5
            return frame

        best_shelf_id = None
        best_confidence = 0.0
        best_homography = None

        for shelf_id, ref_data in self.reference_features.items():
            match_res = self.feature_matcher.match_features(
                query_image=frame.frame_img,
                ref_image=ref_data["image"],
                ref_keypoints=ref_data["keypoints"],
                ref_descriptors=ref_data["descriptors"],
            )

            if match_res.num_matches < self.feature_matcher.min_matches:
                continue

            h_res = self.homography_estimator.estimate_homography(match_res)
            if not h_res.is_valid:
                continue

            if h_res.inlier_ratio > best_confidence:
                best_confidence = h_res.inlier_ratio
                best_shelf_id = shelf_id
                best_homography = h_res.matrix

        if best_shelf_id is not None and best_confidence >= self.min_confidence:
            frame.shelf_id = best_shelf_id
            frame.alignment_confidence = float(best_confidence)
            frame.inlier_ratio = float(best_confidence)
            logger.debug(f"Aligned frame to {best_shelf_id} (conf: {best_confidence:.2f})")
        else:
            # Fall back to fixed shelf id or None
            frame.shelf_id = self.fixed_shelf_id
            frame.alignment_confidence = float(best_confidence)
            frame.inlier_ratio = float(best_confidence)

        return frame
