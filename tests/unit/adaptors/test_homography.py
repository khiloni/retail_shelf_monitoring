"""Unit tests for HomographyEstimator."""
import numpy as np
from shelf_monitor.usecases.shelf_aligner.feature_matcher import MatchResult
from shelf_monitor.usecases.shelf_aligner.homography import HomographyEstimator


def test_homography_identity_estimation():
    estimator = HomographyEstimator(min_inlier_ratio=0.5, min_inliers=4)

    # 4 points matching identity
    pts = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float32)
    match_res = MatchResult(
        num_matches=4,
        good_matches=[1, 2, 3, 4],
        query_points=pts,
        ref_points=pts,
        ref_keypoints=[],
        query_keypoints=[],
    )

    res = estimator.estimate_homography(match_res)
    assert res.is_valid is True
    assert res.inlier_ratio == 1.0
    # Determinant of ~identity should be ~1
    det = np.linalg.det(res.matrix[:2, :2])
    assert 0.9 <= det <= 1.1


def test_homography_insufficient_points():
    estimator = HomographyEstimator()
    pts = np.array([[10, 10], [20, 20]], dtype=np.float32)
    match_res = MatchResult(
        num_matches=2,
        good_matches=[1, 2],
        query_points=pts,
        ref_points=pts,
        ref_keypoints=[],
        query_keypoints=[],
    )
    res = estimator.estimate_homography(match_res)
    assert res.is_valid is False
