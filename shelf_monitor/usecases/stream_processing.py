"""Stream processing usecase.

Orchestrates the entire real-time processing pipeline:
  Video/stream frame
    -> Keyframe selection (interval / scene change)
    -> Shelf alignment (ORB/SIFT homography vs reference images or single-shelf mode)
    -> YOLO product detection
    -> SKU recognition (embedding + FAISS)
    -> SORT tracking between keyframes
    -> Grid mapping & cell state computation (OK / EMPTY / MISPLACED / UNKNOWN)
    -> Temporal consensus (N consecutive frames)
    -> Alert generation / clearance
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from ..adaptors.keyframe_selector import KeyframeSelector
from ..entities.alert import Alert
from ..entities.detection import Detection
from ..entities.frame import Frame
from ..entities.planogram import Planogram
from ..frameworks.logging_config import get_logger
from .alert_generation import AlertGenerationUseCase
from .cell_state_computation import CellStateComputation
from .detection_processing import DetectionProcessingUseCase
from .interfaces.repositories import PlanogramRepository
from .interfaces.tracker import Tracker
from .shelf_aligner.shelf_aligner import ShelfAligner
from .temporal_consensus import TemporalConsensusManager

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    success: bool
    frame: Optional[Frame] = None
    detections: List[Detection] = field(default_factory=list)
    reason: Optional[str] = None


@dataclass
class ComplianceAnalysisResult:
    success: bool
    shelf_id: Optional[str] = None
    cell_states: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)
    summary: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


@dataclass
class StreamProcessingResult:
    success: bool
    frame: Optional[Frame] = None
    detections: List[Detection] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)
    cell_states: List[Dict[str, Any]] = field(default_factory=list)
    summary: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


class StreamProcessingUseCase:
    """End-to-end stream processor for retail shelf video/CCTV feeds."""

    def __init__(
        self,
        shelf_aligner: ShelfAligner,
        detection_processing: DetectionProcessingUseCase,
        planogram_repository: PlanogramRepository,
        tracker: Tracker,
        keyframe_selector: KeyframeSelector,
        cell_state_computation: CellStateComputation,
        temporal_consensus: TemporalConsensusManager,
        alert_generation: AlertGenerationUseCase,
        keyframe_interval: int = 15,
        single_shelf_mode: bool = True,
        fixed_shelf_id: str = "shelf_1",
    ) -> None:
        self.shelf_aligner = shelf_aligner
        self.detection_processing = detection_processing
        self.planogram_repository = planogram_repository
        self.tracker = tracker
        self.keyframe_selector = keyframe_selector
        self.cell_state_computation = cell_state_computation
        self.temporal_consensus = temporal_consensus
        self.alert_generation = alert_generation
        self.keyframe_interval = keyframe_interval
        self.single_shelf_mode = single_shelf_mode
        self.fixed_shelf_id = fixed_shelf_id

        self.frame_index: int = 0
        self._last_frame: Optional[Frame] = None
        self._planograms: Dict[str, Optional[Planogram]] = {}

    async def process_detections(
        self, frame_img: np.ndarray, frame_id: str, timestamp: datetime
    ) -> DetectionResult:
        frame = Frame(
            frame_id=frame_id,
            frame_img=frame_img,
            timestamp=timestamp,
        )

        # Decide if keyframe: periodic interval or scene difference
        is_interval = (self.frame_index % self.keyframe_interval == 0)
        frame = self.keyframe_selector.is_keyframe(frame)
        frame.is_keyframe = frame.is_keyframe or is_interval
        self.frame_index += 1

        if frame.is_keyframe:
            # 1. Align shelf
            frame = self.shelf_aligner.align_to_best_reference(frame)
            if not frame.shelf_id:
                frame.shelf_id = self.fixed_shelf_id

            # 2. Detect & identify SKUs
            detections = self.detection_processing.process_aligned_frame(frame)

            # 3. Update tracker
            if self.tracker and detections:
                detections = self.tracker.update(detections)

            self._last_frame = frame
        else:
            # Propagate detections using tracker on non-keyframes
            if self._last_frame is not None:
                frame.shelf_id = self._last_frame.shelf_id
                frame.alignment_confidence = self._last_frame.alignment_confidence
            else:
                frame.shelf_id = self.fixed_shelf_id

            if self.tracker:
                detections = self.tracker.predict()
            else:
                detections = []

            self._last_frame = frame

        return DetectionResult(success=True, frame=frame, detections=detections)

    async def analyze_compliance(
        self, shelf_id: str, detections: List[Detection], timestamp: datetime, frame_img: Optional[np.ndarray] = None
    ) -> ComplianceAnalysisResult:
        if shelf_id not in self._planograms or self._planograms.get(shelf_id) is None:
            planogram = await self.planogram_repository.get_by_shelf_id(shelf_id)
            if not planogram:
                # Fallback: check any registered planogram in repository
                all_planos = await self.planogram_repository.list_all()
                if all_planos:
                    planogram = all_planos[0]
            if not planogram:
                # Auto-seed if demo reference image exists
                from pathlib import Path
                ref_p = Path("data/demo/shelf_reference.jpg")
                if ref_p.exists():
                    try:
                        import cv2
                        from scripts.run_demo import ClassicalSyntheticDetector, ClassicalSyntheticSkuRecognizer
                        from .grid.grid_detector import GridDetector
                        ref_im = cv2.imread(str(ref_p))
                        if ref_im is not None:
                            d_syn = ClassicalSyntheticDetector().predict(ref_im)
                            c_syn = [ref_im[int(d.bbox.y1):int(d.bbox.y2), int(d.bbox.x1):int(d.bbox.x2)] for d in d_syn]
                            s_syn = ClassicalSyntheticSkuRecognizer().batch_identify_skus(c_syn)
                            d_dicts = [
                                {
                                    "bbox": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                                    "sku_id": s,
                                    "confidence": d.confidence,
                                    "class_id": 0,
                                }
                                for d, s in zip(d_syn, s_syn)
                            ]
                            gd, cparams = GridDetector(clustering_method="dbscan", eps=20.0, min_samples=2).detect_grid(d_dicts)
                            planogram = Planogram(
                                shelf_id=shelf_id,
                                reference_image_path=str(ref_p),
                                grid=gd,
                                clustering_params=cparams,
                                meta={"auto_seeded": True},
                            )
                            await self.planogram_repository.save(planogram)
                    except Exception as e:
                        logger.debug(f"Could not auto-seed planogram: {e}")

            self._planograms[shelf_id] = planogram

        planogram = self._planograms.get(shelf_id)
        if not planogram:
            return ComplianceAnalysisResult(
                success=True,
                shelf_id=shelf_id,
                cell_states=[],
                alerts=[],
                summary={
                    "total_cells": len(detections),
                    "empty_count": 0,
                    "misplaced_count": 0,
                    "fill_pct": 100.0 if detections else 0.0,
                    "compliance_pct": 100.0,
                },
                reason="no_planogram_found",
            )

        # 1. Compute cell states
        cs_result = self.cell_state_computation.compute_cell_states(
            planogram=planogram,
            detections=detections,
            frame_timestamp=timestamp,
        )

        # 2. Update temporal consensus
        consensus_res = self.temporal_consensus.update_cell_states(
            shelf_id=shelf_id,
            cell_state_updates=cs_result["cell_states"],
        )

        # 3. Generate new alerts with evidence images
        generated_alerts = []
        for alert_data in consensus_res["new_alerts"]:
            try:
                evidence_paths: List[str] = []
                if frame_img is not None and planogram is not None:
                    try:
                        ref_cell = planogram.grid.get_cell(alert_data["row_idx"], alert_data["item_idx"])
                        crop_bbox = [ref_cell.bbox.x1, ref_cell.bbox.y1, ref_cell.bbox.x2, ref_cell.bbox.y2] if ref_cell else None
                        ev_path = self.alert_generation.save_evidence_image(
                            frame_img=frame_img,
                            shelf_id=shelf_id,
                            row_idx=alert_data["row_idx"],
                            item_idx=alert_data["item_idx"],
                            alert_type=str(alert_data["alert_type"]),
                            crop_bbox=crop_bbox,
                        )
                        if ev_path:
                            evidence_paths.append(ev_path)
                    except Exception as ev_err:
                        logger.debug(f"Failed to save evidence crop: {ev_err}")

                alert = await self.alert_generation.generate_alert(
                    alert_data,
                    evidence_paths=evidence_paths if evidence_paths else None,
                )
                generated_alerts.append(alert)
            except Exception as e:
                logger.error(f"Error generating alert: {e}")

        # 4. Clear resolved alerts
        for cell_info in consensus_res["cleared_alerts"]:
            try:
                await self.alert_generation.clear_cell_alerts(
                    shelf_id=cell_info["shelf_id"],
                    row_idx=cell_info["row_idx"],
                    item_idx=cell_info["item_idx"],
                )
            except Exception as e:
                logger.error(f"Error clearing alert: {e}")

        return ComplianceAnalysisResult(
            success=True,
            shelf_id=shelf_id,
            cell_states=cs_result["cell_states"],
            alerts=generated_alerts,
            summary=cs_result["summary"],
        )

    async def process_frame(
        self, frame_img: np.ndarray, frame_id: str, timestamp: Optional[datetime] = None
    ) -> StreamProcessingResult:
        ts = timestamp or datetime.now(timezone.utc)
        det_res = await self.process_detections(frame_img, frame_id, ts)

        if not det_res.success:
            return StreamProcessingResult(
                success=False, frame=det_res.frame, reason=det_res.reason
            )

        shelf_id = det_res.frame.shelf_id or self.fixed_shelf_id
        comp_res = await self.analyze_compliance(
            shelf_id=shelf_id,
            detections=det_res.detections,
            timestamp=ts,
            frame_img=frame_img,
        )

        return StreamProcessingResult(
            success=True,
            frame=det_res.frame,
            detections=det_res.detections,
            alerts=comp_res.alerts if comp_res.success else [],
            cell_states=comp_res.cell_states if comp_res.success else [],
            summary=comp_res.summary if comp_res.success else None,
            reason=comp_res.reason if not comp_res.success else None,
        )
