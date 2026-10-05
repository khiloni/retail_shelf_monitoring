"""YOLO product detector adaptor.

Wraps either Ultralytics YOLO or ONNXRuntime for inference.
Predicts bounding boxes with single class "product" (class_id 0).
Supports imgsz, conf, iou, max_det (>=300), and auto CPU/GPU fallback.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from ...entities.common import BoundingBox
from ...entities.detection import Detection
from ...frameworks.exceptions import InferenceError, ModelNotFoundError
from ...frameworks.logging_config import get_logger

logger = get_logger(__name__)


class YoloDetector:
    """Product detector using YOLO11 (ultralytics PyTorch or ONNX Runtime)."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        imgsz: int = 640,
        max_det: int = 300,
        device: str = "auto",
        engine_type: str = "ultralytics",
    ) -> None:
        self.conf = confidence_threshold
        self.iou = nms_threshold
        self.imgsz = imgsz
        self.max_det = max_det
        self.device = self._resolve_device(device)
        self.engine_type = engine_type
        self.model_path = Path(model_path) if model_path else None
        self._model = None
        self._onnx_session = None

        self._init_engine()

    def _resolve_device(self, dev: str) -> str:
        if dev == "auto":
            try:
                import torch
                return "0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return dev

    def _init_engine(self) -> None:
        p_str = str(self.model_path) if self.model_path else "yolo11n.pt"

        # Check if ONNX model was specified
        if p_str.endswith(".onnx") or self.engine_type == "onnxruntime":
            if self.model_path and not self.model_path.exists():
                raise ModelNotFoundError(
                    f"ONNX model file not found: {self.model_path}. "
                    "Run demo mode or train on Colab/Kaggle and copy models/ into place."
                )
            try:
                import onnxruntime as ort
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self.device != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers()
                    else ["CPUExecutionProvider"]
                )
                self._onnx_session = ort.InferenceSession(p_str, providers=providers)
                self.engine_type = "onnxruntime"
                logger.info(f"Initialized ONNX YOLO detector from {p_str} on {providers[0]}")
                return
            except Exception as e:
                logger.warning(f"Failed to load ONNX session: {e}. Falling back to ultralytics.")

        # Fallback to ultralytics YOLO
        try:
            from ultralytics import YOLO
            self._model = YOLO(p_str)
            self.engine_type = "ultralytics"
            logger.info(f"Initialized Ultralytics YOLO detector from {p_str} on device {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {p_str}: {e}")
            raise ModelNotFoundError(
                f"Could not load YOLO model: {e}. "
                "Ensure ultralytics is installed or download model weights."
            )

    def predict(self, frame_img: np.ndarray) -> List[Detection]:
        """Run detection on an image (BGR numpy array)."""
        if frame_img is None or frame_img.size == 0:
            return []

        if self.engine_type == "onnxruntime" and self._onnx_session is not None:
            return self._predict_onnx(frame_img)
        else:
            return self._predict_ultralytics(frame_img)

    def _predict_ultralytics(self, frame_img: np.ndarray) -> List[Detection]:
        try:
            results = self._model.predict(
                source=frame_img,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                max_det=self.max_det,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise InferenceError(f"YOLO inference failed: {e}")

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        now = datetime.now(timezone.utc)
        for i in range(len(boxes)):
            box = boxes[i]
            conf = float(confs[i])
            cls_id = int(classes[i])
            detections.append(
                Detection(
                    detection_id=str(uuid.uuid4()),
                    shelf_id="unassigned",
                    frame_timestamp=now,
                    bbox=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                    class_id=cls_id,
                    sku_id=f"sku_{cls_id}",
                    confidence=conf,
                )
            )

        return detections

    def _predict_onnx(self, frame_img: np.ndarray) -> List[Detection]:
        """Run inference on ONNX export of YOLO11."""
        h0, w0 = frame_img.shape[:2]
        scale = min(self.imgsz / w0, self.imgsz / h0)
        nw, nh = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(frame_img, (nw, nh))

        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        padded[:nh, :nw] = resized
        blob = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        input_name = self._onnx_session.get_inputs()[0].name
        preds = self._onnx_session.run(None, {input_name: blob})[0]
        # Shape: (1, 4 + num_classes, num_boxes) or (1, num_boxes, 4 + num_classes)
        if preds.shape[1] < preds.shape[2]:
            preds = preds.transpose(0, 2, 1)

        predictions = preds[0]
        boxes_list = []
        scores_list = []
        class_ids = []

        for pred in predictions:
            cx, cy, bw, bh = pred[:4]
            scores = pred[4:]
            cls_id = np.argmax(scores) if len(scores) > 0 else 0
            conf = float(scores[cls_id]) if len(scores) > 0 else float(pred[4])

            if conf < self.conf:
                continue

            x1 = (cx - bw / 2) / scale
            y1 = (cy - bh / 2) / scale
            x2 = (cx + bw / 2) / scale
            y2 = (cy + bh / 2) / scale

            boxes_list.append([max(0, x1), max(0, y1), min(w0, x2 - x1), min(h0, y2 - y1)])
            scores_list.append(conf)
            class_ids.append(int(cls_id))

        if not boxes_list:
            return []

        indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, self.conf, self.iou)
        detections = []
        now = datetime.now(timezone.utc)
        if len(indices) > 0:
            for idx in indices.flatten()[: self.max_det]:
                b = boxes_list[idx]
                detections.append(
                    Detection(
                        detection_id=str(uuid.uuid4()),
                        shelf_id="unassigned",
                        frame_timestamp=now,
                        bbox=BoundingBox(x1=b[0], y1=b[1], x2=b[0] + b[2], y2=b[1] + b[3]),
                        class_id=class_ids[idx],
                        sku_id=f"sku_{class_ids[idx]}",
                        confidence=scores_list[idx],
                    )
                )

        return detections
