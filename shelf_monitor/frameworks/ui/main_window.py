"""Main application window for Retail Shelf Monitoring."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ...entities.planogram import Planogram
from ...frameworks.logging_config import get_logger
from .threads.alert_analysis_thread import AlertAnalysisThread
from .threads.alert_thread import AlertThread
from .threads.capture_thread import CaptureThread
from .threads.inference_thread import InferenceThread
from .widgets.alert_panel import AlertPanel
from .widgets.planogram_view import PlanogramView
from .widgets.video_widget import VideoWidget

logger = get_logger(__name__)


class MainWindow(QMainWindow):
    """Main desktop application window."""

    def __init__(self, container: Any) -> None:
        super().__init__()
        self.container = container
        self.setWindowTitle("Retail Shelf Monitoring System")
        self.resize(1360, 840)

        # Worker threads
        self.capture_thread: Optional[CaptureThread] = None
        self.inference_thread: Optional[InferenceThread] = None
        self.alert_analysis_thread: Optional[AlertAnalysisThread] = None
        self.alert_thread: Optional[AlertThread] = None

        self._setup_ui()
        self._load_styles()
        self._setup_threads()
        self._auto_load_active_planogram()
        self._show_model_status_banner()

    def _setup_ui(self) -> None:
        # Central widget
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        # 1. Mode & Status Banner
        self.banner = QLabel()
        self.banner.setFixedHeight(30)
        self.banner.setStyleSheet(
            "padding: 4px 12px; font-weight: bold; border-radius: 4px; font-size: 12px;"
        )
        root_layout.addWidget(self.banner)

        # 2. Metrics Strip
        metrics_box = QGroupBox("Shelf Metrics")
        metrics_box.setFixedHeight(80)
        m_layout = QHBoxLayout(metrics_box)
        m_layout.setContentsMargins(15, 5, 15, 5)

        self.lbl_fps = self._create_metric("FPS", "0.0")
        self.lbl_products = self._create_metric("Products", "0")
        self.lbl_oos = self._create_metric("OOS (Empty)", "0")
        self.lbl_misplaced = self._create_metric("Misplaced", "0")
        self.lbl_fill = self._create_metric("Fill %", "0.0%")
        self.lbl_compliance = self._create_metric("Compliance %", "0.0%")

        for lbl_w in [
            self.lbl_fps, self.lbl_products, self.lbl_oos,
            self.lbl_misplaced, self.lbl_fill, self.lbl_compliance
        ]:
            m_layout.addWidget(lbl_w)

        root_layout.addWidget(metrics_box)

        # 3. Main Splitter (Video on left, Planogram + Alerts on right)
        splitter = QSplitter(Qt.Horizontal)

        # Left: Video Player
        video_box = QGroupBox("Live Monitoring Feed")
        v_layout = QVBoxLayout(video_box)
        self.video_widget = VideoWidget()
        v_layout.addWidget(self.video_widget)
        splitter.addWidget(video_box)

        # Right: Tabs/Panels
        right_panel = QWidget()
        r_layout = QVBoxLayout(right_panel)
        r_layout.setContentsMargins(0, 0, 0, 0)
        r_layout.setSpacing(8)

        # Planogram view
        plano_box = QGroupBox("Active Planogram")
        plano_box.setMaximumHeight(220)
        p_layout = QVBoxLayout(plano_box)
        self.planogram_view = PlanogramView()
        p_layout.addWidget(self.planogram_view)
        r_layout.addWidget(plano_box)

        # Alert panel
        alert_box = QGroupBox("Compliance Alerts")
        a_layout = QVBoxLayout(alert_box)
        self.alert_panel = AlertPanel()
        self.alert_panel.alert_confirmed.connect(self._on_confirm_alert)
        self.alert_panel.alert_dismissed.connect(self._on_dismiss_alert)
        a_layout.addWidget(self.alert_panel)
        r_layout.addWidget(alert_box)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter)

        # Menu Bar
        self._setup_menus()

        # Status Bar
        self.statusBar().showMessage("Ready")

    def _create_metric(self, title: str, default_val: str) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        t_lbl = QLabel(title)
        t_lbl.setObjectName("metrics_title")
        t_lbl.setStyleSheet("color: #61afef; font-size: 11px;")

        v_lbl = QLabel(default_val)
        v_lbl.setObjectName("metric_val")
        v_lbl.setStyleSheet("color: #98c379; font-size: 18px; font-weight: bold;")

        layout.addWidget(t_lbl)
        layout.addWidget(v_lbl)
        w.val_label = v_lbl
        return w

    def _setup_menus(self) -> None:
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("&File")

        open_video_act = QAction("Open Video File...", self)
        open_video_act.triggered.connect(self._open_video_dialog)
        file_menu.addAction(open_video_act)

        open_cam_act = QAction("Open Webcam / Camera...", self)
        open_cam_act.triggered.connect(self._open_webcam)
        file_menu.addAction(open_cam_act)

        open_rtsp_act = QAction("Open RTSP Stream...", self)
        open_rtsp_act.triggered.connect(self._open_rtsp)
        file_menu.addAction(open_rtsp_act)

        file_menu.addSeparator()

        exit_act = QAction("E&xit", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # Planogram Menu
        plano_menu = menubar.addMenu("&Planogram")

        load_plano_act = QAction("Load Planogram from DB...", self)
        load_plano_act.triggered.connect(self._load_planogram_dialog)
        plano_menu.addAction(load_plano_act)

        create_plano_act = QAction("Create Planogram from Reference Image...", self)
        create_plano_act.triggered.connect(self._create_planogram_dialog)
        plano_menu.addAction(create_plano_act)

    def _load_styles(self) -> None:
        qss_path = Path(__file__).parent / "resources" / "styles.qss"
        if qss_path.exists():
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    def _show_model_status_banner(self) -> None:
        registry = getattr(self.container, "model_registry", None)
        if registry:
            mode = registry().mode.value
            if mode == "FULL":
                self.banner.setText(f"System Mode: FULL (Custom YOLO11 + MobileNetV3 + FAISS loaded from models/)")
                self.banner.setStyleSheet("background-color: #1e3a29; color: #52c41a; font-weight: bold; padding: 6px;")
            elif mode == "POSITION-ONLY":
                self.banner.setText("System Mode: POSITION-ONLY (Custom Detector loaded; SKU recognition disabled)")
                self.banner.setStyleSheet("background-color: #382d18; color: #faad14; font-weight: bold; padding: 6px;")
            else:
                self.banner.setText("System Mode: BASELINE (Stock COCO YOLO11; models/ empty — accuracy on dense shelves will be limited)")
                self.banner.setStyleSheet("background-color: #3a1e1e; color: #f5222d; font-weight: bold; padding: 6px;")
        else:
            self.banner.setText("System Mode: ACTIVE")

    def _setup_threads(self) -> None:
        stream_proc = self.container.stream_processing_usecase()
        alert_gen = self.container.alert_generation_usecase()
        alert_mgmt = self.container.alert_management_usecase()

        self.inference_thread = InferenceThread(stream_proc)
        self.inference_thread.detections_ready.connect(self._on_detections_ready)
        self.inference_thread.planogram_ready.connect(self.planogram_view.set_planogram)
        self.inference_thread.start()

        self.alert_analysis_thread = AlertAnalysisThread(alert_gen)
        self.alert_analysis_thread.start()

        self.alert_thread = AlertThread(alert_mgmt)
        self.alert_thread.active_alerts_fetched.connect(self.alert_panel.set_alerts)
        self.alert_thread.start()

    def _auto_load_active_planogram(self) -> None:
        try:
            repo = self.container.planogram_repository()
            cfg = self.container.config()
            shelf_id = getattr(cfg.aligner, "fixed_shelf_id", "S1") or "S1"
            loop = asyncio.new_event_loop()
            try:
                planograms = loop.run_until_complete(repo.list_all())
                if not planograms:
                    ref_p = Path("data/demo/shelf_reference.jpg")
                    if ref_p.exists():
                        gen_uc = self.container.planogram_generation_usecase()
                        plano = loop.run_until_complete(
                            gen_uc.generate_planogram_from_reference(shelf_id, str(ref_p))
                        )
                        planograms = [plano]
                if planograms:
                    plano = next((p for p in planograms if p.shelf_id == shelf_id), planograms[0])
                    self.planogram_view.set_planogram(plano)
                    stream_proc = self.container.stream_processing_usecase()
                    stream_proc.fixed_shelf_id = plano.shelf_id
                    stream_proc._planograms[plano.shelf_id] = plano
            finally:
                loop.close()
        except Exception as exc:
            logger.debug(f"Auto planogram load: {exc}")

    def start_feed(self, source: Any) -> None:
        if self.capture_thread and self.capture_thread.isRunning():
            self.capture_thread.stop()

        self._auto_load_active_planogram()

        self.capture_thread = CaptureThread(source=source)
        self.capture_thread.frame_captured.connect(self._on_frame_captured)
        self.capture_thread.error_occurred.connect(lambda err: QMessageBox.warning(self, "Capture Error", err))
        self.capture_thread.start()
        self.statusBar().showMessage(f"Streaming from: {source}")

    def _on_frame_captured(self, frame_img: np.ndarray, frame_idx: int) -> None:
        if self.inference_thread:
            self.inference_thread.enqueue_frame(frame_img, frame_idx)

    def _on_detections_ready(
        self,
        frame_img: np.ndarray,
        detections: list,
        cell_states: list,
        summary: dict,
        fps: float,
    ) -> None:
        self.video_widget.update_frame(frame_img, detections, cell_states)

        # Update metrics
        self.lbl_fps.val_label.setText(f"{fps:.1f}")
        total = summary.get("total_cells", 0)
        empty = summary.get("empty_count", 0)
        misplaced = summary.get("misplaced_count", 0)
        fill = summary.get("fill_pct", 0.0)
        comp = summary.get("compliance_pct", 100.0)

        # Products = number of products currently detected on shelf
        n_products = len(detections) if len(detections) > 0 else max(0, total - empty)
        self.lbl_products.val_label.setText(str(n_products))
        self.lbl_oos.val_label.setText(str(empty))
        self.lbl_misplaced.val_label.setText(str(misplaced))
        self.lbl_fill.val_label.setText(f"{fill:.1f}%")
        self.lbl_compliance.val_label.setText(f"{comp:.1f}%")

    def _open_video_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        if path:
            self.start_feed(path)

    def _open_webcam(self) -> None:
        idx, ok = QInputDialog.getInt(self, "Open Webcam", "Enter camera index (0, 1, ...):", 0, 0, 10)
        if ok:
            self.start_feed(idx)

    def _open_rtsp(self) -> None:
        url, ok = QInputDialog.getText(self, "Open RTSP Stream", "Enter RTSP URL:")
        if ok and url:
            self.start_feed(url)

    def _load_planogram_dialog(self) -> None:
        shelf_id, ok = QInputDialog.getText(self, "Load Planogram", "Enter Shelf ID:")
        if ok and shelf_id:
            repo = self.container.planogram_repository()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                planogram = loop.run_until_complete(repo.get_by_shelf_id(shelf_id))
                if planogram:
                    self.planogram_view.set_planogram(planogram)
                    self.statusBar().showMessage(f"Loaded planogram for shelf '{shelf_id}'")
                else:
                    QMessageBox.information(self, "Not Found", f"No planogram found for shelf '{shelf_id}'")
            finally:
                loop.close()

    def _create_planogram_dialog(self) -> None:
        shelf_id, ok = QInputDialog.getText(self, "Create Planogram", "Enter Shelf ID (e.g. S1):")
        if not ok or not shelf_id:
            return
        img_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image", "", "Images (*.jpg *.png *.jpeg *.bmp)"
        )
        if not img_path:
            return

        gen_uc = self.container.planogram_generation_usecase()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            plano = loop.run_until_complete(gen_uc.generate_planogram_from_reference(shelf_id, img_path))
            self.planogram_view.set_planogram(plano)
            QMessageBox.information(
                self, "Success", f"Planogram created: {len(plano.grid.rows)} rows, {plano.grid.total_items} items"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create planogram: {e}")
        finally:
            loop.close()

    def _on_confirm_alert(self, alert_id: str, staff_id: str) -> None:
        alert_mgmt = self.container.alert_management_usecase()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(alert_mgmt.confirm_alert(alert_id, staff_id))
            self.statusBar().showMessage(f"Alert {alert_id[:8]} confirmed by {staff_id}")
            alerts = loop.run_until_complete(alert_mgmt.get_active_alerts())
            self.alert_panel.set_alerts(alerts)
        finally:
            loop.close()

    def _on_dismiss_alert(self, alert_id: str) -> None:
        alert_mgmt = self.container.alert_management_usecase()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(alert_mgmt.dismiss_alert(alert_id))
            self.statusBar().showMessage(f"Alert {alert_id[:8]} dismissed")
            alerts = loop.run_until_complete(alert_mgmt.get_active_alerts())
            self.alert_panel.set_alerts(alerts)
        finally:
            loop.close()

    def closeEvent(self, event) -> None:
        if self.capture_thread:
            self.capture_thread.stop()
        if self.inference_thread:
            self.inference_thread.stop()
        if self.alert_analysis_thread:
            self.alert_analysis_thread.stop()
        if self.alert_thread:
            self.alert_thread.stop()
        event.accept()
