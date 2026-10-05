"""Dependency injection container wiring all components together."""
from __future__ import annotations

from dependency_injector import containers, providers

from .adaptors.keyframe_selector import KeyframeSelector
from .adaptors.ml.sku_recognizer import SkuRecognizer
from .adaptors.ml.yolo_detector import YoloDetector
from .adaptors.repositories.memory_alert_store import MemoryAlertStore
from .adaptors.repositories.sqlite_planogram_repo import SqlitePlanogramRepository
from .adaptors.tracking.sort import SortTracker
from .frameworks.config import AppConfig
from .frameworks.database import DatabaseManager
from .frameworks.logging_config import get_logger
from .frameworks.model_registry import ModelRegistry
from .usecases.alert_generation import AlertGenerationUseCase, AlertManagementUseCase
from .usecases.cell_state_computation import CellStateComputation
from .usecases.detection_processing import DetectionProcessingUseCase
from .usecases.grid.grid_detector import GridDetector
from .usecases.planogram_generation import PlanogramGenerationUseCase
from .usecases.shelf_aligner.feature_matcher import FeatureMatcher
from .usecases.shelf_aligner.homography import HomographyEstimator
from .usecases.shelf_aligner.shelf_aligner import ShelfAligner
from .usecases.stream_processing import StreamProcessingUseCase
from .usecases.temporal_consensus import TemporalConsensusManager


def _create_yolo_detector(registry: ModelRegistry, config: AppConfig) -> YoloDetector:
    det_path = registry.detector_path()
    return YoloDetector(
        model_path=str(det_path) if det_path else None,
        confidence_threshold=config.ml.confidence,
        nms_threshold=config.ml.iou,
        imgsz=registry.get_imgsz(),
        max_det=registry.get_max_det(),
        device=config.ml.device,
        engine_type=config.ml.inference_engine,
    )


def _create_sku_recognizer(registry: ModelRegistry, config: AppConfig) -> SkuRecognizer:
    mean, std = registry.get_mean_std()
    return SkuRecognizer(
        embedding_model_path=registry.embedding_path(),
        index_path=registry.index_path(),
        labels_path=registry.labels_path(),
        input_size=config.sku.input_size,
        embedding_dim=registry.get_embedding_dim(),
        similarity_threshold=registry.get_similarity_threshold(),
        top_k=config.sku.top_k,
        mean=mean,
        std=std,
        device=config.ml.device,
        re_id_every_k_frames=config.sku.re_id_every_k_frames,
    )


def _create_alert_store(config: AppConfig):
    if config.alerts.backend == "redis":
        try:
            import redis
            client = redis.Redis(
                host=config.alerts.redis_host,
                port=config.alerts.redis_port,
                db=config.alerts.redis_db,
                password=config.alerts.redis_password,
                decode_responses=True,
            )
            from .adaptors.repositories.redis_alert_store import RedisAlertStore
            return RedisAlertStore(client)
        except Exception:
            return MemoryAlertStore()
    return MemoryAlertStore()


class ApplicationContainer(containers.DeclarativeContainer):
    """Declarative dependency injection container."""

    # 1. Config & Logging
    config = providers.Singleton(AppConfig.load)
    logger = providers.Singleton(get_logger, name="shelf_monitor")

    # 2. Model Registry
    model_registry = providers.Singleton(
        ModelRegistry,
        models_dir=config.provided.app.models_dir,
    )

    # 3. Database & Repositories
    database_manager = providers.Singleton(
        DatabaseManager,
        database_url=config.provided.database.url,
    )

    planogram_repository = providers.Factory(
        SqlitePlanogramRepository,
        session_factory=database_manager.provided.get_session.call(),
    )

    alert_repository = providers.Singleton(
        _create_alert_store,
        config=config,
    )

    # 4. ML Models
    yolo_detector = providers.Singleton(
        _create_yolo_detector,
        registry=model_registry,
        config=config,
    )

    sku_recognizer = providers.Singleton(
        _create_sku_recognizer,
        registry=model_registry,
        config=config,
    )

    # 5. Vision / Tracking Adaptors
    tracker = providers.Singleton(
        SortTracker,
        max_age=config.provided.tracking.max_age,
        min_hits=config.provided.tracking.min_hits,
        iou_threshold=config.provided.tracking.iou_threshold,
    )

    keyframe_selector = providers.Factory(
        KeyframeSelector,
        diff_threshold=config.provided.stream.diff_threshold,
    )

    feature_matcher = providers.Singleton(
        FeatureMatcher,
        feature_type=config.provided.aligner.feature_type,
        max_features=config.provided.aligner.max_features,
        lowe_ratio=config.provided.aligner.lowe_ratio,
    )

    homography_estimator = providers.Singleton(
        HomographyEstimator,
        ransac_reproj_threshold=config.provided.aligner.ransac_reproj,
        min_alignment_confidence=config.provided.aligner.min_alignment_confidence,
    )

    shelf_aligner = providers.Singleton(
        ShelfAligner,
        reference_dir=config.provided.aligner.reference_dir,
        feature_matcher=feature_matcher,
        homography_estimator=homography_estimator,
        min_alignment_confidence=config.provided.aligner.min_alignment_confidence,
        single_shelf_mode=config.provided.stream.single_shelf_mode,
        fixed_shelf_id=config.provided.aligner.fixed_shelf_id,
    )

    grid_detector = providers.Factory(
        GridDetector,
        clustering_method=config.provided.grid.clustering_method,
        eps=config.provided.grid.eps,
        min_samples=config.provided.grid.min_samples,
    )

    # 6. Core Use Cases
    detection_processing_usecase = providers.Factory(
        DetectionProcessingUseCase,
        detector=yolo_detector,
        sku_recognizer=sku_recognizer,
    )

    cell_state_computation = providers.Factory(
        CellStateComputation,
        grid_detector=grid_detector,
        position_tolerance=config.provided.grid.position_tolerance,
        confidence_threshold=config.provided.ml.confidence,
    )

    temporal_consensus_manager = providers.Singleton(
        TemporalConsensusManager,
        n_confirm=config.provided.consensus.min_consecutive_frames,
        n_clear=config.provided.consensus.clear_after_frames,
    )

    alert_generation_usecase = providers.Factory(
        AlertGenerationUseCase,
        alert_repository=alert_repository,
        planogram_repository=planogram_repository,
    )

    alert_management_usecase = providers.Factory(
        AlertManagementUseCase,
        alert_repository=alert_repository,
    )

    planogram_generation_usecase = providers.Factory(
        PlanogramGenerationUseCase,
        planogram_repository=planogram_repository,
        detector=yolo_detector,
        sku_recognizer=sku_recognizer,
        grid_detector=grid_detector,
    )

    stream_processing_usecase = providers.Factory(
        StreamProcessingUseCase,
        shelf_aligner=shelf_aligner,
        detection_processing=detection_processing_usecase,
        planogram_repository=planogram_repository,
        tracker=tracker,
        keyframe_selector=keyframe_selector,
        cell_state_computation=cell_state_computation,
        temporal_consensus=temporal_consensus_manager,
        alert_generation=alert_generation_usecase,
        keyframe_interval=config.provided.stream.keyframe_interval,
        single_shelf_mode=config.provided.stream.single_shelf_mode,
        fixed_shelf_id=config.provided.aligner.fixed_shelf_id,
    )
