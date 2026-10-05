"""Headless CLI commands for shelf_monitor."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np

from .entities.common import CellState
from .frameworks.exceptions import ModelCorruptError
from .frameworks.logging_config import get_logger, setup_logging
from .frameworks.model_registry import ModelMode, ModelRegistry
from .usecases.metrics import ShelfMetrics, compute_metrics_from_cell_states

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shelf-monitor",
        description="Retail Shelf Monitoring System CLI",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--models-dir", default=None, help="Override models directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. check-models
    p_check = subparsers.add_parser("check-models", help="Inspect and validate model artifacts in models/")
    p_check.add_argument("--dir", default=None, help="Directory to check (default: models/)")

    # 2. make-planogram
    p_plano = subparsers.add_parser("make-planogram", help="Generate planogram from reference image")
    p_plano.add_argument("--image", required=True, help="Path to reference shelf image")
    p_plano.add_argument("--shelf-id", required=True, help="Shelf identifier (e.g. S1)")

    # 3. analyze-image
    p_img = subparsers.add_parser("analyze-image", help="Analyze single image for compliance")
    p_img.add_argument("--image", required=True, help="Path to input image")
    p_img.add_argument("--shelf-id", required=True, help="Shelf identifier")
    p_img.add_argument("--out", default="outputs/runs", help="Output directory for annotated image & report")

    # 4. analyze-video
    p_vid = subparsers.add_parser("analyze-video", help="Analyze video stream for compliance")
    p_vid.add_argument("--video", required=True, help="Path to input video file or webcam index")
    p_vid.add_argument("--shelf-id", required=True, help="Shelf identifier")
    p_vid.add_argument("--out", default="outputs/runs", help="Output directory")

    # 5. enroll-skus
    p_enroll = subparsers.add_parser("enroll-skus", help="Enroll SKU reference crops into FAISS index")
    p_enroll.add_argument("--crops", required=True, help="Path to folder containing class-per-folder crops")
    p_enroll.add_argument("--out-dir", default=None, help="Target dir for index and labels (default: models/)")

    # 6. demo
    p_demo = subparsers.add_parser("demo", help="Run full offline demo with synthetic data")
    p_demo.add_argument("--out", default="outputs/runs", help="Output directory for demo run")

    # 7. ui
    subparsers.add_parser("ui", help="Launch desktop PySide6 UI")

    return parser


def cmd_check_models(container: Any, models_dir_override: Optional[str] = None) -> int:
    """Validate contract files and run self-test. Non-zero exit on corrupt file."""
    config = container.config()
    m_dir = models_dir_override or config.app.models_dir
    reg = ModelRegistry(models_dir=m_dir)
    reg.print_status()

    # Print table
    table = reg.status_table()
    print("\nFiles in contract:")
    print(f"{'Filename':<26} | {'Status':<10} | {'Size':<10}")
    print("-" * 52)
    for r in table:
        st = "FOUND" if r["found"] else "MISSING"
        print(f"{r['file']:<26} | {st:<10} | {r['size']:<10}")

    print(f"\nOperational Mode: {reg.mode.value}")

    # Self-test: 1 image inference + 1 embedding + 1 FAISS query
    print("\nRunning self-test...")
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

    try:
        det = container.yolo_detector()
        detections = det.predict(dummy_img)
        print(f"  [✓] Detector self-test passed (found {len(detections)} boxes on blank)")
    except Exception as e:
        print(f"  [✗] Detector self-test failed: {e}")
        return 1

    if reg.mode == ModelMode.FULL:
        try:
            rec = container.sku_recognizer()
            dummy_crop = np.zeros((224, 224, 3), dtype=np.uint8)
            embs = rec.extract_embeddings([dummy_crop])
            print(f"  [✓] Embedding self-test passed (shape: {embs.shape})")
            skus = rec.batch_identify_skus([dummy_crop])
            print(f"  [✓] FAISS query self-test passed (identified: {skus[0]})")
        except Exception as e:
            print(f"  [✗] SKU recognition self-test failed: {e}")
            return 1

    print("\nSelf-test complete: OK")
    return 0


def cmd_make_planogram(container: Any, image_path: str, shelf_id: str) -> int:
    usecase = container.planogram_generation_usecase()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        planogram = loop.run_until_complete(
            usecase.generate_planogram_from_reference(shelf_id, image_path)
        )
        print(f"✓ Planogram generated for shelf '{shelf_id}'")
        print(f"  Rows: {len(planogram.grid.rows)}, Total items: {planogram.grid.total_items}")
        return 0
    except Exception as e:
        print(f"✗ Failed to generate planogram: {e}")
        return 1
    finally:
        loop.close()


def cmd_analyze_image(container: Any, image_path: str, shelf_id: str, out_dir: str) -> int:
    p_img = Path(image_path)
    if not p_img.exists():
        print(f"Error: image not found: {image_path}")
        return 1

    img = cv2.imread(str(p_img))
    if img is None:
        print(f"Error: failed to load image: {image_path}")
        return 1

    stream_proc = container.stream_processing_usecase()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    try:
        now = datetime.now(timezone.utc)
        res = loop.run_until_complete(
            stream_proc.process_frame(img, frame_id=p_img.stem, timestamp=now)
        )

        # Annotate image
        annotated = img.copy()
        for det in res.detections:
            x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
            x2, y2 = int(det.bbox.x2), int(det.bbox.y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 120, 0), 2)
            label = f"{det.sku_id or 'product'} {det.confidence:.2f}"
            cv2.putText(annotated, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out_img_path = out_p / f"{p_img.stem}_annotated.jpg"
        cv2.imwrite(str(out_img_path), annotated)

        # Write JSON report
        report = {
            "image": str(p_img),
            "shelf_id": shelf_id,
            "timestamp": now.isoformat(),
            "detections_count": len(res.detections),
            "summary": res.summary or {},
            "alerts": [a.model_dump(mode="json") for a in res.alerts],
            "annotated_image": str(out_img_path),
        }
        report_path = out_p / f"{p_img.stem}_report.json"
        with report_path.open("w") as f:
            json.dump(report, f, indent=2)

        print(f"✓ Analysis complete for '{p_img.name}'")
        print(f"  Annotated image: {out_img_path}")
        print(f"  JSON report:     {report_path}")
        if res.summary:
            print(f"  Fill %: {res.summary.get('fill_pct')}% | Compliance %: {res.summary.get('compliance_pct')}%")
        return 0
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        return 1
    finally:
        loop.close()


def cmd_analyze_video(container: Any, video_path: str, shelf_id: str, out_dir: str) -> int:
    cap = cv2.VideoCapture(int(video_path) if video_path.isdigit() else video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return 1

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    stem = Path(video_path).stem if not video_path.isdigit() else f"cam_{video_path}"
    out_vid_path = out_p / f"{stem}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (w, h))

    stream_proc = container.stream_processing_usecase()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    all_alerts = []
    last_summary = {}
    frame_idx = 0

    print(f"Processing video {video_path}...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            now = datetime.now(timezone.utc)
            res = loop.run_until_complete(
                stream_proc.process_frame(frame, frame_id=f"f_{frame_idx}", timestamp=now)
            )

            annotated = frame.copy()
            for det in res.detections:
                x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
                x2, y2 = int(det.bbox.x2), int(det.bbox.y2)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det.sku_id or 'prod'}"
                cv2.putText(annotated, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if res.summary:
                last_summary = res.summary
                txt = f"Fill: {res.summary.get('fill_pct', 0)}%  Comp: {res.summary.get('compliance_pct', 0)}%"
                cv2.putText(annotated, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            writer.write(annotated)
            for a in res.alerts:
                all_alerts.append(a.model_dump(mode="json"))

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  Processed {frame_idx} frames...")

    finally:
        cap.release()
        writer.release()
        loop.close()

    report_path = out_p / f"{stem}_report.json"
    report = {
        "video": video_path,
        "shelf_id": shelf_id,
        "frames_processed": frame_idx,
        "summary": last_summary,
        "alerts_count": len(all_alerts),
        "alerts": all_alerts,
        "annotated_video": str(out_vid_path),
    }
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"✓ Video processing complete: {frame_idx} frames")
    print(f"  Annotated video: {out_vid_path}")
    print(f"  JSON report:     {report_path}")
    return 0


def cmd_enroll_skus(container: Any, crops_dir: str, out_dir: Optional[str] = None) -> int:
    """Enrolls class-per-folder SKU reference crops into sku_index.faiss and sku_labels.json."""
    from training.build_sku_index import build_index_from_crops
    config = container.config()
    target_dir = out_dir or config.app.models_dir
    emb_model_path = container.model_registry().embedding_path()
    dim = container.model_registry().get_embedding_dim()

    print(f"Enrolling SKUs from '{crops_dir}' into '{target_dir}'...")
    try:
        num_vecs, num_classes = build_index_from_crops(
            crops_dir=crops_dir,
            output_dir=target_dir,
            embedding_model_path=str(emb_model_path) if emb_model_path else None,
            embedding_dim=dim,
        )
        print(f"✓ Successfully indexed {num_vecs} reference images across {num_classes} SKUs")
        return 0
    except Exception as e:
        print(f"✗ SKU enrollment failed: {e}")
        return 1


def cmd_demo(container: Any, out_dir: str) -> int:
    """Run end-to-end zero-download offline demo."""
    from scripts.run_demo import run_full_demo
    return run_full_demo(out_dir=out_dir)


def cmd_ui(container: Any) -> int:
    """Launch desktop PySide6 UI."""
    from PySide6.QtWidgets import QApplication
    from .frameworks.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(container)
    window.show()
    return app.exec()


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging(level="INFO")
    parser = build_parser()
    args = parser.parse_args(argv)

    from .container import ApplicationContainer
    container = ApplicationContainer()

    # Environment or CLI override
    if args.models_dir:
        container.config.app.models_dir.override(args.models_dir)

    cmd = args.command
    if cmd == "check-models":
        return cmd_check_models(container, args.dir)
    elif cmd == "make-planogram":
        return cmd_make_planogram(container, args.image, args.shelf_id)
    elif cmd == "analyze-image":
        return cmd_analyze_image(container, args.image, args.shelf_id, args.out)
    elif cmd == "analyze-video":
        return cmd_analyze_video(container, args.video, args.shelf_id, args.out)
    elif cmd == "enroll-skus":
        return cmd_enroll_skus(container, args.crops, args.out_dir)
    elif cmd == "demo":
        return cmd_demo(container, args.out)
    elif cmd == "ui":
        return cmd_ui(container)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
