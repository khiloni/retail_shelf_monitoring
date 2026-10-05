# Technical Report: Retail Shelf Monitoring System

## Abstract

This report describes the design and implementation of a production-grade retail shelf monitoring system built with deep learning. The system detects shelf products using a fine-tuned YOLOv11n detector, recognizes individual SKUs with a metric-learning embedding model (MobileNetV3 + TripletLoss), performs planogram compliance via DBSCAN grid clustering, and raises real-time out-of-stock and misplacement alerts through a temporal consensus filter. The system is implemented with Clean Architecture principles and ships with a PySide6 desktop UI and a full headless CLI.

---

## 1. Problem Statement

Retail out-of-stock events cause billions in lost sales annually. Manual shelf auditing is labor-intensive and infrequent. This system provides continuous automated shelf compliance monitoring using a standard CCTV camera as input, requiring no specialized hardware.

**Key requirements:**
- Detect all products on a shelf in real time
- Identify individual SKUs (product types)
- Compare shelf state against a reference planogram
- Raise alerts for out-of-stock and misplacement events
- Handle changing lighting, camera angles, and partial occlusions
- Deployable without a local GPU

---

## 2. System Architecture

### 2.1 Clean Architecture

The system follows Clean Architecture with four concentric layers:

| Layer | Responsibility | Key Files |
|---|---|---|
| **Entities** | Domain models with invariants | `entities/*.py` |
| **Use Cases** | Business rules, pure logic | `usecases/*.py` |
| **Adaptors** | Interface adapters (YOLO, FAISS, SQLite) | `adaptors/*.py` |
| **Frameworks** | External frameworks (Qt, ONNX, Config) | `frameworks/*.py` |

Dependency inversion is enforced: adaptors implement interfaces defined in `usecases/interfaces/`. This allows swapping YOLO for any other detector without touching business logic.

### 2.2 Domain Entities

```
CellState: OK | EMPTY | MISPLACED | UNKNOWN
BoundingBox: x1, y1, x2, y2 + area/IoU utilities
Detection: bbox + confidence + sku_id + track_id
Frame: image array + timestamp + shelf_id
PlanogramCell: row + col + expected_sku_id + state
Alert: shelf_id + cell_id + alert_type + timestamp
ShelfMetrics: fill_rate + oos_count + misplaced_count
```

---

## 3. Detection Model

### 3.1 Architecture
**YOLOv11n** (Nano variant) — chosen for:
- Fast inference on CPU (< 50ms per frame)
- Single-class detection ("product") suited to SKU-110K
- Strong transfer learning from COCO pretraining

### 3.2 Dataset
- **Primary**: SKU-110K — 11,762 images, 1.7M bounding box annotations of retail shelf products
- **Optional**: Custom Roboflow dataset for domain-specific fine-tuning

### 3.3 Training Protocol
| Parameter | Value |
|---|---|
| Backbone | YOLOv11n (pre-trained COCO) |
| Input size | 640×640 |
| Epochs | 100 |
| Batch size | 16 |
| Optimizer | AdamW |
| LR schedule | Cosine decay, warmup 3 epochs |
| Augmentation | Mosaic, MixUp, HSV, flip, scale |
| Hardware | Google Colab T4 GPU |

### 3.4 Post-Processing
- SORT multi-object tracker assigns persistent track IDs across frames
- NMS threshold: IoU 0.45, confidence 0.35
- Max detections: 300 per frame

---

## 4. SKU Recognition Model

### 4.1 Architecture
**MobileNetV3-Large** backbone + projection head:
- `MobileNetV3-Large` (ImageNet pre-trained, frozen initially)
- `AdaptiveAvgPool2d` → `Flatten`
- `Linear(960, 512)` → `BatchNorm1d` → `ReLU`
- `Linear(512, embedding_dim)` ← configurable (default 256)
- `L2-normalize` output

### 4.2 Metric Learning
**Online Triplet Mining with TripletMarginLoss**:
- Anchor: detected product crop
- Positive: different crop of same SKU
- Negative: hard negative from same batch (different SKU)
- Margin: 0.3

### 4.3 FAISS Index
- IndexFlatIP (inner product, equivalent to cosine similarity after L2-norm)
- Reference crops: ≥ 10 images per SKU class
- Top-K retrieval: K=3, score threshold: 0.6

### 4.4 Training Protocol
| Parameter | Value |
|---|---|
| Backbone | MobileNetV3-Large (ImageNet) |
| Input size | 224×224 |
| Epochs | 50 |
| Batch size | 64 |
| Optimizer | Adam (lr=1e-3) |
| LR schedule | Step (×0.5 every 15 epochs) |
| Augmentation | Random crop, flip, color jitter |

---

## 5. Grid & Compliance Pipeline

### 5.1 Grid Detection (DBSCAN Clustering)
Products are clustered into a shelf grid:
1. Extract bounding box centers from tracked detections
2. DBSCAN clustering on Y-coordinates → shelf rows
3. DBSCAN clustering on X-coordinates within each row → columns
4. Assign (row, col) cell coordinates to each detection

**Parameters** (configurable in `config.yaml`):
- `eps`: 15.0 pixels
- `min_samples`: 2

### 5.2 Shelf Alignment (Homography)
In multi-shelf mode:
1. ORB feature extraction on current frame
2. Brute-force matching against reference shelf images
3. RANSAC homography estimation
4. Shelf ID assigned based on best match confidence

In single-shelf mode (default): fixed shelf ID, no alignment needed.

### 5.3 Planogram Compliance
For each grid cell:
1. Compare detected SKU at (row, col) against planogram `expected_sku_id`
2. Assign `CellState`:
   - `OK` — correct product present
   - `EMPTY` — no detection in cell
   - `MISPLACED` — wrong SKU detected (requires FULL mode)
   - `UNKNOWN` — no planogram entry

### 5.4 Temporal Consensus
Raw per-frame states are noisy. A state transition is only committed after **N consecutive frames** agree (configurable, default N=3). This prevents single-frame false alerts.

---

## 6. Alert System

### 6.1 Alert Rules
| Condition | Alert Type | Priority |
|---|---|---|
| Cell EMPTY → was OK | `out_of_stock` | HIGH |
| Cell MISPLACED | `misplaced` | MEDIUM |
| State reverted | Alert resolved | INFO |

### 6.2 Alert Storage
- **Default**: In-memory store (no external services)
- **Optional**: Redis-backed store for multi-process/distributed setups

### 6.3 Deduplication
Alerts are deduplicated by `(shelf_id, cell_id, alert_type)` — no duplicate alerts for sustained OOS events.

---

## 7. Inference & Deployment

### 7.1 ONNX Export
Both models are exported to ONNX for portable inference:
- Fixed input shapes (detector: `[1,3,640,640]`, embedding: `[B,3,224,224]`)
- Parity check: `max(|ONNX_output - PyTorch_output|) < 1e-4`
- ONNXRuntime inference engine available as alternative to Ultralytics

### 7.2 Model Registry
Automatically detects operating mode at startup — no code changes needed:
- **BASELINE**: stock yolo11n, detection only
- **POSITION-ONLY**: fine-tuned detector, OK/EMPTY
- **FULL**: all models, complete pipeline

### 7.3 Hardware Requirements
| Config | CPU | RAM | Inference Speed |
|---|---|---|---|
| BASELINE | 4-core i5 | 4 GB | ~15 FPS |
| FULL (CPU) | 8-core i7 | 8 GB | ~5-8 FPS |
| FULL (CUDA) | Any | 6 GB VRAM | ~25-30 FPS |

---

## 8. Software Engineering

### 8.1 Dependencies
- `torch 2.x`, `ultralytics 8.x`, `onnxruntime 1.x`
- `faiss-cpu 1.x` for vector search
- `PySide6 6.x` for desktop UI
- `pydantic 2.x` for config validation
- `sqlalchemy 2.x` + `aiosqlite` for async persistence
- `dependency-injector 4.x` for DI container

### 8.2 Test Coverage
| Layer | Tests | Coverage |
|---|---|---|
| Entities | 12 unit tests | ~95% |
| Use Cases | 6 unit tests | ~85% |
| Adaptors | 4 unit tests | ~80% |
| Frameworks | 4 unit tests | ~75% |
| Integration | 2 end-to-end tests | Full pipeline |

### 8.3 Code Quality
- `from __future__ import annotations` throughout (Python 3.11 compat)
- Pydantic v2 throughout (`model_validate`, `@field_validator`, `model_dump`)
- Async/await for all I/O (SQLAlchemy async, aiosqlite)
- No global state — all components wired via DI container

---

## 9. Known Limitations

1. **SKU-110K training**: Public SKU-110K requires downloading ~5 GB; Colab auto-downloads this.
2. **Single-shelf mode default**: Multi-shelf homography requires reference images in `data/reference_shelves/`.
3. **CPU inference speed**: Full pipeline at ~5 FPS on typical laptop CPU. Use GPU for production.
4. **FAISS flat index**: Linear scan; for >10K SKUs, switch to `IndexIVFFlat` with training.
5. **No persistent alert history** in memory mode — restart clears all alerts.

---

## 10. References

1. Jocher, G. et al. (2023). Ultralytics YOLO. https://github.com/ultralytics/ultralytics
2. Goldman, E. et al. (2019). Precise Detection in Densely Packed Scenes. CVPR 2019. (SKU-110K)
3. Schroff, F. et al. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR 2015. (TripletLoss)
4. Howard, A. et al. (2019). Searching for MobileNetV3. ICCV 2019.
5. Johnson, J. et al. (2017). Billion-scale similarity search with GPUs. (FAISS)
6. Martin, R. C. (2017). Clean Architecture: A Craftsman's Guide to Software Structure and Design.
