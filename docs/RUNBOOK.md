# Operations Runbook

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [First-Time Setup](#first-time-setup)
4. [Loading Trained Models](#loading-trained-models)
5. [Starting the System](#starting-the-system)
6. [CLI Reference](#cli-reference)
7. [Configuration Reference](#configuration-reference)
8. [Troubleshooting](#troubleshooting)
9. [Logs](#logs)
10. [Updating Models](#updating-models)

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 | Windows 11 |
| Python | 3.11.x | 3.11.3 |
| RAM | 4 GB | 8 GB |
| CPU | 4-core | 8-core |
| GPU (local) | Not required | CUDA 11.8+ |
| GPU (training) | Google Colab | Colab Pro T4/A100 |

---

## Installation

```bash
# 1. Clone / download project
cd "D:\Degree_GLS\sem 7\DLL\DL-PROJECT"

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -m shelf_monitor check-models
```

---

## First-Time Setup

### 1. Copy example config (if config.yaml doesn't exist)
```bash
copy config.example.yaml config.yaml
```

### 2. Review config.yaml
Key settings to adjust for your environment:
```yaml
app:
  models_dir: models         # Where trained models live

ml:
  confidence: 0.35           # Lower = more detections, more false positives
  device: auto               # auto selects CPU if no CUDA

stream:
  keyframe_interval: 15      # Process 1 in every 15 frames for performance
```

---

## Loading Trained Models

After training on Google Colab (see `training/colab_train.ipynb`):

### Step 1: Download `shelf_models.zip` from Colab
The last notebook cell creates and downloads this file automatically.

### Step 2: Unzip into `models/`
```
models/
├── product_detector.pt       ← Required
├── product_detector.onnx     ← Required  
├── sku_embedding.pt          ← Required
├── sku_embedding.onnx        ← Required
├── sku_index.faiss           ← Required
├── sku_labels.json           ← Required
└── model_manifest.json       ← Required
```

### Step 3: Verify
```bash
python -m shelf_monitor check-models
```
Expected output:
```
✓ Model registry initialized
  Mode: FULL
  Detector: models/product_detector.onnx (ONNX)
  Embedding: models/sku_embedding.onnx (ONNX)
  Index: models/sku_index.faiss
  SKU classes: 117
```

---

## Starting the System

### Desktop UI
```bash
python -m shelf_monitor ui
```

### Headless Video Analysis
```bash
python -m shelf_monitor analyze-video \
  --video path/to/video.mp4 \
  --shelf-id S1 \
  --out outputs/runs/
```

### Offline Demo (no models needed)
```bash
python -m shelf_monitor demo --out outputs/demo/
```
The demo generates synthetic shelf images, runs the full pipeline, and saves:
- `demo_annotated_stream.mp4` — annotated video
- `alerts.json` — detected alerts
- `metrics_report.json` — compliance metrics

---

## CLI Reference

### `check-models`
Inspect model files and display operating mode.
```bash
python -m shelf_monitor check-models [--dir models/]
```

### `make-planogram`
Generate a planogram JSON from a reference shelf image.
```bash
python -m shelf_monitor make-planogram \
  --image data/reference_shelves/shelf_S1.jpg \
  --shelf-id S1
```

### `analyze-image`
Analyze a single shelf image.
```bash
python -m shelf_monitor analyze-image \
  --image shelf_photo.jpg \
  --shelf-id S1 \
  --out outputs/
```
Outputs: annotated image + JSON report.

### `analyze-video`
Process a video file or webcam feed.
```bash
# File
python -m shelf_monitor analyze-video --video footage.mp4 --shelf-id S1

# Webcam (index 0)
python -m shelf_monitor analyze-video --video 0 --shelf-id S1
```

### `enroll-skus`
Build the FAISS index from reference crop images.
```bash
python -m shelf_monitor enroll-skus \
  --crops data/sku_crops/ \
  --out-dir models/
```
`data/sku_crops/` must have subfolders named by SKU ID, each containing reference images:
```
data/sku_crops/
├── cereal_A/
│   ├── 001.jpg
│   └── 002.jpg
├── juice_B/
│   └── 001.jpg
```

### `demo`
Run offline synthetic demo.
```bash
python -m shelf_monitor demo [--out outputs/demo/]
```

### `ui`
Launch PySide6 desktop interface.
```bash
python -m shelf_monitor ui
```

---

## Configuration Reference

Full `config.yaml` structure:

```yaml
app:
  name: Retail Shelf Monitor
  debug: false
  models_dir: models              # Overridden by SHELF_MODELS_DIR env var

database:
  backend: sqlite                 # sqlite | postgres
  sqlite_path: outputs/shelf_monitor.db

alerts:
  backend: memory                 # memory | redis

ml:
  inference_engine: ultralytics  # ultralytics | onnxruntime
  confidence: 0.35
  iou: 0.45
  imgsz: 640
  device: auto                   # auto | cpu | cuda | cuda:0

sku:
  top_k: 3
  similarity_threshold: 0.6
  embedding_dim: 256
  input_size: 224
  re_id_every_k_frames: 10

grid:
  clustering_method: dbscan      # dbscan | kmeans
  eps: 15.0
  min_samples: 2
  position_tolerance: 1

aligner:
  single_shelf_mode: true        # true = skip alignment, use fixed_shelf_id
  fixed_shelf_id: shelf_1

stream:
  keyframe_interval: 15          # Process 1 in N frames
  diff_threshold: 0.05           # Minimum pixel diff to process a frame

tracking:
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3

consensus:
  min_consecutive_frames: 3
  clear_after_frames: 2
  state_timeout_sec: 300

logging:
  level: INFO                    # DEBUG | INFO | WARNING | ERROR
  format: text                   # text | json
```

### Environment Variable Overrides
```bash
set SHELF_MODELS_DIR=D:\custom\models   # Override models directory
```

---

## Troubleshooting

### `check-models` shows BASELINE mode
**Problem**: No trained models found in `models/`.  
**Fix**: Complete training on Colab, download `shelf_models.zip`, unzip into `models/`.

### `ModuleNotFoundError: No module named 'shelf_monitor'`
**Fix**: Make sure virtual environment is activated:
```bash
.venv\Scripts\activate
```

### Camera/video not opening
**Fix**: Check the path or webcam index. For webcam: `--video 0`.

### High CPU / slow inference
**Fix**: Increase `keyframe_interval` in `config.yaml`:
```yaml
stream:
  keyframe_interval: 30   # Process 1 in 30 frames
```

### All cells show UNKNOWN state
**Cause**: No planogram loaded for the shelf ID.  
**Fix**: Run `make-planogram` first with a reference image.

### PySide6 UI won't start / crashes
**Fix**: Ensure display environment is available. On headless servers, the UI command requires a display.

### SQLite database locked
**Fix**: Kill any other running instances, or change `sqlite_path` to a new path in `config.yaml`.

### ONNX inference slower than expected
**Fix**: Switch to ultralytics engine:
```yaml
ml:
  inference_engine: ultralytics
```

---

## Logs

Logs are printed to console by default. To log to a file:
```yaml
logging:
  level: DEBUG
  file_path: outputs/logs/shelf_monitor.log
```

Log files rotate automatically when they exceed 10 MB.

---

## Updating Models

To retrain and update models:
1. Modify training hyperparameters in `training/colab_train.ipynb`
2. Re-run the notebook on Colab
3. Download the new `shelf_models.zip`
4. **Replace** all files in `models/` with the new ones
5. Run `python -m shelf_monitor check-models` to verify

> **No code changes required** — the system reads model metadata from `model_manifest.json` automatically.
