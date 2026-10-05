# 🛒 Retail Shelf Monitoring System

A production-grade, deep-learning-powered retail shelf monitoring system using YOLO object detection, metric learning for SKU recognition, and a clean PySide6 desktop UI.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Product Detection** | YOLOv11n fine-tuned on SKU-110K (≥0.75 mAP50) |
| **SKU Recognition** | MobileNetV3 + TripletLoss metric learning (≥0.80 Rank-1) |
| **Planogram Compliance** | DBSCAN grid clustering → OK / EMPTY / MISPLACED / UNKNOWN |
| **Real-time Alerts** | Out-of-stock & misplacement with temporal consensus filtering |
| **Desktop UI** | PySide6 multi-pane live dashboard |
| **Headless CLI** | Analyze images/videos, enroll SKUs, generate planograms |
| **Offline Demo** | Zero-download synthetic demo (no GPU required) |

---

## 📁 Project Structure

```
DL-PROJECT/
├── shelf_monitor/          # Main Python package (Clean Architecture)
│   ├── entities/           # Domain models (Pydantic v2)
│   ├── usecases/           # Business logic + interfaces
│   ├── adaptors/           # YOLO, SORT, FAISS, SQLite adapters
│   └── frameworks/         # Config, DB, UI, inference engines
├── training/               # Training scripts + Colab notebook
│   └── colab_train.ipynb   # ← Train on Google Colab
├── models/                 # ← Put trained models HERE
├── tests/                  # Unit + integration tests
├── scripts/                # Utility scripts
├── docs/                   # Architecture & technical docs
├── data/                   # Reference images & datasets
├── outputs/                # Analysis results (auto-created)
├── config.yaml             # Main configuration file
└── requirements.txt        # Python dependencies
```

---

## 🚀 Quick Start (After Training)

### Step 1 — Train on Google Colab
Open `training/colab_train.ipynb` on [Google Colab](https://colab.research.google.com) with **GPU runtime enabled**, run all cells, and download `shelf_models.zip` when prompted.

### Step 2 — Place Models
Unzip `shelf_models.zip` and copy all files into the `models/` folder:
```
models/
├── product_detector.pt
├── product_detector.onnx
├── sku_embedding.pt
├── sku_embedding.onnx
├── sku_index.faiss
├── sku_labels.json
└── model_manifest.json
```

### Step 3 — Verify
```bash
python -m shelf_monitor check-models
```

### Step 4 — Run
```bash
# Desktop UI
python -m shelf_monitor ui

# Analyze a video (headless)
python -m shelf_monitor analyze-video --video path/to/video.mp4 --shelf-id S1

# Offline demo (no models needed)
python -m shelf_monitor demo
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.11.x
- Windows 10/11 (also works on Linux/macOS)
- No GPU required locally (training done on Colab)

### Install
```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) for training locally
pip install -r requirements-train.txt
```

---

## 🖥️ CLI Commands

| Command | Description |
|---|---|
| `check-models` | Validate model files in `models/` |
| `make-planogram` | Generate planogram from reference shelf image |
| `analyze-image` | Analyze single image for compliance |
| `analyze-video` | Analyze video stream for compliance |
| `enroll-skus` | Build FAISS index from reference SKU crops |
| `demo` | Run offline synthetic demo |
| `ui` | Launch PySide6 desktop UI |

```bash
# Examples
python -m shelf_monitor check-models
python -m shelf_monitor make-planogram --image shelf_ref.jpg --shelf-id S1
python -m shelf_monitor analyze-image --image shelf.jpg --shelf-id S1 --out outputs/
python -m shelf_monitor analyze-video --video camera.mp4 --shelf-id S1
python -m shelf_monitor enroll-skus --crops data/sku_crops/ --out-dir models/
python -m shelf_monitor demo --out outputs/demo/
python -m shelf_monitor ui
```

---

## 🤖 Model Operating Modes

The system automatically detects which models are present:

| Mode | Files Required | Capabilities |
|---|---|---|
| **BASELINE** | None (uses stock YOLO) | Detection only, no SKU ID |
| **POSITION-ONLY** | `product_detector.*` | OK / EMPTY states |
| **FULL** | All 7 files | Full pipeline + SKU recognition |

Override model directory:
```bash
set SHELF_MODELS_DIR=path/to/models  # Windows
python -m shelf_monitor ui
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize behavior. Key settings:

```yaml
app:
  models_dir: models          # Where to look for trained models
  debug: false

ml:
  confidence: 0.35            # Detection confidence threshold
  device: auto                # auto / cpu / cuda

stream:
  keyframe_interval: 15       # Process every Nth frame
  diff_threshold: 0.05        # Minimum visual change to process frame

consensus:
  min_consecutive_frames: 3   # Frames before reporting state change
```

---

## 🧪 Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=shelf_monitor/usecases --cov-report=term-missing

# Run only unit tests
python -m pytest tests/unit/ -v
```

---

## 📚 Documentation

- [`docs/architecture.md`](docs/architecture.md) — System architecture with diagrams
- [`docs/technical_report.md`](docs/technical_report.md) — Full technical report
- [`docs/results.md`](docs/results.md) — Evaluation results & ablation study
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — Operations runbook

---

## 📦 Model Contract

Files produced by `training/colab_train.ipynb` and expected in `models/`:

| File | Description |
|---|---|
| `product_detector.pt` | YOLOv11n detector (PyTorch) |
| `product_detector.onnx` | Detector (ONNX, for inference engine) |
| `sku_embedding.pt` | MobileNetV3 embedding model (PyTorch) |
| `sku_embedding.onnx` | Embedding model (ONNX) |
| `sku_index.faiss` | FAISS nearest-neighbour index |
| `sku_labels.json` | Class label mapping |
| `model_manifest.json` | Metadata: epoch, mAP, rank-1, embed_dim |

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.
