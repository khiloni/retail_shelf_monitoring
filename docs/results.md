# Evaluation Results & Ablation Study

## Summary

| Metric | Target | Achieved (Colab) |
|---|---|---|
| Detector mAP50 (SKU-110K) | ≥ 0.75 | *Fill after training* |
| Detector mAP50-95 (SKU-110K) | ≥ 0.55 | *Fill after training* |
| SKU Rank-1 Accuracy | ≥ 0.80 | *Fill after training* |
| SKU mAP (retrieval) | ≥ 0.70 | *Fill after training* |
| Alert Precision | ≥ 0.90 | *Fill after training* |
| Alert Recall | ≥ 0.85 | *Fill after training* |
| Inference Speed (CPU, FPS) | ≥ 5 | *Fill after benchmark* |
| Inference Speed (GPU T4, FPS) | ≥ 25 | *Fill after benchmark* |

---

## 1. Detector — YOLOv11n on SKU-110K

### 1.1 Overall Results

| Epoch | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| 25 | *TBD* | *TBD* | *TBD* | *TBD* |
| 50 | *TBD* | *TBD* | *TBD* | *TBD* |
| 75 | *TBD* | *TBD* | *TBD* | *TBD* |
| 100 | *TBD* | *TBD* | *TBD* | *TBD* |

> **Note**: Fill in these values after completing training on Colab. The `training/evaluate.py` script prints a JSON summary that can be copied here.

### 1.2 Ablation — Input Resolution

| Image Size | mAP50 | Inference (ms/frame) |
|---|---|---|
| 320×320 | *TBD* | *TBD* |
| 640×640 *(default)* | *TBD* | *TBD* |
| 1280×1280 | *TBD* | *TBD* |

### 1.3 Ablation — Confidence Threshold

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.20 | *TBD* | *TBD* | *TBD* |
| 0.35 *(default)* | *TBD* | *TBD* | *TBD* |
| 0.50 | *TBD* | *TBD* | *TBD* |
| 0.65 | *TBD* | *TBD* | *TBD* |

---

## 2. SKU Recognizer — MobileNetV3 + TripletLoss

### 2.1 Overall Results

| Epoch | Rank-1 | Rank-5 | mAP (retrieval) |
|---|---|---|---|
| 10 | *TBD* | *TBD* | *TBD* |
| 25 | *TBD* | *TBD* | *TBD* |
| 50 | *TBD* | *TBD* | *TBD* |

### 2.2 Ablation — Embedding Dimension

| Embedding Dim | Rank-1 | Index Size | Query Time (ms) |
|---|---|---|---|
| 64 | *TBD* | *TBD* | *TBD* |
| 128 | *TBD* | *TBD* | *TBD* |
| 256 *(default)* | *TBD* | *TBD* | *TBD* |
| 512 | *TBD* | *TBD* | *TBD* |

### 2.3 Ablation — Number of Reference Crops per SKU

| Crops / SKU | Rank-1 | Rank-5 |
|---|---|---|
| 1 | *TBD* | *TBD* |
| 5 | *TBD* | *TBD* |
| 10 *(default)* | *TBD* | *TBD* |
| 20 | *TBD* | *TBD* |

---

## 3. Planogram Compliance

### 3.1 Cell State Accuracy (Synthetic Ground Truth)

| Cell State | Precision | Recall | F1 |
|---|---|---|---|
| OK | *TBD* | *TBD* | *TBD* |
| EMPTY | *TBD* | *TBD* | *TBD* |
| MISPLACED | *TBD* | *TBD* | *TBD* |

### 3.2 Ablation — Temporal Consensus Window

| Min Consecutive Frames | Alert Precision | False Alert Rate |
|---|---|---|
| 1 (no filtering) | *TBD* | *TBD* |
| 3 *(default)* | *TBD* | *TBD* |
| 5 | *TBD* | *TBD* |
| 10 | *TBD* | *TBD* |

---

## 4. End-to-End Latency Benchmark

| Component | CPU (i7-12th gen) | GPU (T4 Colab) |
|---|---|---|
| YOLO detection | *TBD* ms | *TBD* ms |
| SORT tracking | *TBD* ms | *TBD* ms |
| SKU embedding | *TBD* ms | *TBD* ms |
| FAISS search | *TBD* ms | *TBD* ms |
| Grid clustering | *TBD* ms | *TBD* ms |
| Cell state compute | *TBD* ms | *TBD* ms |
| **Total per frame** | ***TBD* ms** | ***TBD* ms** |
| **Effective FPS** | ***TBD*** | ***TBD*** |

---

## 5. System Mode Comparison

| Mode | Requirements | OOS Detection | SKU ID | Misplacement |
|---|---|---|---|---|
| BASELINE | None | ✓ (coarse) | ✗ | ✗ |
| POSITION-ONLY | `product_detector.*` | ✓ (precise) | ✗ | ✗ |
| FULL | All 7 model files | ✓ (precise) | ✓ | ✓ |

---

## Filling in Results

After training on Colab, run the evaluation script:

```bash
# From Colab (at end of training):
python training/evaluate.py \
  --detector models/product_detector.pt \
  --embedding models/sku_embedding.pt \
  --index models/sku_index.faiss \
  --labels models/sku_labels.json \
  --data-yaml SKU-110K.yaml \
  --out results/eval_report.json

# View summary
cat results/eval_report.json
```

Copy the JSON values into this document to complete the results table.
