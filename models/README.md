# Model Directory

This directory is the single fixed location where your trained models are auto-loaded by the Retail Shelf Monitoring system.

## Required Files from Google Colab / Kaggle

After running `training/colab_train.ipynb`, download `shelf_models.zip` and extract its contents directly into this folder (`models/`).

The exact files required:

| Filename | Purpose |
|---|---|
| `product_detector.pt` | Trained YOLO11 product detector weights (Ultralytics) |
| `product_detector.onnx` | Exported ONNX detector for fast CPU inference (preferred) |
| `sku_embedding.pt` | MobileNetV3-Large embedding network state_dict |
| `sku_embedding.onnx` | Exported ONNX embedding model (optional) |
| `sku_index.faiss` | FAISS index of reference SKU vectors |
| `sku_labels.json` | Mapping of FAISS vector indices to SKU labels & names |
| `model_manifest.json` | Metadata manifest (input sizes, thresholds, class names) |

After copying these files here, **no code or configuration changes are required**. The system automatically detects them on launch and activates **FULL** operational mode.
