"""Export trained PyTorch models to ONNX and verify numerical parity."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from shelf_monitor.frameworks.pytorch_models.embedding_net import EmbeddingNet


def export_detector_onnx(
    pt_path: str | Path,
    out_path: str | Path,
    imgsz: int = 640,
) -> Path:
    """Exports Ultralytics YOLO model to ONNX."""
    from ultralytics import YOLO

    pt_p = Path(pt_path)
    out_p = Path(out_path)

    print(f"Exporting detector: {pt_p} -> {out_p}...")
    model = YOLO(str(pt_p))
    exported_file = model.export(format="onnx", imgsz=imgsz, dynamic=False, opset=12)

    # Move to expected target if needed
    if Path(exported_file) != out_p:
        import shutil
        shutil.copy2(exported_file, out_p)

    print(f"✓ Detector ONNX export saved to: {out_p}")
    return out_p


def export_embedding_onnx(
    pt_path: str | Path,
    out_path: str | Path,
    embedding_dim: int = 256,
    input_size: int = 224,
) -> Path:
    """Exports EmbeddingNet PyTorch model to ONNX and runs numerical parity check."""
    pt_p = Path(pt_path)
    out_p = Path(out_path)

    print(f"Exporting embedding model: {pt_p} -> {out_p}...")
    model = EmbeddingNet(embedding_dim=embedding_dim, input_size=input_size, pretrained=False)
    if pt_p.exists():
        state = torch.load(str(pt_p), map_location="cpu")
        model.load_state_dict(state)

    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)

    torch.onnx.export(
        model,
        dummy_input,
        str(out_p),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        opset_version=12,
    )

    # Parity check
    with torch.no_grad():
        pt_out = model(dummy_input).numpy()

    sess = ort.InferenceSession(str(out_p), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy_input.numpy()})[0]

    max_diff = np.max(np.abs(pt_out - onnx_out))
    print(f"  Parity check: max absolute difference = {max_diff:.6f}")
    if max_diff < 1e-4:
        print("  [✓] PyTorch vs ONNX parity verified (< 1e-4)")
    else:
        print("  [!] Warning: slight divergence between PyTorch and ONNX")

    print(f"✓ Embedding ONNX export saved to: {out_p}")
    return out_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX")
    parser.add_argument("--detector-pt", default="outputs/models/product_detector.pt")
    parser.add_argument("--detector-onnx", default="outputs/models/product_detector.onnx")
    parser.add_argument("--embedding-pt", default="outputs/models/sku_embedding.pt")
    parser.add_argument("--embedding-onnx", default="outputs/models/sku_embedding.onnx")
    args = parser.parse_args()

    if Path(args.detector_pt).exists():
        export_detector_onnx(args.detector_pt, args.detector_onnx)
    if Path(args.embedding_pt).exists():
        export_embedding_onnx(args.embedding_pt, args.embedding_onnx)
