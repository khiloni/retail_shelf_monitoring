"""Build FAISS index and sku_labels.json from reference product crops."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import faiss
import numpy as np
import torch

from shelf_monitor.frameworks.pytorch_models.embedding_net import EmbeddingNet


def build_index_from_crops(
    crops_dir: str | Path,
    output_dir: str | Path = "models",
    embedding_model_path: Optional[str | Path] = None,
    embedding_dim: int = 256,
    input_size: int = 224,
    device: str = "cpu",
) -> Tuple[int, int]:
    """Generates sku_index.faiss and sku_labels.json from class-per-folder crops."""
    crops_p = Path(crops_dir)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    model = EmbeddingNet(embedding_dim=embedding_dim, input_size=input_size, pretrained=False)
    if embedding_model_path and Path(embedding_model_path).exists():
        state = torch.load(str(embedding_model_path), map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded embedding weights from {embedding_model_path}")
    else:
        print("Using base/initialized EmbeddingNet weights")

    model.eval()
    dev = torch.device(device)
    model.to(dev)

    # 2. Gather crops
    class_dirs = [d for d in sorted(crops_p.iterdir()) if d.is_dir()]
    if not class_dirs:
        # Check if there are loose images
        loose_images = list(crops_p.glob("*.*"))
        if not loose_images:
            raise ValueError(f"No crop images found in {crops_dir}")
        class_dirs = [crops_p]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    embeddings_list: List[np.ndarray] = []
    labels_map: Dict[str, Dict[str, str]] = {}
    vector_idx = 0

    for cdir in class_dirs:
        sku_id = cdir.name if cdir != crops_p else "sku_0"
        for img_path in cdir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            im = cv2.imread(str(img_path))
            if im is None:
                continue

            im = cv2.resize(im, (input_size, input_size))
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            norm = (rgb - mean) / std
            t = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(dev)

            with torch.no_grad():
                emb = model(t).cpu().numpy()[0]  # Already L2-normalized

            embeddings_list.append(emb)
            labels_map[str(vector_idx)] = {
                "sku_id": sku_id,
                "name": sku_id.replace("_", " ").title(),
                "image_path": str(img_path),
            }
            vector_idx += 1

    if not embeddings_list:
        raise ValueError("No valid image embeddings were extracted")

    # 3. Create FAISS Inner-Product Index (cosine similarity on L2-normalized vectors)
    emb_matrix = np.vstack(embeddings_list).astype(np.float32)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(emb_matrix)

    # 4. Save artifacts
    index_file = out_p / "sku_index.faiss"
    faiss.write_index(index, str(index_file))

    labels_file = out_p / "sku_labels.json"
    with labels_file.open("w", encoding="utf-8") as f:
        json.dump(labels_map, f, indent=2)

    unique_skus = len(set(v["sku_id"] for v in labels_map.values()))
    print(f"✓ FAISS index built: {index.ntotal} vectors, {unique_skus} unique SKUs")
    print(f"  Index saved:  {index_file}")
    print(f"  Labels saved: {labels_file}")
    return index.ntotal, unique_skus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from SKU reference crops")
    parser.add_argument("--crops", required=True, help="Directory containing SKU crops")
    parser.add_argument("--model", help="Path to sku_embedding.pt")
    parser.add_argument("--out-dir", default="models", help="Output directory")
    parser.add_argument("--dim", type=int, default=256, help="Embedding dimension")
    args = parser.parse_args()

    build_index_from_crops(
        crops_dir=args.crops,
        output_dir=args.out_dir,
        embedding_model_path=args.model,
        embedding_dim=args.dim,
    )
