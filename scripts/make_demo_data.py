"""Synthesizes shelf images and video with known ground truth for offline demo mode."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# Product SKU palette: color -> (B, G, R)
SKU_COLORS = {
    "sku_cereal_box": (30, 80, 220),       # Red/Orange
    "sku_soda_can": (220, 50, 40),          # Blue
    "sku_coffee_jar": (30, 180, 60),        # Green
    "sku_snack_pack": (20, 200, 240),       # Yellow
    "sku_tea_box": (160, 40, 180),          # Purple
}


def draw_product_box(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    sku_id: str,
    label: Optional[str] = None,
) -> None:
    color = SKU_COLORS.get(sku_id, (120, 120, 120))
    # Fill product body
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    # Inner border / packaging line
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), 2)
    # SKU label text on the product
    text = label or sku_id.split("_")[1][:6]
    cv2.putText(img, text, (x + 4, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def generate_reference_shelf(
    width: int = 800,
    height: int = 600,
    num_rows: int = 3,
    items_per_row: int = 5,
) -> Tuple[np.ndarray, List[Dict]]:
    """Generates a perfect reference shelf image and its ground truth planogram items."""
    img = np.full((height, width, 3), 235, dtype=np.uint8)

    row_height = height // (num_rows + 1)
    sku_keys = list(SKU_COLORS.keys())

    planogram_items = []
    item_w = 90
    item_h = 110

    for r in range(num_rows):
        y_shelf = (r + 1) * row_height
        # Draw physical shelf bar
        cv2.rectangle(img, (40, y_shelf + item_h + 2), (width - 40, y_shelf + item_h + 10), (140, 140, 150), -1)

        x_spacing = (width - 100 - items_per_row * item_w) // (items_per_row - 1)
        for i in range(items_per_row):
            x = 50 + i * (item_w + x_spacing)
            y = y_shelf
            sku = sku_keys[(r * items_per_row + i) % len(sku_keys)]

            draw_product_box(img, x, y, item_w, item_h, sku)
            planogram_items.append(
                {
                    "row_idx": r,
                    "item_idx": i,
                    "sku_id": sku,
                    "bbox": [x, y, x + item_w, y + item_h],
                }
            )

    return img, planogram_items


def generate_demo_video(
    ref_items: List[Dict],
    out_video_path: str | Path,
    num_frames: int = 60,
    width: int = 800,
    height: int = 600,
    num_rows: int = 3,
) -> None:
    """Generates a synthetic CCTV video where products disappear (OOS) and swap (misplaced)."""
    out_p = Path(out_video_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, 15.0, (width, height))
    row_height = height // (num_rows + 1)
    item_h = 110

    # Pick test events:
    # Cell (row 1, item 2) becomes EMPTY starting at frame 10
    # Cell (row 0, item 3) gets MISPLACED product starting at frame 15
    for f in range(num_frames):
        frame = np.full((height, width, 3), 235, dtype=np.uint8)

        # Draw shelf bars
        for r in range(num_rows):
            y_shelf = (r + 1) * row_height
            cv2.rectangle(frame, (40, y_shelf + item_h + 2), (width - 40, y_shelf + item_h + 10), (140, 140, 150), -1)

        for item in ref_items:
            r = item["row_idx"]
            i = item["item_idx"]
            x1, y1, x2, y2 = item["bbox"]
            bw, bh = x2 - x1, y2 - y1
            sku = item["sku_id"]

            # Scenario 1: Cell (1, 2) becomes OUT OF STOCK after frame 10
            if r == 1 and i == 2 and f >= 10:
                continue  # empty shelf gap!

            # Scenario 2: Cell (0, 3) gets MISPLACED after frame 15
            if r == 0 and i == 3 and f >= 15:
                sku = "sku_tea_box"  # Wrong SKU!

            draw_product_box(frame, x1, y1, bw, bh, sku)

        # Add slight camera jitter / noise for realism
        noise = np.random.normal(0, 2, frame.shape).astype(np.int16)
        noisy_frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Timestamp banner
        cv2.putText(
            noisy_frame,
            f"SYNTHETIC DEMO FEED - FRAME {f:03d}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 40, 40),
            2,
        )
        writer.write(noisy_frame)

    writer.release()
    print(f"✓ Generated synthetic demo video ({num_frames} frames) at: {out_p}")


def generate_crop_database(ref_items: List[Dict], ref_img: np.ndarray, out_dir: str | Path) -> None:
    """Saves reference crops organized in class-per-folder structure."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(ref_items):
        sku = item["sku_id"]
        x1, y1, x2, y2 = item["bbox"]
        crop = ref_img[y1:y2, x1:x2]

        cdir = out_p / sku
        cdir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cdir / f"crop_{idx}.jpg"), crop)

    print(f"✓ Saved SKU reference crops into: {out_p}")


def make_all_demo_data(data_dir: str | Path = "data/demo") -> Tuple[Path, Path, Path]:
    d_p = Path(data_dir)
    d_p.mkdir(parents=True, exist_ok=True)

    ref_img_path = d_p / "shelf_reference.jpg"
    video_path = d_p / "shelf_stream.mp4"
    crops_dir = d_p / "sku_crops"

    ref_img, items = generate_reference_shelf()
    cv2.imwrite(str(ref_img_path), ref_img)
    print(f"✓ Generated reference shelf image: {ref_img_path}")

    generate_demo_video(items, video_path)
    generate_crop_database(items, ref_img, crops_dir)

    # Save ground truth metadata
    meta = {
        "synthetic": True,
        "items": items,
        "events": [
            {"frame": 10, "cell": [1, 2], "event": "EMPTY (Out of Stock)"},
            {"frame": 15, "cell": [0, 3], "event": "MISPLACED (Wrong SKU)"},
        ],
    }
    with (d_p / "ground_truth.json").open("w") as f:
        json.dump(meta, f, indent=2)

    return ref_img_path, video_path, crops_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic demo dataset")
    parser.add_argument("--out-dir", default="data/demo", help="Target demo directory")
    args = parser.parse_args()
    make_all_demo_data(args.out_dir)
