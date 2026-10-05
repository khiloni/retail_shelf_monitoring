"""Prepare SKU-110K dataset for YOLO training.

SKU-110K is an object detection dataset for retail shelves (11,762 images, dense packaging).
Reference: Eran Goldman et al., 'Precise Detection in Densely Packed Scenes', CVPR 2019.

This script converts SKU-110K annotations into YOLO format:
  images/{train,val,test}/*.jpg
  labels/{train,val,test}/*.txt  (class_id x_center y_center width height, normalized)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from tqdm import tqdm


def convert_sku110k_to_yolo(
    data_dir: str | Path,
    output_dir: str | Path,
    max_images: int = -1,
    fraction: float = 1.0,
) -> None:
    """Converts SKU-110K annotations to YOLO structure.

    Expected input structure in data_dir:
      annotations/
        annotations_train.csv
        annotations_val.csv
        annotations_test.csv
      images/
        *.jpg
    """
    data_p = Path(data_dir)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]

    for split in splits:
        csv_file = data_p / "annotations" / f"annotations_{split}.csv"
        if not csv_file.exists():
            print(f"Notice: {csv_file} not found. If SKU-110K is not downloaded, use instructions below.")
            _print_download_instructions(data_p)
            return

        split_img_dir = out_p / "images" / split
        split_lbl_dir = out_p / "labels" / split
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Parse CSV: image_name, x1, y1, x2, y2, class, image_width, image_height
        img_boxes: Dict[str, List[Tuple[float, float, float, float, int, int]]] = {}
        with csv_file.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 8:
                    continue
                img_name, x1, y1, x2, y2, cls_name, w, h = row[:8]
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                w, h = int(w), int(h)
                img_boxes.setdefault(img_name, []).append((x1, y1, x2, y2, w, h))

        images_to_process = list(img_boxes.keys())
        if fraction < 1.0:
            count = max(1, int(len(images_to_process) * fraction))
            images_to_process = images_to_process[:count]
        if max_images > 0:
            images_to_process = images_to_process[:max_images]

        print(f"Converting {len(images_to_process)} images for {split} split...")
        for img_name in tqdm(images_to_process, desc=f"SKU110K {split}"):
            src_img = data_p / "images" / img_name
            dst_img = split_img_dir / img_name
            if src_img.exists() and not dst_img.exists():
                # Copy or symlink
                import shutil
                shutil.copy2(src_img, dst_img)

            lbl_file = split_lbl_dir / (Path(img_name).stem + ".txt")
            with lbl_file.open("w") as lf:
                for x1, y1, x2, y2, w, h in img_boxes[img_name]:
                    cx = ((x1 + x2) / 2.0) / w
                    cy = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    # All products are single class 0: "product"
                    lf.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # Write data.yaml
    yaml_content = f"""path: {out_p.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: product
"""
    with (out_p / "data.yaml").open("w") as yf:
        yf.write(yaml_content)
    print(f"✓ SKU-110K preparation complete. Dataset YAML at: {out_p / 'data.yaml'}")


def _print_download_instructions(target_path: Path) -> None:
    print("\n" + "=" * 60)
    print("SKU-110K Download Instructions (for Google Colab or Kaggle):")
    print("=" * 60)
    print("1. In Google Colab or Kaggle, you can download SKU-110K using:")
    print("   !git clone https://github.com/eg4000/SKU110K_CVPR19.git")
    print("   or via ultralytics auto-download: YOLO('yolo11s.pt').train(data='SKU-110K.yaml')")
    print(f"2. Or download directly from: http://data.brainimaging.org/sku110k/")
    print(f"   and extract into: {target_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SKU-110K dataset to YOLO format")
    parser.add_argument("--data-dir", default="data/SKU110K_fixed", help="Raw SKU-110K directory")
    parser.add_argument("--out-dir", default="data/sku110k_yolo", help="Output YOLO dataset directory")
    parser.add_argument("--max-images", type=int, default=-1, help="Limit max images (useful for CPU/smoke)")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use (0.0 to 1.0)")
    args = parser.parse_args()
    convert_sku110k_to_yolo(args.data_dir, args.out_dir, args.max_images, args.fraction)
