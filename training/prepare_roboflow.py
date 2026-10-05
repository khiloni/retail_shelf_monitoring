"""Prepare and merge Roboflow custom dataset into the training pipeline."""
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional


def prepare_roboflow_dataset(
    zip_path: Optional[str | Path] = None,
    api_key: Optional[str] = None,
    workspace: Optional[str] = None,
    project: Optional[str] = None,
    version: int = 1,
    output_dir: str | Path = "data/custom_yolo",
) -> Path:
    """Downloads or extracts a Roboflow YOLO export."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if zip_path and Path(zip_path).exists():
        print(f"Extracting Roboflow export from {zip_path}...")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(out_p))
        print(f"✓ Extracted dataset to {out_p}")
        return out_p

    if api_key and workspace and project:
        print(f"Downloading dataset from Roboflow ({workspace}/{project} v{version})...")
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=api_key)
            proj = rf.workspace(workspace).project(project)
            dataset = proj.version(version).download("yolov11", location=str(out_p))
            print(f"✓ Downloaded Roboflow dataset to {dataset.location}")
            return Path(dataset.location)
        except Exception as e:
            print(f"Error downloading from Roboflow: {e}")
            raise

    print("No Roboflow credentials or zip provided. Skipping custom data preparation.")
    return out_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Roboflow custom dataset")
    parser.add_argument("--zip", help="Path to Roboflow YOLO zip export")
    parser.add_argument("--api-key", help="Roboflow API key")
    parser.add_argument("--workspace", help="Roboflow workspace")
    parser.add_argument("--project", help="Roboflow project")
    parser.add_argument("--version", type=int, default=1, help="Roboflow version")
    parser.add_argument("--out-dir", default="data/custom_yolo", help="Output directory")
    args = parser.parse_args()
    prepare_roboflow_dataset(
        zip_path=args.zip,
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        output_dir=args.out_dir,
    )
