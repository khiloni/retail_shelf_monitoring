"""Packages the repository into project_for_colab.zip for training on Google Colab or Kaggle."""
from __future__ import annotations

import zipfile
from pathlib import Path


def package_project(output_zip: str | Path = "project_for_colab.zip") -> Path:
    out_p = Path(output_zip)
    root = Path(".")

    include_patterns = [
        "shelf_monitor/**/*.py",
        "shelf_monitor/**/*.qss",
        "training/**/*.py",
        "scripts/**/*.py",
        "requirements-train.txt",
        "requirements.txt",
        "pyproject.toml",
        "config.example.yaml",
        "README.md",
    ]

    exclude_dirs = [
        ".venv",
        ".git",
        "__pycache__",
        ".pytest_cache",
        "outputs",
        "data",
        "models",
        "reference",
    ]

    print(f"Creating {out_p} for Colab/Kaggle upload...")

    files_added = 0
    with zipfile.ZipFile(str(out_p), "w", zipfile.ZIP_DEFLATED) as zf:
        for pattern in include_patterns:
            for p in root.glob(pattern):
                if any(ex in p.parts for ex in exclude_dirs):
                    continue
                if p.is_file():
                    zf.write(p, arcname=p.as_posix())
                    files_added += 1

    print(f"[OK] Packed {files_added} files into {out_p} ({out_p.stat().st_size // 1024} KB)")
    print(f"  Upload this file to Google Colab or Kaggle when running training/colab_train.ipynb")
    return out_p


if __name__ == "__main__":
    package_project()
