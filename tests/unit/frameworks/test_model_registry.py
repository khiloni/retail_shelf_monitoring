"""Unit tests for ModelRegistry (modes, dimension validation, missing files, corrupt manifest)."""
import json
from pathlib import Path
import pytest
from shelf_monitor.frameworks.exceptions import ModelCorruptError
from shelf_monitor.frameworks.model_registry import ModelMode, ModelRegistry


def test_model_registry_baseline_mode(tmp_path: Path):
    # Empty directory -> BASELINE mode
    reg = ModelRegistry(models_dir=tmp_path)
    assert reg.mode == ModelMode.BASELINE
    assert reg.embedding_path() is None
    assert reg.index_path() is None


def test_model_registry_position_only_mode(tmp_path: Path):
    # Only detector present -> POSITION-ONLY mode
    (tmp_path / "product_detector.pt").write_text("dummy_weights")
    reg = ModelRegistry(models_dir=tmp_path)
    assert reg.mode == ModelMode.POSITION_ONLY
    assert reg.embedding_path() is None


def test_model_registry_full_mode(tmp_path: Path):
    # All artifacts present -> FULL mode
    (tmp_path / "product_detector.pt").write_text("dummy")
    (tmp_path / "sku_embedding.pt").write_text("dummy")
    (tmp_path / "sku_index.faiss").write_text("dummy")
    (tmp_path / "sku_labels.json").write_text(json.dumps({"0": {"sku_id": "cereal"}}))

    manifest_data = {
        "schema_version": "1.0",
        "embedding": {"dim": 256},
        "index": {"dim": 256},
    }
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest_data))

    reg = ModelRegistry(models_dir=tmp_path)
    assert reg.mode == ModelMode.FULL
    assert reg.sku_id_for_index(0) == "cereal"


def test_model_registry_dim_mismatch_raises(tmp_path: Path):
    # Embedding dim != FAISS index dim -> ModelCorruptError
    (tmp_path / "product_detector.pt").write_text("dummy")
    (tmp_path / "sku_embedding.pt").write_text("dummy")
    (tmp_path / "sku_index.faiss").write_text("dummy")
    (tmp_path / "sku_labels.json").write_text(json.dumps({"0": "cereal"}))

    manifest_data = {
        "schema_version": "1.0",
        "embedding": {"dim": 256},
        "index": {"dim": 512},  # Mismatch!
    }
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest_data))

    with pytest.raises(ModelCorruptError):
        ModelRegistry(models_dir=tmp_path)
