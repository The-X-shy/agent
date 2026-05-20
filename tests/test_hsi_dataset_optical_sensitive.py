"""Test optical-sensitive synthetic HSI dataset patterns."""

import numpy as np

from optiresearch.hsi.dataset import SyntheticHSIDataset
from optiresearch.schemas.hsi import build_default_synthetic_hsi_dataset_spec


def test_mixed_materials_deterministic(tmp_path):
    spec = build_default_synthetic_hsi_dataset_spec(
        train_size=3, val_size=2, test_size=2, height=16, width=16, spectral_bands=8,
        spectral_pattern_type="mixed_materials", material_count=4, depth_aware=True,
    )
    first = SyntheticHSIDataset(spec, seed=7)
    second = SyntheticHSIDataset(spec, seed=7)
    train_a = first.generate_split("train")
    train_b = second.generate_split("train")
    assert train_a["hsi"].shape == (3, 8, 16, 16)
    assert np.allclose(train_a["hsi"], train_b["hsi"])


def test_mixed_materials_depth_labels(tmp_path):
    spec = build_default_synthetic_hsi_dataset_spec(
        train_size=9, val_size=2, test_size=2, height=16, width=16, spectral_bands=8,
        spectral_pattern_type="mixed_materials",
    )
    dataset = SyntheticHSIDataset(spec, seed=7)
    train = dataset.generate_split("train")
    assert len(set(int(d) for d in train["depth_indices"])) >= 2


def test_backward_compat_smooth_low_rank(tmp_path):
    spec = build_default_synthetic_hsi_dataset_spec(
        train_size=3, val_size=2, test_size=2, height=16, width=16, spectral_bands=8,
        spectral_pattern_type="smooth_low_rank",
    )
    dataset = SyntheticHSIDataset(spec, seed=7)
    train = dataset.generate_split("train")
    assert train["hsi"].shape == (3, 8, 16, 16)
    assert train["hsi"].max() <= 1.0


def test_sparse_peaks_pattern(tmp_path):
    spec = build_default_synthetic_hsi_dataset_spec(
        train_size=3, val_size=2, test_size=2, height=16, width=16, spectral_bands=8,
        spectral_pattern_type="sparse_peaks",
    )
    dataset = SyntheticHSIDataset(spec, seed=7)
    train = dataset.generate_split("train")
    assert train["hsi"].shape == (3, 8, 16, 16)
    assert train["hsi"].min() >= 0.0


def test_edge_spectral_contrast_pattern(tmp_path):
    spec = build_default_synthetic_hsi_dataset_spec(
        train_size=3, val_size=2, test_size=2, height=16, width=16, spectral_bands=8,
        spectral_pattern_type="edge_spectral_contrast",
    )
    dataset = SyntheticHSIDataset(spec, seed=7)
    train = dataset.generate_split("train")
    assert train["hsi"].shape == (3, 8, 16, 16)


def test_save_with_new_pattern(tmp_path):
    spec = build_default_synthetic_hsi_dataset_spec(
        train_size=3, val_size=2, test_size=2, height=16, width=16, spectral_bands=8,
        spectral_pattern_type="mixed_materials",
    )
    dataset = SyntheticHSIDataset(spec, seed=7)
    manifest = dataset.save(tmp_path)
    assert manifest["spectral_pattern_type"] == "mixed_materials"
    assert (tmp_path / "dataset_manifest.json").exists()
