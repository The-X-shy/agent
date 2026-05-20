import numpy as np

from optiresearch.hsi.dataset import SyntheticHSIDataset
from optiresearch.schemas.hsi import build_default_synthetic_hsi_dataset_spec


def test_synthetic_hsi_dataset_is_deterministic_and_saves(tmp_path):
    spec = build_default_synthetic_hsi_dataset_spec(train_size=3, val_size=2, test_size=2, height=16, width=16, spectral_bands=8)
    first = SyntheticHSIDataset(spec, seed=7)
    second = SyntheticHSIDataset(spec, seed=7)

    train_a = first.generate_split("train")
    train_b = second.generate_split("train")
    manifest = first.save(tmp_path)

    assert train_a["hsi"].shape == (3, 8, 16, 16)
    assert train_a["depth_indices"].shape == (3,)
    assert np.allclose(train_a["hsi"], train_b["hsi"])
    assert (tmp_path / "train.npz").exists()
    assert (tmp_path / "dataset_manifest.json").exists()
    assert manifest["dataset_id"] == spec.dataset_id
