import os

import pytest

from optiresearch.hsi.public_datasets import LocalNPZHSIAdapter


@pytest.mark.skipif(not os.getenv("OPTIRESEARCH_HSI_DATASET_PATH"), reason="local HSI dataset path not configured")
def test_real_hsi_dataset_adapter_can_prepare_configured_path(tmp_path):
    adapter = LocalNPZHSIAdapter()

    result = adapter.prepare(tmp_path / "local_npz")

    assert result["status"] == "prepared"
    assert (tmp_path / "local_npz" / "dataset_manifest.json").exists()

