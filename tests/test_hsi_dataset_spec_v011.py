import json

from optiresearch.schemas.hsi import HSIDatasetSpec, build_default_synthetic_hsi_dataset_spec, validate_hsi_dataset_spec


def test_hsi_dataset_spec_v011_defaults_are_backward_compatible(tmp_path):
    spec = HSIDatasetSpec(
        dataset_id="old_synthetic",
        dataset_name="old synthetic",
        source="synthetic",
        spectral_bands=8,
        height=16,
        width=16,
        train_size=2,
        val_size=1,
        test_size=1,
        wavelength_range_nm=(450.0, 700.0),
    )

    assert spec.dataset_family == "synthetic"
    assert spec.normalization == "per_band"
    assert spec.crop_size == 32
    assert spec.patch_stride == 32
    assert spec.split_seed == 42
    assert validate_hsi_dataset_spec(spec)["status"] == "valid"


def test_hsi_dataset_spec_manifest_records_phase11_fields(tmp_path):
    spec = build_default_synthetic_hsi_dataset_spec(wavelengths_nm=[450.0, 550.0, 650.0], spectral_bands=3)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(spec.model_dump(mode="json"), indent=2), encoding="utf-8")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["dataset_family"] == "synthetic"
    assert payload["wavelengths_nm"] == [450.0, 550.0, 650.0]
    assert payload["data_license_note"] is not None


def test_validate_hsi_dataset_spec_reports_errors():
    spec = build_default_synthetic_hsi_dataset_spec()
    spec.crop_size = 64
    spec.height = 32
    spec.width = 32

    result = validate_hsi_dataset_spec(spec)

    assert result["status"] == "invalid"
    assert "CROP_SIZE_EXCEEDS_IMAGE" in result["error_codes"]

