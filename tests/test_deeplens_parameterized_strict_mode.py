"""Test strict_deeplens mode in parameterized PSF generator."""
from optiresearch.adapters.deeplens_parameterized_psf import DeepLensParameterizedPSFGenerator


def test_strict_mode_without_deeplens_returns_unavailable(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPLENS_REPO_PATH", raising=False)
    output_dir = tmp_path / "strict_output"
    output_dir.mkdir()

    gen = DeepLensParameterizedPSFGenerator(strict_deeplens=True)
    # Force unavailable
    gen._deeplens = None

    result = gen.generate_psf_cube(
        {"surface_curvature": 0.5, "chromatic_shift": 0.3},
        output_dir,
    )

    assert result["status"] == "unavailable"
    assert result["psf_cube"] is None
    assert result["fallback_used"] is False
    assert result["error_code"] == "DEEPLENS_UNAVAILABLE"


def test_non_strict_mode_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPLENS_REPO_PATH", raising=False)
    output_dir = tmp_path / "nonstrict_output"
    output_dir.mkdir()

    gen = DeepLensParameterizedPSFGenerator(strict_deeplens=False)
    gen._deeplens = None

    result = gen.generate_psf_cube(
        {"surface_curvature": 0.5, "chromatic_shift": 0.3},
        output_dir,
    )

    assert result["status"] == "fallback"
    assert result["fallback_used"] is True
    assert "fallback_reason" in result


def test_strict_mode_includes_metadata(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPLENS_REPO_PATH", raising=False)
    output_dir = tmp_path / "meta_output"
    output_dir.mkdir()

    gen = DeepLensParameterizedPSFGenerator(strict_deeplens=True)
    gen._deeplens = None

    result = gen.generate_psf_cube(
        {"surface_curvature": 0.5},
        output_dir,
    )

    assert "metadata" in result
    assert result["metadata"]["strict_deeplens"] is True


def test_strict_mode_with_repo_deeplens_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPLENS_REPO_PATH", "/Users/lilin/Desktop/external/DeepLens")
    output_dir = tmp_path / "with_dl"
    output_dir.mkdir()

    gen = DeepLensParameterizedPSFGenerator(strict_deeplens=True)

    assert gen.deeplens_available is True
    assert gen._is_source_checkout is True

    result = gen.generate_psf_cube(
        {"surface_curvature": 0.5, "chromatic_shift": 0.3, "depth_variation": 0.5},
        output_dir,
    )

    assert "status" in result
    # Should either succeed, partial, or have structured error
    assert result["status"] in ("succeeded", "partial", "unavailable")
    if result["status"] in ("succeeded", "partial"):
        assert result["psf_cube"] is not None
        assert result["fallback_used"] is False
