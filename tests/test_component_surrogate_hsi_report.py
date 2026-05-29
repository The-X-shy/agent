"""Report tests for component surrogate HSI co-design."""

from optiresearch.reports.component_surrogate_hsi_report import export_component_surrogate_hsi_report
from optiresearch.runtime.component_surrogate_hsi_codesign import run_component_surrogate_hsi_codesign
from optiresearch.schemas.component_surrogate_psf import ComponentSurrogateHSICoDesignSpec


def test_component_surrogate_hsi_report_contains_required_sections(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run_component_surrogate_hsi_codesign(
        ComponentSurrogateHSICoDesignSpec(
            component_type="fresnel",
            steps=3,
            band_count=4,
            image_size=16,
            psf_size=9,
            batch_size=1,
        )
    )

    path = export_component_surrogate_hsi_report(result.run_id)

    content = path.read_text()
    for section in [
        "Component backend source",
        "Surrogate PSF construction",
        "HSI forward model",
        "Reconstruction metrics",
        "Gradient flow",
        "Parameter update",
        "Claim boundary",
        "What not to claim",
    ]:
        assert section in content
    assert "full GeoLens lens-level optimization" in content
