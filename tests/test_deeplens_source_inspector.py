"""Test DeepLens source inspector."""
from optiresearch.adapters.deeplens_source_inspector import DeepLensSourceInspector


def _fake_deeplens_repo(tmp_path):
    repo = tmp_path / "DeepLens"
    package = repo / "deeplens"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "paraxiallens.py").write_text(
        "class ParaxialLens:\n"
        "    def psf(self):\n"
        "        pass\n"
        "    def get_optimizer(self):\n"
        "        pass\n",
        encoding="utf-8",
    )
    (package / "geolens.py").write_text(
        "class GeoLens:\n"
        "    def analysis_spot(self):\n"
        "        pass\n"
        "    def optimize(self):\n"
        "        pass\n",
        encoding="utf-8",
    )
    (package / "phase_surface.py").write_text(
        "class Phase:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (package / "diffractive_surface.py").write_text(
        "class Fresnel:\n"
        "    pass\n",
        encoding="utf-8",
    )
    return repo


def test_inspector_detects_repo_path(tmp_path):
    inspector = DeepLensSourceInspector(str(_fake_deeplens_repo(tmp_path)))
    assert inspector.available is True


def test_inspector_scan_returns_modules(tmp_path):
    inspector = DeepLensSourceInspector(str(_fake_deeplens_repo(tmp_path)))
    result = inspector.scan()
    assert result["available"] is True
    assert "modules" in result
    assert "paraxiallens" in result["modules"] or "geolens" in result["modules"]


def test_inspector_finds_classes(tmp_path):
    inspector = DeepLensSourceInspector(str(_fake_deeplens_repo(tmp_path)))
    result = inspector.scan()
    assert "classes" in result
    total_classes = sum(len(v) for v in result["classes"].values())
    assert total_classes > 0


def test_inspector_finds_likely_methods(tmp_path):
    inspector = DeepLensSourceInspector(str(_fake_deeplens_repo(tmp_path)))
    result = inspector.scan()

    assert "likely_psf_methods" in result
    assert "likely_optimization_methods" in result
    assert "likely_surface_classes" in result
    assert "likely_phase_classes" in result
    assert "likely_doe_classes" in result


def test_inspector_without_repo_is_unavailable():
    inspector = DeepLensSourceInspector("/nonexistent/path")
    assert inspector.available is False
    result = inspector.scan()
    assert result["available"] is False
