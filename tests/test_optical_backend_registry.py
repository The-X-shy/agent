"""Tests for optical backend registry."""

import json
import tempfile
from pathlib import Path

from optiresearch.backends.base import OpticalBackend
from optiresearch.backends.registry import (
    export_backend_registry_json,
    export_backend_registry_markdown,
    get_backend,
    get_backend_by_claim_ceiling,
    get_backend_registry,
    list_backends,
    register_backend,
)


def test_registry_contains_8_backends():
    backends = list_backends()
    assert len(backends) == 8


def test_all_backends_have_claim_ceiling():
    for b in list_backends():
        assert b.claim_ceiling, f"{b.backend_id} missing claim_ceiling"
        assert b.claim_ceiling != ""


def test_get_backend_returns_correct_backend():
    b = get_backend("deeplens_geolens_geometric")
    assert b is not None
    assert b.backend_id == "deeplens_geolens_geometric"
    assert b.claim_ceiling == "native_lens_simulation"
    assert b.supports_native_optimization is True


def test_get_backend_unknown_returns_none():
    assert get_backend("nonexistent") is None


def test_geolens_geometric_claim_ceiling():
    b = get_backend("deeplens_geolens_geometric")
    assert b.claim_ceiling == "native_lens_simulation"
    assert b.supports_full_waveoptics is False


def test_phase_to_fft_proxy_claim_ceiling():
    b = get_backend("phase_to_fft_proxy")
    assert b.claim_ceiling == "native_full_reconstruction_proxy"


def test_coherent_asm_not_differentiable():
    b = get_backend("deeplens_coherent_asm")
    assert b.differentiability_level == "none"
    assert b.supports_native_optimization is False


def test_mock_deeplens_claim_ceiling():
    b = get_backend("mock_deeplens")
    assert b.claim_ceiling == "mock_simulation"


def test_register_backend_override():
    original = get_backend("mock_deeplens")
    updated = OpticalBackend(
        backend_id="mock_deeplens",
        label="Updated Mock",
        backend_type="mock",
        differentiability_level="none",
        claim_ceiling="mock_simulation",
        known_failure_modes=["updated failure mode"],
    )
    register_backend(updated)
    fetched = get_backend("mock_deeplens")
    assert fetched is not None
    assert "updated failure mode" in fetched.known_failure_modes
    # Restore
    if original:
        register_backend(original)


def test_get_backend_by_claim_ceiling():
    backends = get_backend_by_claim_ceiling("native_lens_simulation")
    ids = {b.backend_id for b in backends}
    assert "deeplens_geolens_geometric" in ids
    assert "deeplens_coherent_asm" in ids


def test_export_markdown(tmp_path):
    path = tmp_path / "registry.md"
    result = export_backend_registry_markdown(path)
    assert result.exists()
    content = result.read_text()
    assert "Optical Backend Registry" in content
    assert "deeplens_geolens_geometric" in content


def test_export_json(tmp_path):
    path = tmp_path / "registry.json"
    result = export_backend_registry_json(path)
    assert result.exists()
    data = json.loads(result.read_text())
    assert "deeplens_geolens_geometric" in data
    assert data["deeplens_geolens_geometric"]["claim_ceiling"] == "native_lens_simulation"


def test_get_backend_registry_returns_copy():
    reg1 = get_backend_registry()
    reg2 = get_backend_registry()
    assert reg1 is not reg2
    assert reg1 == reg2


def test_fresnel_component_failure_modes():
    b = get_backend("deeplens_fresnel_component")
    assert b is not None
    assert any("FFT proxy" in fm for fm in b.known_failure_modes)
