"""Phase 31: Backend progression alternative fallback tests."""

from optiresearch.backends.progression import get_all_edges_from


def test_get_all_edges_from_phase_to_fft_proxy():
    edges = get_all_edges_from("phase_to_fft_proxy")
    assert len(edges) >= 2
    targets = {e["next_backend"] for e in edges}
    assert "deeplens_geolens_geometric" in targets
    assert "deeplens_fresnel_component" in targets


def test_get_all_edges_has_required_fields():
    edges = get_all_edges_from("phase_to_fft_proxy")
    for edge in edges:
        assert "next_backend" in edge
        assert "runtime_cost" in edge
        assert "fallback_backend" in edge
        assert "allowed_task_types" in edge


def test_terminal_backend_returns_empty():
    edges = get_all_edges_from("unknown_backend")
    assert edges == []


def test_geolens_has_progression():
    edges = get_all_edges_from("deeplens_geolens_geometric")
    assert len(edges) >= 1
    targets = {e["next_backend"] for e in edges}
    assert "deeplens_coherent_asm" in targets


def test_all_edges_have_string_ids():
    for backend in ("phase_to_fft_proxy", "deeplens_geolens_geometric", "mock_deeplens"):
        edges = get_all_edges_from(backend)
        for edge in edges:
            assert isinstance(edge["next_backend"], str)
            assert isinstance(edge["reason"], str)
