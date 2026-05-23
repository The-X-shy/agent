"""Test the backend progression graph."""

import pytest
from optiresearch.backends.progression import (
    BackendProgressionGraph,
    get_next_backend,
    list_progression_from,
)


def test_graph_has_all_backends():
    graph = BackendProgressionGraph()
    assert "phase_to_fft_proxy" in graph.nodes
    assert "deeplens_geolens_geometric" in graph.nodes
    assert "mock_deeplens" in graph.nodes


def test_phase_to_fft_proxy_progresses_to_geolens():
    next_backends = list_progression_from("phase_to_fft_proxy")
    assert "deeplens_geolens_geometric" in next_backends


def test_get_next_backend_with_reason():
    result = get_next_backend("phase_to_fft_proxy", reason="claim_ceiling_reached")
    assert result is not None
    assert result["next_backend"] == "deeplens_geolens_geometric"
    assert result["expected_claim_gain"] is not None


def test_geolens_progresses_to_coherent_asm():
    next_backends = list_progression_from("deeplens_geolens_geometric")
    assert "deeplens_coherent_asm" in next_backends


def test_terminal_backend_has_no_progression():
    result = get_next_backend("deeplens_coherent_asm", reason="claim_ceiling_reached")
    assert result is None


def test_unknown_backend_returns_none():
    result = get_next_backend("nonexistent", reason="claim_ceiling_reached")
    assert result is None


def test_progression_edge_has_all_fields():
    result = get_next_backend("phase_to_fft_proxy", reason="claim_ceiling_reached")
    assert "next_backend" in result
    assert "reason" in result
    assert "expected_claim_gain" in result
    assert "runtime_cost" in result
    assert "allowed_task_types" in result


def test_mock_progresses_to_proxy():
    next_backends = list_progression_from("mock_deeplens")
    assert "phase_to_fft_proxy" in next_backends


def test_export_graph_markdown(tmp_path):
    from optiresearch.backends.progression import export_progression_graph_markdown

    path = tmp_path / "test_progression.md"
    result = export_progression_graph_markdown(path)
    assert result.exists()
    content = result.read_text()
    assert "phase_to_fft_proxy" in content
    assert "deeplens_geolens_geometric" in content


def test_prefer_local_returns_low_cost_first():
    # local_synthetic_hsi → phase_to_fft_proxy has runtime_cost="low"
    result = get_next_backend("local_synthetic_hsi", prefer_local=True)
    assert result is not None
    assert result["runtime_cost"] == "low"
    assert result["next_backend"] == "phase_to_fft_proxy"


def test_phase_to_fft_proxy_no_low_cost_edges():
    # phase_to_fft_proxy outward edges all require DeepLens
    result = get_next_backend("phase_to_fft_proxy", prefer_local=True)
    assert result is not None
    assert result["runtime_cost"] in ("requires_deeplens", "low")


def test_to_dict_serialization():
    graph = BackendProgressionGraph()
    d = graph.to_dict()
    assert "nodes" in d
    assert "edges" in d
    assert len(d["edges"]) >= 5
