"""Tests for ResearchMemoryV2."""

from optiresearch.memory.research_memory_v2 import (
    ResearchMemoryEntry,
    ResearchMemoryV2,
    SEEDED_RULES,
)


def test_seeded_rules_populated():
    mem = ResearchMemoryV2()
    snapshot = mem.compile_snapshot()
    total = sum(len(v) for v in snapshot.values())
    assert total >= 7


def test_seeded_rules_has_required_types():
    mem = ResearchMemoryV2()
    snapshot = mem.compile_snapshot()
    assert "ClaimBoundary" in snapshot
    assert "OptimizationPolicy" in snapshot
    assert "FailureMode" in snapshot
    assert "RemoteExecution" in snapshot
    assert "NegativeResult" in snapshot


def test_query_by_memory_type():
    mem = ResearchMemoryV2()
    results = mem.query(memory_type="ClaimBoundary")
    assert len(results) >= 2
    for r in results:
        assert r.memory_type == "ClaimBoundary"


def test_query_by_tag():
    mem = ResearchMemoryV2()
    results = mem.query(tags=["rollback"])
    assert len(results) >= 1
    assert any("rollback" in r.tags for r in results)


def test_query_by_content():
    mem = ResearchMemoryV2()
    results = mem.query(content_contains="requires_grad")
    assert len(results) >= 1
    assert any("requires_grad" in r.content.lower() for r in results)


def test_query_combined_filters():
    mem = ResearchMemoryV2()
    results = mem.query(memory_type="OptimizationPolicy", tags=["stability"])
    assert len(results) >= 1
    for r in results:
        assert r.memory_type == "OptimizationPolicy"


def test_query_min_confidence():
    mem = ResearchMemoryV2()
    results = mem.query(min_confidence=0.95)
    assert len(results) >= 1
    for r in results:
        assert r.confidence >= 0.95


def test_compile_snapshot():
    mem = ResearchMemoryV2()
    snapshot = mem.compile_snapshot()
    assert isinstance(snapshot, dict)
    for entries in snapshot.values():
        assert isinstance(entries, list)
        for e in entries:
            assert isinstance(e, ResearchMemoryEntry)


def test_add_entry():
    mem = ResearchMemoryV2()
    new_entry = ResearchMemoryEntry(
        memory_id="test_entry",
        memory_type="ExperimentOutcome",
        content="Test experiment completed successfully.",
        tags=["test", "experiment"],
        confidence=0.5,
    )
    mem.add_entry(new_entry)
    results = mem.query(tags=["test"])
    assert len(results) == 1
    assert results[0].content == "Test experiment completed successfully."


def test_export_markdown(tmp_path):
    mem = ResearchMemoryV2()
    path = tmp_path / "memory.md"
    result = mem.export_markdown(path)
    assert result.exists()
    content = result.read_text()
    assert "Research Memory v2" in content
    assert "ClaimBoundary" in content


def test_to_json():
    mem = ResearchMemoryV2()
    json_str = mem.to_json()
    assert "rule_geolens_geometric_not_coherent" in json_str
    assert "ClaimBoundary" in json_str


def test_rule_geolens_geometric():
    mem = ResearchMemoryV2()
    results = mem.query(tags=["geolens", "geometric"])
    assert len(results) >= 1
    content = " ".join(r.content for r in results)
    assert "coherent" in content.lower() or "wave-optics" in content.lower()


def test_rule_phase_to_fft_ceiling():
    mem = ResearchMemoryV2()
    results = mem.query(tags=["phase_to_fft"])
    assert len(results) >= 1
    content = " ".join(r.content for r in results)
    assert "proxy" in content.lower() or "native_full_reconstruction_proxy" in content


def test_rule_synthetic_not_real():
    mem = ResearchMemoryV2()
    results = mem.query(memory_type="NegativeResult")
    assert len(results) >= 1
    content = " ".join(r.content for r in results)
    assert "real" in content.lower() or "hsi" in content.lower()
