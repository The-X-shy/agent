"""Tests for agent system report generation."""

from optiresearch.reports.agent_system_report import export_agent_system_report


def test_report_exports_markdown(tmp_path):
    path = export_agent_system_report(output_dir=tmp_path)
    assert path.exists()
    content = path.read_text()
    assert "Agentic Differentiable Optics Framework" in content
    assert "Backend Registry" in content
    assert "Experiment Controller v2" in content
    assert "Strategy Engine" in content
    assert "Research Memory v2" in content
    assert "Claim Gate v2" in content
    assert "Objective Library" in content
    assert "Autograd Auditor" in content
    assert "Example Workflow" in content
    assert "Current Capability Limits" in content
    assert "Next Development Roadmap" in content


def test_report_sections_are_non_empty(tmp_path):
    path = export_agent_system_report(output_dir=tmp_path)
    content = path.read_text()
    sections = content.split("\n## ")
    # At least 11 sections
    assert len(sections) >= 11
    for section in sections[1:]:
        assert len(section.strip()) > 0


def test_report_mentions_backends(tmp_path):
    path = export_agent_system_report(output_dir=tmp_path)
    content = path.read_text()
    assert "deeplens_geolens_geometric" in content
    assert "native_lens_simulation" in content


def test_report_includes_cli_examples(tmp_path):
    path = export_agent_system_report(output_dir=tmp_path)
    content = path.read_text()
    assert "python -m optiresearch.cli" in content
