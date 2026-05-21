"""Tests for Phase 19B DeepLens optimization path scanning."""

from __future__ import annotations

import json
from pathlib import Path

from optiresearch.adapters.deeplens_native_inspector import (
    DeepLensOptimizationPathScanner,
    export_optimization_path_scan,
)


def _write_fake_deeplens_repo(root: Path) -> Path:
    repo = root / "DeepLens"
    (repo / "deeplens" / "diffractive_surface").mkdir(parents=True)
    (repo / "deeplens" / "phase_surface").mkdir(parents=True)
    (repo / "deeplens" / "geolens_pkg").mkdir(parents=True)
    (repo / "test").mkdir(parents=True)

    (repo / "deeplens" / "diffractive_surface" / "fresnel.py").write_text(
        """
import torch

class Fresnel:
    def phase_func(self):
        return self.f0

    def get_optimizer_params(self, lr=0.001):
        self.f0.requires_grad = True
        return [{"params": [self.f0], "lr": lr}]
""",
        encoding="utf-8",
    )
    (repo / "deeplens" / "phase_surface" / "binary2.py").write_text(
        """
class Binary2Phase:
    def phi(self, x, y):
        return self.order2

    def get_optimizer_params(self, lrs=[1e-4, 1e-2], optim_mat=False):
        self.d.requires_grad = True
        self.order2.requires_grad = True
        self.order4.requires_grad = True
        return [{"params": [self.d], "lr": lrs[0]}]
""",
        encoding="utf-8",
    )
    (repo / "deeplens" / "geolens.py").write_text(
        """
class GeoLens:
    def read_lens_json(self, filename):
        pass

    def get_optimizer(self):
        return torch.optim.Adam(self.get_optimizer_params())
""",
        encoding="utf-8",
    )
    (repo / "test" / "test_optim.py").write_text(
        """
loss.backward()
optimizer.step()
""",
        encoding="utf-8",
    )
    return repo


def test_path_scanner_finds_surface_and_lens_optimization_entries(tmp_path):
    repo = _write_fake_deeplens_repo(tmp_path)

    result = DeepLensOptimizationPathScanner(repo_path=repo).scan()

    assert result["available"] is True
    entries = {entry["class"]: entry for entry in result["entries"] if entry["class"]}
    assert entries["Fresnel"]["file"].endswith("deeplens/diffractive_surface/fresnel.py")
    assert "get_optimizer_params" in entries["Fresnel"]["optimization_method"]
    assert entries["Fresnel"]["trainable_parameters"] == ["f0"]
    assert entries["Fresnel"]["can_instantiate_no_file"] is True
    assert entries["Fresnel"]["likely_probe_type"] == "surface_phase"
    assert "run-deeplens-surface-optimization-probe" in entries["Fresnel"]["recommended_probe"]

    assert entries["GeoLens"]["requires_lens_file"] is True
    assert entries["GeoLens"]["likely_probe_type"] == "lens_file"

    for field in [
        "file",
        "class",
        "optimization_method",
        "trainable_parameters",
        "requires_lens_file",
        "can_instantiate_no_file",
        "likely_probe_type",
        "recommended_probe",
    ]:
        assert field in entries["Fresnel"]


def test_export_optimization_path_scan_writes_json_and_markdown(tmp_path, monkeypatch):
    repo = _write_fake_deeplens_repo(tmp_path)
    report_root = tmp_path / "reports"
    monkeypatch.setenv("OPTIRESEARCH_REPORT_ROOT", str(report_root))

    result = export_optimization_path_scan(repo_path=repo)

    json_path = report_root / "deeplens_optimization_path_scan.json"
    md_path = report_root / "deeplens_optimization_path_scan.md"
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["entry_count"] == result["summary"]["entry_count"]
    assert "| file | class | optimization_method |" in md_path.read_text(encoding="utf-8")
