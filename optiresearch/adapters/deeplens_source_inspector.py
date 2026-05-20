"""DeepLens source code inspector.

Scans the DeepLens GitHub source checkout to discover:
- Available modules
- Key classes and their methods
- Likely PSF methods
- Likely optimization methods
- Likely surface/phase/DOE classes
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any


class DeepLensSourceInspector:
    """Inspect DeepLens source structure without executing heavy code."""

    def __init__(self, repo_path: str | None = None) -> None:
        self._repo_path = repo_path or os.getenv("DEEPLENS_REPO_PATH", "")
        self._deeplens_dir: Path | None = None
        if self._repo_path:
            candidate = Path(self._repo_path) / "deeplens"
            if candidate.is_dir():
                self._deeplens_dir = candidate

    @property
    def available(self) -> bool:
        return self._deeplens_dir is not None and self._deeplens_dir.is_dir()

    def scan(self) -> dict[str, Any]:
        """Scan the source tree and return structured inspection results."""
        if not self.available:
            return {
                "available": False,
                "error": "DEEPLENS_REPO_PATH not set or deeplens/ not found.",
                "repo_path": self._repo_path,
            }

        modules = self._list_modules()
        classes_map = self._find_classes()
        functions_map = self._find_functions()

        likely_psf = self._find_likely_psf_methods(classes_map)
        likely_optim = self._find_likely_optimization_methods(classes_map)
        likely_surface = self._find_likely_surface_classes(classes_map, modules)
        likely_phase = self._find_likely_phase_classes(classes_map, modules)
        likely_doe = self._find_likely_doe_classes(classes_map, modules)

        return {
            "available": True,
            "repo_path": self._repo_path,
            "deeplens_dir": str(self._deeplens_dir),
            "modules": modules,
            "classes": classes_map,
            "functions": functions_map,
            "likely_psf_methods": likely_psf,
            "likely_optimization_methods": likely_optim,
            "likely_surface_classes": likely_surface,
            "likely_phase_classes": likely_phase,
            "likely_doe_classes": likely_doe,
            "missing_modules": [m for m in [
                "geolens", "hybridlens", "diffraclens", "paraxiallens",
                "psfnetlens", "geometric_surface", "diffractive_surface",
                "phase_surface", "imgsim", "geolens_pkg",
            ] if m not in modules],
        }

    def _list_modules(self) -> dict[str, dict[str, Any]]:
        modules: dict[str, dict[str, Any]] = {}
        if not self._deeplens_dir:
            return modules
        for item in sorted(self._deeplens_dir.iterdir()):
            if item.name.startswith("_"):
                continue
            if item.is_dir() and (item / "__init__.py").exists():
                modules[item.name] = {
                    "type": "package",
                    "path": str(item),
                }
            elif item.suffix == ".py" and item.name != "__init__.py":
                modules[item.stem] = {
                    "type": "module",
                    "path": str(item),
                    "size_bytes": item.stat().st_size,
                }
        return modules

    def _find_classes(self) -> dict[str, list[dict[str, Any]]]:
        classes: dict[str, list[dict[str, Any]]] = {}
        if not self._deeplens_dir:
            return classes
        for py_file in self._deeplens_dir.rglob("*.py"):
            if any(skip in str(py_file) for skip in ["__pycache__", ".git"]):
                continue
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
                mod_name = py_file.stem
                found = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        methods = [
                            n.name for n in ast.walk(node)
                            if isinstance(n, ast.FunctionDef)
                        ]
                        found.append({
                            "name": node.name,
                            "methods": methods,
                            "line": node.lineno,
                        })
                if found:
                    classes[mod_name] = found
            except SyntaxError:
                pass
        return classes

    def _find_functions(self) -> dict[str, list[str]]:
        functions: dict[str, list[str]] = {}
        if not self._deeplens_dir:
            return functions
        for py_file in self._deeplens_dir.rglob("*.py"):
            if any(skip in str(py_file) for skip in ["__pycache__", ".git"]):
                continue
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
                found = [
                    node.name for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
                    and not any(
                        isinstance(parent, ast.ClassDef)
                        for parent in ast.walk(tree)
                        if isinstance(parent, ast.ClassDef)
                        for child in ast.walk(parent)
                        if child is node
                    )
                ]
                # Simpler: just get top-level functions
                top_level = [
                    node.name for node in ast.iter_child_nodes(tree)
                    if isinstance(node, ast.FunctionDef)
                ]
                if top_level:
                    functions[py_file.stem] = top_level
            except SyntaxError:
                pass
        return functions

    def _find_likely_psf_methods(self, classes_map: dict) -> list[str]:
        psf_keywords = ["psf", "point_spread", "spot", "ray_trace"]
        results = []
        for mod, cls_list in classes_map.items():
            for cls in cls_list:
                for method in cls.get("methods", []):
                    if any(kw in method.lower() for kw in psf_keywords):
                        results.append(f"{mod}.{cls['name']}.{method}")
        return results

    def _find_likely_optimization_methods(self, classes_map: dict) -> list[str]:
        opt_keywords = ["optim", "minimize", "loss", "gradient", "backward", "train"]
        results = []
        for mod, cls_list in classes_map.items():
            for cls in cls_list:
                for method in cls.get("methods", []):
                    if any(kw in method.lower() for kw in opt_keywords):
                        results.append(f"{mod}.{cls['name']}.{method}")
        return results

    def _find_likely_surface_classes(self, classes_map: dict, modules: dict) -> list[str]:
        surface_keywords = ["surface", "sag", "height", "curvature", "profile"]
        results = []
        for mod, cls_list in classes_map.items():
            if any(kw in mod for kw in ["geometric_surface", "surface", "geolens"]):
                for cls in cls_list:
                    results.append(f"{mod}.{cls['name']}")
        return results

    def _find_likely_phase_classes(self, classes_map: dict, modules: dict) -> list[str]:
        phase_keywords = ["phase", "wavefront", "zernike"]
        results = []
        for mod, cls_list in classes_map.items():
            for cls in cls_list:
                if any(kw in cls["name"].lower() for kw in phase_keywords):
                    results.append(f"{mod}.{cls['name']}")
        return results

    def _find_likely_doe_classes(self, classes_map: dict, modules: dict) -> list[str]:
        doe_keywords = ["binary", "grating", "diffractive", "diffraction", "fresnel"]
        results = []
        for mod, cls_list in classes_map.items():
            for cls in cls_list:
                if any(kw in cls["name"].lower() for kw in doe_keywords):
                    results.append(f"{mod}.{cls['name']}")
        return results


def export_source_inspection(output_dir: Path | None = None) -> dict[str, Any]:
    """CLI entry point: scan and export source inspection."""
    inspector = DeepLensSourceInspector()
    result = inspector.scan()

    root = output_dir or Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)

    (root / "deeplens_source_inspection.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    lines = ["# DeepLens Source Inspection", ""]
    if result.get("available"):
        lines.extend([
            f"**Repo path:** `{result['repo_path']}`",
            f"**DeepLens dir:** `{result['deeplens_dir']}`",
            "",
            "## Modules",
            "| Module | Type |",
            "|---|---|",
        ])
        for name, info in result.get("modules", {}).items():
            lines.append(f"| {name} | {info['type']} |")

        lines.extend(["", "## Key Classes", ""])
        for mod, cls_list in result.get("classes", {}).items():
            for cls in cls_list[:3]:
                lines.append(f"- `{mod}.{cls['name']}` (line {cls['line']})")

        lines.extend(["", "## Likely PSF Methods", ""])
        for m in result.get("likely_psf_methods", []):
            lines.append(f"- `{m}`")

        lines.extend(["", "## Likely Optimization Methods", ""])
        for m in result.get("likely_optimization_methods", []):
            lines.append(f"- `{m}`")

        lines.extend(["", "## Likely DOE / Diffractive Classes", ""])
        for c in result.get("likely_doe_classes", []):
            lines.append(f"- `{c}`")

    else:
        lines.append(f"**Error:** {result.get('error', 'Unknown')}")

    (root / "deeplens_source_inspection.md").write_text("\n".join(lines), encoding="utf-8")

    return result
