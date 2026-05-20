"""Safe DeepLens API probe."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


SUBMODULES = [
    "deeplens.optics",
    "deeplens.lens",
    "deeplens.surfaces",
    "deeplens.diffractive",
    "deeplens.basics",
    "deeplens.geolens",
    "deeplens.psf",
    "deeplens.utils",
]


def probe_deeplens_api() -> dict[str, Any]:
    result = {
        "available": False,
        "deeplens_version": None,
        "import_path": None,
        "python_executable": sys.executable,
        "modules_discovered": [],
        "classes_discovered": [],
        "functions_discovered": [],
        "candidate_lens_classes": [],
        "candidate_surface_classes": [],
        "candidate_phase_or_doe_classes": [],
        "candidate_optimization_methods": [],
        "notes": [],
        "errors": [],
    }
    try:
        deeplens = importlib.import_module("deeplens")
    except Exception as exc:
        result["errors"].append({"code": "DEEPLENS_IMPORT_FAILED", "message": str(exc)})
        return result
    result["available"] = True
    result["import_path"] = getattr(deeplens, "__file__", None)
    result["deeplens_version"] = getattr(deeplens, "__version__", None) or _version()
    _inspect_module("deeplens", deeplens, result)
    for module_name in SUBMODULES:
        try:
            module = importlib.import_module(module_name)
            result["modules_discovered"].append(module_name)
            _inspect_module(module_name, module, result)
        except Exception as exc:
            result["errors"].append({"code": "SUBMODULE_IMPORT_FAILED", "module": module_name, "message": str(exc)})
    result["classes_discovered"] = sorted(set(result["classes_discovered"]))
    result["functions_discovered"] = sorted(set(result["functions_discovered"]))
    result["candidate_lens_classes"] = sorted(set(result["candidate_lens_classes"]))
    result["candidate_surface_classes"] = sorted(set(result["candidate_surface_classes"]))
    result["candidate_phase_or_doe_classes"] = sorted(set(result["candidate_phase_or_doe_classes"]))
    result["candidate_optimization_methods"] = sorted(set(result["candidate_optimization_methods"]))
    if result["candidate_phase_or_doe_classes"]:
        result["notes"].append("Phase/DOE-like classes were detected; experimental semi-native support can be evaluated.")
    return result


def export_deeplens_api_probe() -> dict[str, Path]:
    payload = probe_deeplens_api()
    root = Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "deeplens_api_probe.json"
    md_path = root / "deeplens_api_probe.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def _inspect_module(module_name: str, module: Any, result: dict[str, Any]) -> None:
    if module_name not in result["modules_discovered"]:
        result["modules_discovered"].append(module_name)
    for name, member in inspect.getmembers(module):
        full_name = f"{module_name}.{name}"
        lower = name.lower()
        if inspect.isclass(member):
            result["classes_discovered"].append(full_name)
            if "lens" in lower:
                result["candidate_lens_classes"].append(full_name)
            if "surface" in lower or "aspheric" in lower:
                result["candidate_surface_classes"].append(full_name)
            if "phase" in lower or "doe" in lower or "diffractive" in lower:
                result["candidate_phase_or_doe_classes"].append(full_name)
        elif inspect.isfunction(member):
            result["functions_discovered"].append(full_name)
            if "optim" in lower or "train" in lower or "fit" in lower:
                result["candidate_optimization_methods"].append(full_name)


def _version() -> str | None:
    try:
        return importlib_metadata.version("deeplens-core")
    except importlib_metadata.PackageNotFoundError:
        return None


def _markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# DeepLens API Probe",
            "",
            f"Available: `{payload['available']}`",
            f"Version: `{payload.get('deeplens_version')}`",
            f"Import path: `{payload.get('import_path')}`",
            "",
            "## Candidate Classes",
            "",
            f"- Lens: {len(payload['candidate_lens_classes'])}",
            f"- Surface: {len(payload['candidate_surface_classes'])}",
            f"- Phase/DOE: {len(payload['candidate_phase_or_doe_classes'])}",
            f"- Optimization methods: {len(payload['candidate_optimization_methods'])}",
            "",
        ]
    )
