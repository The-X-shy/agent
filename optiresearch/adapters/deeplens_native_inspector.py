"""DeepLens native optimization capability inspector.

Inspects DeepLens lens classes at runtime to detect whether they support
true native differentiable optimization (activate_grad, get_optimizer,
trainable parameters, autograd-aware PSF methods).

Unlike the source inspector (AST-based), this module uses runtime
introspection — importing classes and checking their actual behavior.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from optiresearch.adapters.deeplens import DeepLensAdapter


LENS_CLASSES = [
    "ParaxialLens",
    "GeoLens",
    "DiffractiveLens",
    "HybridLens",
    "PSFNetLens",
]

MODULE_MAP = {
    "ParaxialLens": "deeplens.paraxiallens",
    "GeoLens": "deeplens.geolens",
    "DiffractiveLens": "deeplens.diffraclens",
    "HybridLens": "deeplens.hybridlens",
    "PSFNetLens": "deeplens.psfnetlens",
}

OPTIMIZATION_PATTERNS = [
    "get_optimizer_params",
    "get_optimizer",
    "requires_grad",
    "torch.optim",
    "loss.backward",
    "optimizer.step",
    "activate_grad",
    "read_lens_json",
    "init_from_dict",
]

SOURCE_SCAN_TARGETS = [
    Path("deeplens/diffractive_surface"),
    Path("deeplens/phase_surface"),
    Path("deeplens/geometric_surface"),
    Path("deeplens/geolens.py"),
    Path("deeplens/hybridlens.py"),
    Path("deeplens/diffraclens.py"),
    Path("deeplens/geolens_pkg/optim.py"),
    Path("examples"),
    Path("test"),
    Path("tests"),
]

LENS_FILE_CLASSES = {"GeoLens", "HybridLens", "DiffractiveLens"}

NO_FILE_SURFACE_CLASSES = {
    "Fresnel",
    "Binary2",
    "Zernike",
    "Grating",
    "Pixel2D",
    "ThinLens",
    "Binary2Phase",
    "CubicPhase",
    "ZernikePhase",
    "PolyPhase",
    "GratingPhase",
    "FresnelPhase",
    "NURBSPhase",
    "QPhase",
    "VortexPhase",
}


class DeepLensOptimizationPathScanner:
    """Source scanner for DeepLens differentiable optimization entry points."""

    def __init__(self, repo_path: str | Path | None = None) -> None:
        self.repo_path = self._resolve_repo_path(repo_path)

    @property
    def available(self) -> bool:
        return self.repo_path is not None and self.repo_path.exists()

    def scan(self) -> dict[str, Any]:
        if not self.available or self.repo_path is None:
            return {
                "available": False,
                "repo_path": None,
                "entries": [],
                "summary": {"entry_count": 0, "surface_candidates": 0, "lens_file_candidates": 0},
                "error": "DeepLens source checkout not found",
            }

        entries: list[dict[str, Any]] = []
        for path in self._iter_scan_files():
            entries.extend(self._scan_file(path))

        summary = {
            "entry_count": len(entries),
            "surface_candidates": sum(1 for item in entries if item.get("likely_probe_type") == "surface_phase"),
            "lens_file_candidates": sum(1 for item in entries if item.get("likely_probe_type") == "lens_file"),
            "files_scanned": len({item["file"] for item in entries}),
        }
        return {
            "available": True,
            "repo_path": str(self.repo_path),
            "scan_targets": [target.as_posix() for target in SOURCE_SCAN_TARGETS],
            "patterns": OPTIMIZATION_PATTERNS,
            "entries": entries,
            "summary": summary,
        }

    def _iter_scan_files(self) -> list[Path]:
        assert self.repo_path is not None
        files: list[Path] = []
        for target in SOURCE_SCAN_TARGETS:
            root = self.repo_path / target
            if root.is_file() and root.suffix == ".py":
                files.append(root)
            elif root.is_dir():
                files.extend(sorted(root.rglob("*.py")))
        return sorted(dict.fromkeys(files))

    def _scan_file(self, path: Path) -> list[dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not any(pattern in text for pattern in OPTIMIZATION_PATTERNS):
            return []

        rel = self._relative(path)
        classes = self._classes_in_file(text)
        if not classes:
            return [self._entry(rel, None, text)]
        return [self._entry(rel, cls_name, source) for cls_name, source in classes]

    def _entry(self, rel_file: str, cls_name: str | None, source: str) -> dict[str, Any]:
        methods = [pattern for pattern in OPTIMIZATION_PATTERNS if pattern in source]
        trainable = self._extract_trainable_parameters(source)
        requires_file = self._requires_lens_file(rel_file, cls_name)
        can_no_file = self._can_instantiate_no_file(rel_file, cls_name, requires_file)
        probe_type = self._probe_type(rel_file, cls_name, requires_file)
        return {
            "file": rel_file,
            "class": cls_name,
            "optimization_method": methods,
            "trainable_parameters": trainable,
            "requires_lens_file": requires_file,
            "can_instantiate_no_file": can_no_file,
            "likely_probe_type": probe_type,
            "recommended_probe": self._recommended_probe(cls_name, probe_type),
        }

    def _relative(self, path: Path) -> str:
        assert self.repo_path is not None
        try:
            return path.relative_to(self.repo_path).as_posix()
        except ValueError:
            return path.as_posix()

    @staticmethod
    def _classes_in_file(text: str) -> list[tuple[str, str]]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        lines = text.splitlines()
        result: list[tuple[str, str]] = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            end = getattr(node, "end_lineno", node.lineno)
            source = "\n".join(lines[node.lineno - 1 : end])
            if any(pattern in source for pattern in OPTIMIZATION_PATTERNS):
                result.append((node.name, source))
        return result

    @staticmethod
    def _extract_trainable_parameters(source: str) -> list[str]:
        names: list[str] = []
        for match in re.finditer(r"self\.([A-Za-z_]\w*)\.requires_grad_?\s*(?:=|\()", source):
            names.append(match.group(1))
        for match in re.finditer(r'"params"\s*:\s*\[?self\.([A-Za-z_]\w*)', source):
            names.append(match.group(1))
        return _unique(names)

    @staticmethod
    def _requires_lens_file(rel_file: str, cls_name: str | None) -> bool:
        if cls_name in LENS_FILE_CLASSES:
            return True
        return rel_file in {"deeplens/geolens.py", "deeplens/hybridlens.py", "deeplens/diffraclens.py"}

    @staticmethod
    def _can_instantiate_no_file(rel_file: str, cls_name: str | None, requires_file: bool) -> bool:
        if requires_file:
            return False
        if cls_name in NO_FILE_SURFACE_CLASSES:
            return True
        return "surface" in rel_file and cls_name is not None

    @staticmethod
    def _probe_type(rel_file: str, cls_name: str | None, requires_file: bool) -> str:
        if requires_file:
            return "lens_file"
        if "diffractive_surface" in rel_file or "phase_surface" in rel_file:
            return "surface_phase"
        if "geometric_surface" in rel_file:
            return "geometric_surface"
        if cls_name is None:
            return "example_or_test"
        return "source_pattern"

    @staticmethod
    def _recommended_probe(cls_name: str | None, probe_type: str) -> str | None:
        if cls_name is None:
            return None
        if probe_type == "surface_phase":
            objective = "match_target_phase" if cls_name == "Binary2Phase" else "minimize_phase_variance"
            return (
                "python -m optiresearch.cli run-deeplens-surface-optimization-probe "
                f"--surface {cls_name} --objective {objective} --max-steps 3"
            )
        if probe_type == "lens_file":
            return (
                "python -m optiresearch.cli run-deeplens-lensfile-optimization-probe "
                f"--lens-class {cls_name} --max-files 5 --max-steps 2"
            )
        return None

    @staticmethod
    def _resolve_repo_path(repo_path: str | Path | None) -> Path | None:
        if repo_path is not None:
            candidate = Path(repo_path)
            return candidate if candidate.exists() else None

        env_path = os.getenv("DEEPLENS_REPO_PATH")
        if env_path and Path(env_path).exists():
            return Path(env_path)

        try:
            spec = importlib.util.find_spec("deeplens")
        except Exception:
            spec = None
        if spec is not None and spec.origin:
            package_dir = Path(spec.origin).parent
            repo = package_dir.parent
            if (repo / "deeplens").is_dir():
                return repo
        return None


class DeepLensNativeOptimizationInspector:
    """Inspect DeepLens lens classes for native differentiable optimization support."""

    def __init__(self, adapter: DeepLensAdapter | None = None) -> None:
        self._adapter = adapter or DeepLensAdapter()
        self._deeplens = self._adapter._deeplens
        self._env = self._adapter.validate_environment()

    @property
    def available(self) -> bool:
        return self._env.get("available", False)

    def scan(self) -> dict[str, Any]:
        """Scan all lens classes and return structured capability report."""
        if not self.available:
            return {
                "available": False,
                "error": "DeepLens is not installed or not importable.",
                "error_detail": self._env.get("error", {}).get("detail", ""),
                "lens_classes": {name: self._unavailable_result(name) for name in LENS_CLASSES},
            }

        results: dict[str, dict[str, Any]] = {}
        for cls_name in LENS_CLASSES:
            results[cls_name] = self._inspect_lens_class(cls_name)

        return {
            "available": True,
            "deeplens_version": self._env.get("deeplens_version"),
            "import_path": self._env.get("import_path"),
            "is_source_checkout": self._env.get("is_source_checkout", False),
            "lens_classes": results,
        }

    def _inspect_lens_class(self, cls_name: str) -> dict[str, Any]:
        """Inspect a single lens class for optimization capabilities."""
        module_path = MODULE_MAP.get(cls_name)
        if module_path is None:
            return self._unavailable_result(cls_name, unsupported_reason="Unknown lens class name")

        # Try importing the class
        cls = None
        import_error = None
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name, None)
        except ImportError as exc:
            import_error = str(exc)
        except Exception as exc:
            import_error = str(exc)

        if cls is None:
            return self._unavailable_result(
                cls_name,
                unsupported_reason=f"Class not importable: {import_error or 'not found'}",
                import_path=module_path,
                import_error=import_error,
            )

        result: dict[str, Any] = {
            "class_available": True,
            "import_path": module_path,
            "constructor_signature": self._inspect_constructor(cls),
            "methods": self._list_public_methods(cls),
            "has_activate_grad": callable(getattr(cls, "activate_grad", None)),
            "has_get_optimizer": callable(getattr(cls, "get_optimizer", None)),
            "has_get_optimizer_params": callable(getattr(cls, "get_optimizer_params", None)),
            "has_parameters_method": (
                callable(getattr(cls, "parameters", None))
                or callable(getattr(cls, "get_optimizer_params", None))
            ),
            "has_trainable_parameters": hasattr(cls, "trainable_parameters"),
            "has_psf_method": self._check_psf_methods(cls),
            "has_forward_method": callable(getattr(cls, "forward", None)),
            "has_render_method": callable(getattr(cls, "render", None)),
            "has_analysis_method": callable(getattr(cls, "analysis", None)),
            "likely_differentiable": self._assess_likely_differentiable(cls),
            "required_constructor_args": self._inspect_required_args(cls),
            "can_instantiate_minimal": False,
            "grad_activation_testable": False,
            "optimizer_testable": False,
            "unsupported_reason": None,
            "caveats": [],
        }

        # Try minimal instantiation
        lens_instance = None
        try:
            lens_instance = self._try_instantiate(cls_name, cls)
            if lens_instance is not None:
                result["can_instantiate_minimal"] = True
                # Update instance-level checks
                inst_has_params = (
                    callable(getattr(lens_instance, "parameters", None))
                    or callable(getattr(lens_instance, "get_optimizer_params", None))
                )
                if inst_has_params:
                    result["has_parameters_method"] = True
                inst_has_activate = callable(getattr(lens_instance, "activate_grad", None))
                if inst_has_activate and not result["has_activate_grad"]:
                    result["has_activate_grad"] = True
                inst_has_optim = callable(getattr(lens_instance, "get_optimizer", None))
                if inst_has_optim and not result["has_get_optimizer"]:
                    result["has_get_optimizer"] = True
                result["likely_differentiable"] = (
                    result["has_activate_grad"]
                    and result["has_get_optimizer"]
                    and result["has_parameters_method"]
                )
        except Exception as exc:
            result["caveats"].append(f"Instantiation failed: {exc}")

        # Try activate_grad
        if lens_instance is not None:
            try:
                grad_info = self._try_activate_grad(lens_instance)
                result["grad_activation_testable"] = grad_info.get("called", False)
                if not grad_info.get("called"):
                    error_msg = grad_info.get("error") or "NotImplemented or raised exception"
                    result["caveats"].append(f"activate_grad did not succeed: {error_msg}")
                    if "NotImplemented" in str(error_msg) or not grad_info.get("error"):
                        result["has_activate_grad"] = False
                        result["likely_differentiable"] = False
                        if not result["unsupported_reason"]:
                            result["unsupported_reason"] = "activate_grad not implemented (raises NotImplementedError)"
                elif grad_info.get("called"):
                    result["has_activate_grad"] = True
            except Exception as exc:
                result["caveats"].append(f"activate_grad failed: {exc}")

        # Try get_optimizer
        if lens_instance is not None:
            try:
                opt_info = self._try_get_optimizer(lens_instance)
                result["optimizer_testable"] = opt_info.get("produced_optimizer", False)
                result["optimizer_class"] = opt_info.get("optimizer_class")
                if not opt_info.get("produced_optimizer"):
                    error_msg = opt_info.get("error") or "NotImplemented or returned None"
                    result["caveats"].append(f"get_optimizer did not succeed: {error_msg}")
                    if "NotImplemented" in str(error_msg) or not opt_info.get("error"):
                        result["has_get_optimizer"] = False
                        result["likely_differentiable"] = False
                        if not result["unsupported_reason"]:
                            result["unsupported_reason"] = "get_optimizer not implemented (raises NotImplementedError)"
            except Exception as exc:
                result["caveats"].append(f"get_optimizer failed: {exc}")

        if not result["likely_differentiable"] and not result["unsupported_reason"]:
            reasons = []
            if not result["has_activate_grad"]:
                reasons.append("no activate_grad")
            if not result["has_get_optimizer"]:
                reasons.append("no get_optimizer")
            if not result["has_psf_method"]:
                reasons.append("no PSF method")
            if reasons:
                result["unsupported_reason"] = "; ".join(reasons)

        return result

    def _try_instantiate(self, cls_name: str, cls: type) -> Any:
        """Try to instantiate a lens class with minimal arguments."""
        if cls_name == "ParaxialLens":
            try:
                return cls(foclen=50.0, fnum=2.8, device="cpu")
            except Exception:
                pass
            try:
                return cls()
            except Exception:
                pass
            return None

        # GeoLens, DiffractiveLens, HybridLens, PSFNetLens need lens files
        lens_file = self._find_lens_file(cls_name)
        if lens_file is not None:
            try:
                return cls(str(lens_file), device="cpu")
            except Exception:
                pass
            try:
                return cls(str(lens_file))
            except Exception:
                pass

        # Try no-arg instantiation as last resort
        try:
            return cls()
        except Exception:
            pass

        return None

    def _find_lens_file(self, cls_name: str) -> Path | None:
        """Search for a suitable lens JSON file in DEEPLENS_REPO_PATH."""
        repo_path = os.getenv("DEEPLENS_REPO_PATH", "")
        if not repo_path:
            return None

        repo = Path(repo_path)
        search_dirs = [
            repo / "deeplens" / "samples",
            repo / "samples",
            repo / "examples",
            repo / "lenses",
            repo / "deeplens" / "lenses",
        ]

        patterns = {
            "GeoLens": ["geolens", "geo_lens", "spheric"],
            "DiffractiveLens": ["diffractive", "diffraclens", "fresnel"],
            "HybridLens": ["hybrid", "hybridlens"],
            "PSFNetLens": ["psfnet", "psf_net"],
        }

        keywords = patterns.get(cls_name, [cls_name.lower()])

        for search_dir in search_dirs:
            if not search_dir.is_dir():
                continue
            for ext in [".json", ".JSON"]:
                for candidate in search_dir.rglob(f"*{ext}"):
                    name_lower = candidate.name.lower()
                    if any(kw in name_lower for kw in keywords):
                        return candidate

        # Fallback: any JSON file in samples
        for search_dir in search_dirs:
            if search_dir.is_dir():
                for candidate in sorted(search_dir.rglob("*.json")):
                    return candidate

        return None

    def _try_activate_grad(self, lens_instance: Any) -> dict[str, Any]:
        """Try calling activate_grad on a lens instance."""
        if not hasattr(lens_instance, "activate_grad"):
            return {"called": False, "error": "Method not found"}
        try:
            result = lens_instance.activate_grad(True)
            return {"called": True, "result": str(result) if result is not None else None}
        except Exception as exc:
            return {"called": False, "error": str(exc)}

    def _try_get_optimizer(self, lens_instance: Any) -> dict[str, Any]:
        """Try calling get_optimizer on a lens instance."""
        if not hasattr(lens_instance, "get_optimizer"):
            return {"produced_optimizer": False, "error": "Method not found"}
        try:
            optimizer = lens_instance.get_optimizer()
            if optimizer is None:
                return {"produced_optimizer": False, "error": "Returned None"}
            cls_name = type(optimizer).__name__
            return {
                "produced_optimizer": True,
                "optimizer_class": cls_name,
                "is_torch_optimizer": "Optimizer" in cls_name or hasattr(optimizer, "step"),
            }
        except Exception as exc:
            return {"produced_optimizer": False, "error": str(exc)}

    @staticmethod
    def _inspect_constructor(cls: type) -> str:
        """Get constructor signature as a readable string."""
        try:
            sig = inspect.signature(cls.__init__)
            return str(sig)
        except Exception:
            return "unknown"

    @staticmethod
    def _inspect_required_args(cls: type) -> list[str]:
        """Get required constructor arguments."""
        try:
            sig = inspect.signature(cls.__init__)
            required = []
            for name, param in sig.parameters.items():
                if name == "self":
                    continue
                if param.default is inspect.Parameter.empty:
                    required.append(name)
            return required
        except Exception:
            return []

    @staticmethod
    def _list_public_methods(cls: type) -> list[str]:
        """List public methods on the class."""
        return sorted(
            name
            for name in dir(cls)
            if not name.startswith("_") and callable(getattr(cls, name, None))
        )

    @staticmethod
    def _check_psf_methods(cls: type) -> bool:
        """Check if class has any PSF-related methods."""
        for name in dir(cls):
            if not name.startswith("_") and callable(getattr(cls, name, None)):
                if any(kw in name.lower() for kw in ["psf", "point_spread", "render", "simulate"]):
                    return True
        return False

    @staticmethod
    def _assess_likely_differentiable(cls: type) -> bool:
        """Assess if the class is likely differentiable based on available methods."""
        has_activate = callable(getattr(cls, "activate_grad", None))
        has_optim = callable(getattr(cls, "get_optimizer", None))
        # DeepLens uses dynamic __getattr__, so these may only be accessible on instances
        has_params = (
            callable(getattr(cls, "parameters", None))
            or hasattr(cls, "trainable_parameters")
            or callable(getattr(cls, "get_optimizer_params", None))
        )
        return has_activate and has_optim and has_params

    @staticmethod
    def _unavailable_result(
        cls_name: str,
        unsupported_reason: str = "DeepLens not installed",
        import_path: str | None = None,
        import_error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "class_available": False,
            "import_path": import_path,
            "constructor_signature": None,
            "methods": [],
            "has_activate_grad": False,
            "has_get_optimizer": False,
            "has_get_optimizer_params": False,
            "has_parameters_method": False,
            "has_trainable_parameters": False,
            "has_psf_method": False,
            "has_forward_method": False,
            "has_render_method": False,
            "has_analysis_method": False,
            "likely_differentiable": False,
            "required_constructor_args": [],
            "can_instantiate_minimal": False,
            "grad_activation_testable": False,
            "optimizer_testable": False,
            "unsupported_reason": unsupported_reason,
            "caveats": [import_error] if import_error else [],
        }


def export_native_optimization_inspection(
    output_dir: Path | None = None,
    adapter: Any = None,
) -> dict[str, Any]:
    """CLI entry point: inspect and export native optimization capabilities."""
    inspector = DeepLensNativeOptimizationInspector(adapter=adapter)
    result = inspector.scan()

    root = output_dir or Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)

    (root / "deeplens_native_optimization_inspection.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    lines = _build_markdown(result)
    (root / "deeplens_native_optimization_inspection.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    return result


def _build_markdown(result: dict[str, Any]) -> list[str]:
    lines = ["# DeepLens Native Optimization Inspection", ""]

    if not result.get("available"):
        lines.append(f"**Status:** DeepLens not available — {result.get('error', 'unknown')}")
        return lines

    lines.extend([
        f"**DeepLens version:** {result.get('deeplens_version', 'unknown')}",
        f"**Import path:** `{result.get('import_path', 'unknown')}`",
        f"**Source checkout:** {result.get('is_source_checkout', False)}",
        "",
        "## Lens Class Capability Table",
        "",
        "| Class | Available | activate_grad | get_optimizer | PSF Method | Likely Diffable | Instantiable | Grad Testable | Optim Testable |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    lens_classes = result.get("lens_classes", {})
    for cls_name, info in lens_classes.items():
        lines.append(
            f"| {cls_name} "
            f"| {_yn(info.get('class_available'))} "
            f"| {_yn(info.get('has_activate_grad'))} "
            f"| {_yn(info.get('has_get_optimizer'))} "
            f"| {_yn(info.get('has_psf_method'))} "
            f"| {_yn(info.get('likely_differentiable'))} "
            f"| {_yn(info.get('can_instantiate_minimal'))} "
            f"| {_yn(info.get('grad_activation_testable'))} "
            f"| {_yn(info.get('optimizer_testable'))} |"
        )

    lines.extend(["", "## Per-Class Details", ""])
    for cls_name, info in lens_classes.items():
        lines.extend([
            f"### {cls_name}",
            f"- **Constructor:** `{info.get('constructor_signature', 'unknown')}`",
            f"- **Required args:** {info.get('required_constructor_args', [])}",
            f"- **Likely differentiable:** {info.get('likely_differentiable')}",
            f"- **Unsupported reason:** {info.get('unsupported_reason', 'none')}",
            f"- **Caveats:** {info.get('caveats', [])}",
            f"- **Key methods:** {', '.join(info.get('methods', [])[:20])}",
            "",
        ])

    return lines


def export_optimization_path_scan(
    output_dir: Path | None = None,
    repo_path: str | Path | None = None,
) -> dict[str, Any]:
    """Export Phase 19B DeepLens optimization path scan artifacts."""
    scanner = DeepLensOptimizationPathScanner(repo_path=repo_path)
    result = scanner.scan()

    root = output_dir or Path(os.getenv("OPTIRESEARCH_REPORT_ROOT", "./workspace/reports"))
    root.mkdir(parents=True, exist_ok=True)
    (root / "deeplens_optimization_path_scan.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (root / "deeplens_optimization_path_scan.md").write_text(
        "\n".join(_build_optimization_path_markdown(result)), encoding="utf-8"
    )
    return result


def _build_optimization_path_markdown(result: dict[str, Any]) -> list[str]:
    lines = [
        "# DeepLens Optimization Path Scan",
        "",
        f"**Available:** {result.get('available', False)}",
        f"**Repo path:** `{result.get('repo_path')}`",
        "",
        "## Summary",
        "",
        f"- Entry count: {result.get('summary', {}).get('entry_count', 0)}",
        f"- Surface candidates: {result.get('summary', {}).get('surface_candidates', 0)}",
        f"- Lens-file candidates: {result.get('summary', {}).get('lens_file_candidates', 0)}",
        "",
        "## Optimization Paths",
        "",
        "| file | class | optimization_method | trainable_parameters | requires_lens_file | can_instantiate_no_file | likely_probe_type | recommended_probe |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in result.get("entries", []):
        lines.append(
            "| {file} | {class_name} | {methods} | {params} | {requires_file} | {can_no_file} | {probe_type} | {recommended} |".format(
                file=entry.get("file", "-"),
                class_name=entry.get("class") or "-",
                methods=", ".join(entry.get("optimization_method", [])) or "-",
                params=", ".join(entry.get("trainable_parameters", [])) or "-",
                requires_file=_yn(entry.get("requires_lens_file")),
                can_no_file=_yn(entry.get("can_instantiate_no_file")),
                probe_type=entry.get("likely_probe_type") or "-",
                recommended=entry.get("recommended_probe") or "-",
            )
        )
    return lines


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _yn(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return str(value)
