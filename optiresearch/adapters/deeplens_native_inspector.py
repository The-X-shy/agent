"""DeepLens native optimization capability inspector.

Inspects DeepLens lens classes at runtime to detect whether they support
true native differentiable optimization (activate_grad, get_optimizer,
trainable parameters, autograd-aware PSF methods).

Unlike the source inspector (AST-based), this module uses runtime
introspection — importing classes and checking their actual behavior.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
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


def _yn(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return str(value)
