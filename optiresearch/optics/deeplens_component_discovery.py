"""DeepLens component backend discovery for Phase 62.

Probes the importability and instantiatability of individual DeepLens surface
component classes (Fresnel, Binary2Phase, diffractive candidates) without
running any optimization loop.  This is a lightweight "smoke check" that can
run on any Python environment — the import attempts will fail gracefully when
DeepLens is not installed.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Component-to-surface-class mapping (mirrors the runtime probe mapping).
COMPONENT_TO_SURFACE: dict[str, str] = {
    "fresnel": "Fresnel",
    "binary2phase": "Binary2Phase",
    "diffractive": "Fresnel",
}

# Module paths from the existing surface optimization probe infrastructure.
COMPONENT_MODULE_PATHS: dict[str, str] = {
    "Fresnel": "deeplens.diffractive_surface.fresnel",
    "Binary2Phase": "deeplens.phase_surface.binary2",
}

# Known trainable parameter names per surface class.
COMPONENT_TRAINABLE_PARAM_NAMES: dict[str, list[str]] = {
    "Fresnel": ["f0"],
    "Binary2Phase": ["d", "order2", "order4", "order6", "order8", "order10", "order12"],
}

# Additional candidate classes to check for diffractive availability.
DIFFRACTIVE_CANDIDATE_CLASSES: list[str] = [
    "Fresnel",
    "Binary2Phase",
    "Zernike",
    "Grating",
    "Pixel2D",
    "ThinLens",
    "FresnelPhase",
    "CubicPhase",
    "ZernikePhase",
    "PolyPhase",
    "GratingPhase",
    "NURBSPhase",
    "QPhase",
    "VortexPhase",
]


@dataclass
class ComponentDiscoveryResult:
    component: str
    surface_class: str
    importable: bool = False
    import_path: str = ""
    import_error: str | None = None
    instantiatable: bool = False
    instantiation_error: str | None = None
    has_phase_func: bool = False
    has_phi: bool = False
    has_get_optimizer: bool = False
    has_get_optimizer_params: bool = False
    trainable_param_names: list[str] = field(default_factory=list)
    differentiability_hints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.importable and self.instantiatable


@dataclass
class DiscoveryManifest:
    deeplens_available: bool = False
    deeplens_version: str = ""
    timestamp: str = ""
    component_candidates: list[str] = field(default_factory=list)
    available_components: list[str] = field(default_factory=list)
    unavailable_components: list[str] = field(default_factory=list)
    results: list[ComponentDiscoveryResult] = field(default_factory=list)
    diffractive_candidates_found: list[str] = field(default_factory=list)
    differentiable_candidate_found: bool = False
    import_paths_checked: list[str] = field(default_factory=list)
    constructor_signatures: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def discover_deeplens_components(
    components: list[str] | None = None,
    device: str = "cpu",
) -> DiscoveryManifest:
    """Discover available DeepLens component backends.

    Parameters
    ----------
    components:
        Component names to probe (e.g. ``["fresnel", "binary2phase"]``).
        Defaults to all three: fresnel, binary2phase, diffractive.
    device:
        Device string passed to component constructors.

    Returns
    -------
    DiscoveryManifest
        Structured discovery results for every probed component.
    """
    if components is None:
        components = ["fresnel", "binary2phase", "diffractive"]

    # Ensure DEEPLENS_REPO_PATH is on sys.path if set.
    _prepare_import_path()

    import_paths_checked: list[str] = []
    results: list[ComponentDiscoveryResult] = []
    available: list[str] = []
    unavailable: list[str] = []
    diffractive_candidates: list[str] = []
    constructor_signatures: dict[str, str] = {}
    all_warnings: list[str] = []
    all_errors: list[str] = []

    deeplens_available = _check_deeplens_available()
    deeplens_version = _get_deeplens_version()

    for component in components:
        surface_class = COMPONENT_TO_SURFACE.get(component, component)
        result = _discover_one(component, surface_class, device)
        results.append(result)
        import_paths_checked.append(result.import_path or surface_class)

        if result.warnings:
            all_warnings.extend(result.warnings)

        if result.available:
            available.append(component)
        else:
            unavailable.append(component)
            if result.import_error:
                all_errors.append(f"{component}: {result.import_error}")

    # Diffractive candidate scan.
    differentiable_found = False
    if "diffractive" in components or not components:
        for cls_name in DIFFRACTIVE_CANDIDATE_CLASSES:
            if _class_importable(cls_name):
                diffractive_candidates.append(cls_name)
                if cls_name in COMPONENT_TRAINABLE_PARAM_NAMES:
                    differentiable_found = True

    for cls_name in diffractive_candidates:
        sig = _probe_constructor_signature(cls_name)
        if sig:
            constructor_signatures[cls_name] = sig

    return DiscoveryManifest(
        deeplens_available=deeplens_available,
        deeplens_version=deeplens_version,
        timestamp=_now_iso(),
        component_candidates=components,
        available_components=available,
        unavailable_components=unavailable,
        results=results,
        diffractive_candidates_found=diffractive_candidates,
        differentiable_candidate_found=differentiable_found,
        import_paths_checked=import_paths_checked,
        constructor_signatures=constructor_signatures,
        warnings=all_warnings,
        errors=all_errors,
    )


def _discover_one(
    component: str,
    surface_class: str,
    device: str,
) -> ComponentDiscoveryResult:
    module_path = COMPONENT_MODULE_PATHS.get(surface_class, "")
    result = ComponentDiscoveryResult(
        component=component,
        surface_class=surface_class,
        import_path=module_path,
    )

    cls = _import_class(surface_class, module_path)
    if cls is None:
        result.import_error = f"Could not import {surface_class} from {module_path}"
        result.warnings.append(result.import_error)
        return result
    result.importable = True

    if hasattr(cls, "get_optimizer"):
        result.has_get_optimizer = callable(getattr(cls, "get_optimizer", None))
    if hasattr(cls, "get_optimizer_params"):
        result.has_get_optimizer_params = callable(getattr(cls, "get_optimizer_params", None))

    # Collect differentiability hints from class attributes.
    hints: list[str] = []
    for attr in ("is_differentiable", "supports_autograd", "trainable"):
        if getattr(cls, attr, False):
            hints.append(attr)
    result.differentiability_hints = hints

    # Build trainable param names list.
    known = COMPONENT_TRAINABLE_PARAM_NAMES.get(surface_class, [])
    result.trainable_param_names = list(known)

    try:
        instance = _instantiate_surface(surface_class, cls, device)
        result.instantiatable = True
        result.has_phase_func = callable(getattr(instance, "phase_func", None))
        result.has_phi = callable(getattr(instance, "phi", None))
    except Exception as exc:
        result.instantiation_error = str(exc)
        result.warnings.append(f"Instantiation failed for {surface_class}: {exc}")

    return result


def _prepare_import_path() -> None:
    repo_path = os.getenv("DEEPLENS_REPO_PATH")
    if repo_path and Path(repo_path).is_dir() and repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _import_class(surface_class: str, module_path: str) -> type | None:
    if not module_path:
        return None
    try:
        module = importlib.import_module(module_path)
        return getattr(module, surface_class, None)
    except Exception:
        return None


def _class_importable(surface_class: str) -> bool:
    module_path = COMPONENT_MODULE_PATHS.get(surface_class)
    if module_path is None:
        for candidate_path in [
            f"deeplens.diffractive_surface.{surface_class.lower()}",
            f"deeplens.phase_surface.{surface_class.lower()}",
        ]:
            try:
                importlib.import_module(candidate_path)
                return True
            except Exception:
                continue
        return False
    return _import_class(surface_class, module_path) is not None


def _check_deeplens_available() -> bool:
    try:
        importlib.import_module("deeplens")
        return True
    except ImportError:
        return False


def _get_deeplens_version() -> str:
    try:
        dl = importlib.import_module("deeplens")
        return getattr(dl, "__version__", "unknown")
    except ImportError:
        return ""


def _instantiate_surface(surface_class: str, surface_cls: type, device: str) -> Any:
    if surface_class == "Fresnel":
        return surface_cls(d=0.0, f0=50.0, res=48, device=device)
    if surface_class == "Binary2Phase":
        return surface_cls(
            r=5.0, d=0.0, order2=1.0, order4=0.2, order6=0.05,
            order8=0.0, order10=0.0, order12=0.0, device=device,
        )
    if surface_class.endswith("Phase"):
        try:
            return surface_cls(r=5.0, d=0.0, device=device)
        except TypeError:
            return surface_cls(r=5.0, d=0.0)
    try:
        return surface_cls(d=0.0, res=48, device=device)
    except TypeError:
        return surface_cls(d=0.0, res=48)


def _probe_constructor_signature(surface_class: str) -> str:
    module_path = COMPONENT_MODULE_PATHS.get(surface_class)
    if module_path is None:
        return ""
    cls = _import_class(surface_class, module_path)
    if cls is None:
        return ""
    try:
        import inspect
        return str(inspect.signature(cls.__init__))
    except Exception:
        return ""


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
