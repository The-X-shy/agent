"""Cross-platform DeepLens lens file resolver.

Translates logical lens identifiers (``auto:cooke``) into real filesystem paths
across macOS, WSL, external repos, and installed packages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LensFileResolutionResult:
    requested_lens_file: str
    resolved_path: str | None = None
    exists: bool = False
    source: str = ""
    checked_paths: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    error_code: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_lens_file": self.requested_lens_file,
            "resolved_path": self.resolved_path,
            "exists": self.exists,
            "source": self.source,
            "checked_paths": self.checked_paths,
            "alternatives": self.alternatives,
            "error_code": self.error_code,
            "warnings": self.warnings,
        }


_SAFE_ROOTS: list[Path] = []


def _safe_roots() -> list[Path]:
    global _SAFE_ROOTS
    if _SAFE_ROOTS:
        return _SAFE_ROOTS
    roots: list[Path] = []
    project_root = Path(__file__).resolve().parent.parent.parent
    roots.append(project_root)

    wsl_agent = Path("/mnt/d/agent")
    if wsl_agent.exists():
        roots.append(wsl_agent)

    for env_var in ("DEEPLENS_REPO_PATH", "OPTIRESEARCH_COOKE_LENS_FILE"):
        val = os.getenv(env_var)
        if val:
            p = Path(val)
            roots.append(p if p.is_dir() else p.parent)

    home = Path.home()
    desktop_external = home / "Desktop" / "external"
    if desktop_external.exists():
        roots.append(desktop_external)

    _SAFE_ROOTS = roots
    return _SAFE_ROOTS


def resolve_lens_file(
    lens_file: str,
    backend_id: str | None = None,
) -> LensFileResolutionResult:
    result = LensFileResolutionResult(requested_lens_file=lens_file)

    if not lens_file or not lens_file.strip():
        result.error_code = "LENS_FILE_EMPTY"
        result.warnings.append("empty lens_file argument")
        return result

    # Absolute path
    abs_path = Path(lens_file)
    if abs_path.is_absolute():
        result.checked_paths.append(str(abs_path))
        if abs_path.exists():
            result.resolved_path = str(abs_path)
            result.exists = True
            result.source = "absolute_path"
            return result
        result.error_code = "LENS_FILE_NOT_FOUND"
        result.warnings.append(f"absolute path does not exist: {abs_path}")
        return result

    # Parse auto: prefix
    search_name = lens_file
    if lens_file.startswith("auto:"):
        search_name = lens_file[5:]
        if not search_name.endswith(".json"):
            search_name = f"{search_name}.json"

    # Relative path resolution
    if "/" in search_name or search_name.startswith("."):
        candidates = [
            Path.cwd() / search_name,
            Path(__file__).resolve().parent.parent.parent / search_name,
        ]
        for c in candidates:
            result.checked_paths.append(str(c))
            if c.exists():
                result.resolved_path = str(c)
                result.exists = True
                result.source = "relative_path"
                return result
        result.error_code = "LENS_FILE_NOT_FOUND"
        return result

    # Named lens resolution (e.g., cooke.json)
    return _resolve_named_lens(search_name, result)


def _resolve_named_lens(
    name: str,
    result: LensFileResolutionResult,
) -> LensFileResolutionResult:
    # Priority 1: OPTIRESEARCH_COOKE_LENS_FILE
    env_path = os.getenv("OPTIRESEARCH_COOKE_LENS_FILE")
    if env_path:
        p = Path(env_path)
        result.checked_paths.append(str(p))
        if p.exists():
            result.resolved_path = str(p)
            result.exists = True
            result.source = "env_OPTIRESEARCH_COOKE_LENS_FILE"
            return result

    # Priority 2: DEEPLENS_REPO_PATH
    repo_path = os.getenv("DEEPLENS_REPO_PATH")
    if repo_path:
        for sub in ["datasets/lenses", "samples"]:
            p = Path(repo_path) / sub / name
            result.checked_paths.append(str(p))
            if p.exists():
                result.resolved_path = str(p)
                result.exists = True
                result.source = "env_DEEPLENS_REPO_PATH"
                return result

    # Remaining priorities: known safe paths
    known_paths = [
        ("/mnt/d/agent/external/DeepLens/datasets/lenses", "wsl_project_relative"),
        ("/mnt/d/DeepLens/datasets/lenses", "wsl_standalone"),
        ("/mnt/d/external/DeepLens/datasets/lenses", "wsl_external"),
        (str(Path.home() / "Desktop" / "external" / "DeepLens" / "datasets" / "lenses"), "macos_external"),
    ]
    for base, source_label in known_paths:
        p = Path(base) / name
        result.checked_paths.append(str(p))
        if p.exists():
            result.resolved_path = str(p)
            result.exists = True
            result.source = source_label
            return result

    # Priority: installed deeplens package
    try:
        import deeplens
        pkg_dirs = getattr(deeplens, "__path__", [])
        for pkg_dir in pkg_dirs:
            for candidate_dir in [
                Path(pkg_dir).parent / "datasets" / "lenses",
                Path(pkg_dir) / "datasets" / "lenses",
            ]:
                p = candidate_dir / name
                result.checked_paths.append(str(p))
                if p.exists():
                    result.resolved_path = str(p)
                    result.exists = True
                    result.source = "installed_deeplens_package"
                    return result
    except ImportError:
        pass

    # Priority: limited safe root search
    for root in _safe_roots():
        for sub in ["datasets/lenses", "samples", "lenses", "examples"]:
            search_dir = root / sub
            if not search_dir.is_dir():
                continue
            if not _is_safe_root(str(search_dir)):
                continue
            p = search_dir / name
            result.checked_paths.append(str(p))
            if p.exists():
                result.resolved_path = str(p)
                result.exists = True
                result.source = f"safe_root_search:{root.name}"
                return result

    result.error_code = "LENS_FILE_NOT_FOUND"
    result.warnings.append(f"could not resolve {name} in any known location")
    return result


def _is_safe_root(path: str) -> bool:
    """Reject paths outside known safe roots."""
    path_obj = Path(path).resolve()
    for root in _safe_roots():
        try:
            path_obj.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    # Allow standard locations
    for allowed in ["/mnt/", str(Path.home()), "/opt/", "/usr/local/", "/tmp/"]:
        try:
            path_obj.relative_to(Path(allowed).resolve())
            return True
        except (ValueError, OSError):
            continue
    return False
