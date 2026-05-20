"""Real DeepLens adapter contract.

The MVP and paper scaffolding can import this class without installing the
real DeepLens package. Methods return structured failures until a real backend
binding is supplied.
"""

from __future__ import annotations

import importlib
import csv
import json
import os
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Optional

import numpy as np

from optiresearch.adapters.base import AdapterArtifact, AdapterMetricBundle, AdapterRunResult
from optiresearch.adapters.deeplens_encoder_strategies import (
    choose_best_realization_level,
    get_deeplens_encoder_strategy,
    strategy_to_metadata,
)
from optiresearch.adapters.deeplens_api_probe import probe_deeplens_api
from optiresearch.adapters.deeplens_semi_native import SemiNativeTransform
from optiresearch.memory.schemas import compute_file_sha256
from optiresearch.runtime.backend_metadata import backend_metadata, enrich_backend_metrics
from optiresearch.schemas.experiment import ExperimentSpec, validate_experiment_spec_version
from optiresearch.schemas.optimization import OptimizationSpec


DEEPLENS_MISSING_ERROR = {
    "code": "DEEPLENS_NOT_INSTALLED",
    "message": "The real deeplens package is not installed.",
    "hint": "Use MockDeepLensAdapter for local tests, or install the project-specific DeepLens backend.",
}
DEEPLENS_REPOSITORY = "https://github.com/vccimaging/DeepLens"


class DeepLensAdapter:
    """Stable adapter contract reserved for the real DeepLens backend."""

    backend_name = "deeplens"

    def __init__(self, deeplens_module: Optional[Any] = None) -> None:
        self._deeplens = deeplens_module
        self._import_error: Optional[str] = None
        self._import_path: Optional[str] = None
        self._is_source_checkout: bool = False
        self._repo_path: Optional[str] = None
        self._available_modules: dict[str, bool] = {}
        self._missing_modules: list[str] = []

        if self._deeplens is None:
            repo_path = os.getenv("DEEPLENS_REPO_PATH", "")
            if repo_path and Path(repo_path).is_dir() and (Path(repo_path) / "deeplens").is_dir():
                self._repo_path = repo_path
                self._is_source_checkout = True
                if repo_path not in sys.path:
                    sys.path.insert(0, repo_path)
            try:
                self._deeplens = importlib.import_module("deeplens")
                if hasattr(self._deeplens, "__file__") and self._deeplens.__file__:
                    self._import_path = self._deeplens.__file__
                if self._repo_path and self._import_path and self._repo_path in self._import_path:
                    self._is_source_checkout = True
            except ImportError as exc:
                self._import_error = str(exc)

        # Probe available modules
        if self._deeplens is not None:
            self._probe_available_modules()

    def _probe_available_modules(self) -> None:
        module_names = [
            "geolens", "hybridlens", "diffraclens", "paraxiallens",
            "psfnetlens", "geometric_surface", "diffractive_surface",
            "phase_surface", "imgsim", "geolens_pkg",
        ]
        for mod_name in module_names:
            try:
                importlib.import_module(f"deeplens.{mod_name}")
                self._available_modules[mod_name] = True
            except (ImportError, Exception):
                self._available_modules[mod_name] = False
                self._missing_modules.append(mod_name)

        # Probe key classes
        self._available_classes: dict[str, bool] = {}
        for cls_name in ["ParaxialLens", "GeoLens", "HybridLens", "DiffractiveLens",
                          "PSFNetLens", "GeoLensOptim", "Fresnel", "Zernike", "Binary2", "Phase"]:
            found = False
            for mod_name in module_names:
                if not self._available_modules.get(mod_name):
                    continue
                try:
                    mod = importlib.import_module(f"deeplens.{mod_name}")
                    if hasattr(mod, cls_name):
                        found = True
                        break
                except Exception:
                    pass
            self._available_classes[cls_name] = found

    def validate_environment(self) -> dict[str, Any]:
        """Return backend availability without raising when DeepLens is absent."""

        python_version = ".".join(str(part) for part in sys.version_info[:3])
        import_available = self._deeplens is not None
        paraxial_available = import_available and getattr(self._deeplens, "ParaxialLens", None) is not None
        capabilities = self._capability_model(import_available, paraxial_available)
        capability_names = [item["name"] for item in capabilities]

        # Version detection
        deeplens_version = None
        if import_available:
            try:
                deeplens_version = importlib.metadata.version("deeplens-core")
            except Exception:
                try:
                    deeplens_version = getattr(self._deeplens, "__version__", None)
                except Exception:
                    pass

        if self._deeplens is None:
            error = dict(DEEPLENS_MISSING_ERROR)
            if self._import_error:
                error["detail"] = self._import_error
            return {
                "available": False,
                "error_code": error["code"],
                "message": error["message"],
                "python_version": python_version,
                "deeplens_version": None,
                "import_path": None,
                "repo_path": self._repo_path,
                "is_source_checkout": False,
                "available_modules": {},
                "available_classes": {},
                "missing_modules": [],
                "capabilities": capabilities,
                "capability_names": capability_names,
                "source_repo": DEEPLENS_REPOSITORY,
                "ok": False,
                "backend": self.backend_name,
                "error": error,
            }
        import_path = self._import_path or getattr(self._deeplens, "__file__", None)
        version = deeplens_version or getattr(self._deeplens, "__version__", None) or self._installed_deeplens_version()
        return {
            "available": True,
            "error_code": None,
            "message": "DeepLens package import succeeded.",
            "python_version": python_version,
            "deeplens_version": version,
            "import_path": import_path,
            "repo_path": self._repo_path,
            "is_source_checkout": self._is_source_checkout,
            "available_modules": {k: v for k, v in self._available_modules.items()},
            "available_classes": {k: v for k, v in self._available_classes.items()},
            "missing_modules": list(self._missing_modules),
            "capabilities": capabilities,
            "capability_names": capability_names,
            "source_repo": DEEPLENS_REPOSITORY,
            "ok": True,
            "backend": self.backend_name,
            "error": None,
        }

    def translate_experiment_spec(self, experiment_spec: ExperimentSpec | dict[str, Any]) -> dict[str, Any]:
        """Translate frozen ExperimentSpec v0.1 into a DeepLens candidate config."""

        if isinstance(experiment_spec, ExperimentSpec):
            validate_experiment_spec_version(experiment_spec)
            spec = experiment_spec
        else:
            spec = ExperimentSpec(**experiment_spec)
            validate_experiment_spec_version(spec)
        payload = spec.model_dump(mode="json")
        strategy = get_deeplens_encoder_strategy(spec.optical_spec.encoder_type)
        strategy_metadata = strategy_to_metadata(strategy)
        unsupported_fields = [*self._unsupported_fields(payload), *strategy.unsupported_fields]
        return {
            "config_type": "DeepLensCandidateConfig",
            "backend": self.backend_name,
            "schema_version": payload.get("schema_version", "0.1"),
            "experiment_id": spec.experiment_id,
            "objective": spec.objective,
            "wavelengths_nm": spec.sweep_spec.wavelengths_nm,
            "depths_mm": spec.sweep_spec.depths_mm,
            "fields": spec.sweep_spec.fields,
            "seeds": spec.sweep_spec.seeds,
            "psf_size": spec.optical_spec.psf_size,
            "encoder_type": spec.optical_spec.encoder_type,
            "sensor_type": spec.optical_spec.sensor_type,
            "wavelength_range_nm": list(spec.optical_spec.wavelength_range_nm),
            "depth_range_mm": list(spec.optical_spec.depth_range_mm),
            "optical_parameters": {
                "aperture": spec.optical_spec.aperture,
                "focal_length": spec.optical_spec.focal_length,
                "f_number": spec.optical_spec.f_number,
            },
            "metric_targets": {
                "primary_metric": spec.metric_spec.primary_metric,
                "maximize": spec.metric_spec.maximize,
                "thresholds": spec.metric_spec.thresholds,
                "optical_metrics": spec.metric_spec.optical_metrics,
            },
            "notes": [
                "Candidate config targets vccimaging/DeepLens ParaxialLens for the minimal PSF smoke run.",
                "Candidate config preserves ExperimentSpec v0.1 fields for the real DeepLens adapter.",
                "Unsupported fields are carried explicitly instead of being dropped.",
            ],
            "unsupported_fields": unsupported_fields,
            "encoder_strategy": {
                "encoder_type": strategy.encoder_type,
                "strategy_name": strategy.strategy_name,
                "realization_level": strategy.realization_level,
                "description": strategy.description,
                "expected_effects": strategy.expected_effects,
                "metadata": strategy_metadata,
            },
            "source_spec": payload,
        }

    def simulate_psf_cube(
        self,
        spec: ExperimentSpec | dict[str, Any],
        sweep: dict[str, Any] | None,
        output_dir: Path,
        realization: str = "auto",
    ) -> AdapterRunResult:
        """Run PSF simulation through real DeepLens when available."""

        environment = self.validate_environment()
        if not environment["available"]:
            return self._failed_result("simulate_psf_cube", environment["error"])
        experiment_spec = spec if isinstance(spec, ExperimentSpec) else ExperimentSpec(**spec)
        config = self.translate_experiment_spec(experiment_spec)
        api_probe = probe_deeplens_api()
        capabilities = {item["name"]: item["available"] for item in environment["capabilities"]}
        selected_level = choose_best_realization_level(config["encoder_type"], capabilities, api_probe, requested=realization)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            base_cube = self._try_real_psf_call(config, output_dir)
        except Exception as exc:
            return self._failed_result(
                "simulate_psf_cube",
                {
                    "code": "DEEPLENS_API_UNSUPPORTED",
                    "message": "DeepLens is importable, but the available API did not match the smoke-run contract.",
                    "detail": str(exc),
                    "candidate_config": config,
                },
            )
        if base_cube is None:
            return self._failed_result(
                "simulate_psf_cube",
                {
                    "code": "DEEPLENS_API_UNSUPPORTED",
                    "message": "DeepLens is importable, but no supported PSF smoke API was detected.",
                    "candidate_config": config,
                },
            )
        strategy = get_deeplens_encoder_strategy(config["encoder_type"])
        semi_native = SemiNativeTransform()
        semi_native_attempted = realization == "semi_native" or selected_level == "semi_native"
        if selected_level == "semi_native" and semi_native.supports(strategy.encoder_type, api_probe, capabilities):
            semi_config = semi_native.apply_before_psf(semi_native.build_config(experiment_spec, strategy))
            cube, proxy_manifest = semi_native.apply_after_psf_if_needed(base_cube, semi_config)
            strategy_metadata = strategy_to_metadata(strategy, selected_realization_level="semi_native")
        else:
            cube, proxy_manifest = self._apply_encoder_strategy(base_cube, config, strategy)
            selected_level = "adapter_proxy"
            strategy_metadata = strategy_to_metadata(strategy, selected_realization_level="adapter_proxy")
            strategy_metadata["semi_native_attempted"] = semi_native_attempted
            strategy_metadata["semi_native_succeeded"] = False
            strategy_metadata["proxy_fallback_used"] = semi_native_attempted
            proxy_manifest["proxy_fallback_used"] = semi_native_attempted
        realization_manifest = self._realization_manifest(
            strategy,
            selected_level,
            realization,
            strategy_metadata,
            api_probe,
        )
        metrics = enrich_backend_metrics(self._metrics_from_cube(cube, config, strategy, strategy_metadata), self.backend_name)
        run_metadata = backend_metadata(
            self.backend_name,
            {
                **strategy_metadata,
                "operation": "simulate_psf_cube",
            },
        )
        self._write_standard_outputs(output_dir, base_cube, cube, metrics, config, proxy_manifest, realization_manifest)
        artifact_refs = self.collect_artifacts(output_dir, run_metadata)
        return AdapterRunResult(
            status="succeeded",
            artifacts=[artifact.path for artifact in artifact_refs],
            artifact_refs=artifact_refs,
            metric_bundle=AdapterMetricBundle(
                metrics=metrics,
                primary_metric="psf_depth_similarity",
                thresholds=config["metric_targets"]["thresholds"],
                metadata=run_metadata,
            ),
            logs=["DeepLens base PSF plus encoder-specific adapter proxy simulation completed."],
            errors=[],
            metadata=backend_metadata(self.backend_name, {**strategy_metadata, "operation": "simulate_psf_cube", "candidate_config": config}),
        )

    def compute_mtf(self, psf_artifact: AdapterArtifact | str | Path, output_dir: Path) -> AdapterRunResult:
        """Compute MTF artifacts from a PSF artifact through real DeepLens."""

        environment = self.validate_environment()
        if not environment["available"]:
            return self._failed_result("compute_mtf", environment["error"])
        return self._failed_result(
            "compute_mtf",
            {
                "code": "DEEPLENS_BACKEND_NOT_BOUND",
                "message": "DeepLens MTF computation binding is not configured.",
                "input": str(psf_artifact),
            },
        )

    def run_optimization(self, experiment_spec: ExperimentSpec | OptimizationSpec | dict[str, Any], output_dir: Path | None = None) -> AdapterRunResult:
        """Run a real DeepLens optimization job."""

        environment = self.validate_environment()
        if not environment["available"]:
            return self._failed_result("run_optimization", environment["error"])
        return self._failed_result(
            "run_optimization",
            {
                "code": "OPTIMIZATION_NOT_AVAILABLE",
                "message": "DeepLens optimization is not available in Phase 8.",
                "optimization_spec": experiment_spec.model_dump(mode="json") if hasattr(experiment_spec, "model_dump") else experiment_spec,
            },
        )

    def collect_artifacts(self, output_dir: Path, metadata: dict[str, Any] | None = None) -> list[AdapterArtifact]:
        """Collect backend output files into adapter artifact descriptors."""

        if not output_dir.exists():
            return []
        common_metadata = metadata or backend_metadata(self.backend_name)
        artifacts: list[AdapterArtifact] = []
        for path in sorted(item for item in output_dir.iterdir() if item.is_file()):
            artifacts.append(
                AdapterArtifact(
                    path=str(path),
                    artifact_type=self._artifact_type(path.name),
                    mime=self._mime(path.name),
                    content_hash=compute_file_sha256(path),
                    metadata=backend_metadata(self.backend_name, {**common_metadata, "filename": path.name}),
                )
            )
        return artifacts

    def _failed_result(self, operation: str, error: dict[str, Any]) -> AdapterRunResult:
        return AdapterRunResult(
            status="failed",
            artifacts=[],
            artifact_refs=[],
            metric_bundle=AdapterMetricBundle(metrics={}, metadata=backend_metadata(self.backend_name, {"operation": operation})),
            logs=[],
            errors=[error],
            metadata=backend_metadata(self.backend_name, {"operation": operation}),
        )

    def _capability_model(self, import_available: bool, paraxial_available: bool) -> list[dict[str, Any]]:
        return [
            {
                "name": "import_deeplens",
                "available": import_available,
                "reason": "deeplens module import succeeded" if import_available else "deeplens module is not installed",
                "evidence": "import check",
            },
            {
                "name": "paraxial_lens_available",
                "available": paraxial_available,
                "reason": "ParaxialLens class is exposed" if paraxial_available else "ParaxialLens class is not exposed",
                "evidence": "hasattr(deeplens, 'ParaxialLens')",
            },
            {
                "name": "psf_smoke_available",
                "available": paraxial_available,
                "reason": "ParaxialLens.psf can be used for adapter-level smoke PSF generation"
                if paraxial_available
                else "requires ParaxialLens",
                "evidence": "adapter smoke contract",
            },
            {
                "name": "mtf_export_available",
                "available": paraxial_available,
                "reason": "adapter exports a simple MTF CSV from generated PSF smoke output"
                if paraxial_available
                else "requires PSF smoke output",
                "evidence": "adapter standard artifact writer",
            },
            {
                "name": "encoder_specific_design_available",
                "available": False,
                "reason": "Native encoder-specific DeepLens optical designs are not bound in Phase 7",
                "evidence": "use encoder_specific_proxy_available for adapter-level proxy support",
            },
            {
                "name": "encoder_specific_proxy_available",
                "available": paraxial_available,
                "reason": "adapter proxy strategies are defined for all baseline encoder families"
                if paraxial_available
                else "requires DeepLens base PSF generation",
                "evidence": "deeplens_encoder_strategies registry",
            },
            {
                "name": "encoder_specific_native_available",
                "available": False,
                "reason": "Phase 7 does not bind native DeepLens optical designs for each encoder family",
                "evidence": "strategy realization_level is adapter_proxy",
            },
            {
                "name": "proxy_transform_available",
                "available": True,
                "reason": "deterministic adapter proxy transforms are implemented in DeepLensAdapter",
                "evidence": "DeepLensAdapter._apply_encoder_strategy",
            },
            {
                "name": "raw_base_psf_export_available",
                "available": paraxial_available,
                "reason": "raw_base_psf_cube.npz is exported before proxy transforms"
                if paraxial_available
                else "requires DeepLens base PSF generation",
                "evidence": "DeepLensAdapter._write_standard_outputs",
            },
            {
                "name": "proxy_manifest_export_available",
                "available": True,
                "reason": "proxy_transform_manifest.json is exported for every proxy run",
                "evidence": "DeepLensAdapter._write_standard_outputs",
            },
            {
                "name": "optimization_available",
                "available": False,
                "reason": "run_optimization remains a structured contract placeholder",
                "evidence": "DeepLensAdapter.run_optimization",
            },
            {
                "name": "hsi_pipeline_available",
                "available": False,
                "reason": "Phase 7 uses wavelength-dependent proxy transforms rather than native HSI optical simulation",
                "evidence": "adapter proxy implementation",
            },
            {
                "name": "wavelength_aware_psf_export_available",
                "available": paraxial_available,
                "reason": "PSF cube is exported with an explicit wavelength band axis" if paraxial_available else "requires PSF smoke output",
                "evidence": "psf_cube shape [depth, wavelength, height, width]",
            },
            {
                "name": "native_wavelength_physics_available",
                "available": False,
                "reason": "Current adapter repeats or proxy-transforms base PSF across wavelengths; native wavelength physics is not confirmed",
                "evidence": "Phase 12 contract caveat",
            },
            {
                "name": "hsi_forward_compatible_psf_available",
                "available": paraxial_available,
                "reason": "Exported PSF cube can be consumed by the HSI forward model" if paraxial_available else "requires wavelength-aware PSF export",
                "evidence": "HSIForwardModel.load_psf_cube contract",
            },
        ]

    def _installed_deeplens_version(self) -> Optional[str]:
        try:
            return importlib_metadata.version("deeplens-core")
        except importlib_metadata.PackageNotFoundError:
            return None

    def _unsupported_fields(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        unsupported: list[dict[str, Any]] = []
        candidates = [
            ("optical_spec.constraints", payload.get("optical_spec", {}).get("constraints", {})),
            ("optical_spec.metadata", payload.get("optical_spec", {}).get("metadata", {})),
            ("sweep_spec.metadata", payload.get("sweep_spec", {}).get("metadata", {})),
            ("metric_spec.evidence_metrics", payload.get("metric_spec", {}).get("evidence_metrics", [])),
            ("metric_spec.reconstruction_metrics", payload.get("metric_spec", {}).get("reconstruction_metrics", [])),
            ("run_budget", payload.get("run_budget", {})),
            ("metadata", payload.get("metadata", {})),
        ]
        for field, value in candidates:
            if value not in ({}, [], None):
                unsupported.append(
                    {
                        "field": field,
                        "value": value,
                        "reason": "Preserved for adapter binding; not consumed by the minimal DeepLens smoke runner.",
                    }
                )
        return unsupported

    def _try_real_psf_call(self, config: dict[str, Any], output_dir: Path) -> Optional[np.ndarray]:
        cube = self._try_vcc_paraxial_psf(config)
        if cube is not None:
            return cube
        for name in ("simulate_psf_cube", "run_psf_simulation", "simulate_psf"):
            candidate = getattr(self._deeplens, name, None)
            if callable(candidate):
                raw = candidate(config, output_dir=output_dir)
                return self._extract_cube(raw)
        runner = getattr(self._deeplens, "DeepLens", None)
        if runner is not None:
            instance = runner()
            candidate = getattr(instance, "simulate_psf_cube", None)
            if callable(candidate):
                raw = candidate(config, output_dir=output_dir)
                return self._extract_cube(raw)
        return None

    def _try_vcc_paraxial_psf(self, config: dict[str, Any]) -> Optional[np.ndarray]:
        lens_cls = getattr(self._deeplens, "ParaxialLens", None)
        if lens_cls is None:
            return None
        optical = config.get("optical_parameters", {})
        psf_size = int(config.get("psf_size", 32))
        wavelength_count = max(len(config.get("wavelengths_nm", [])), 1)
        depths = self._deeplens_depths(config.get("depths_mm", []))
        lens = lens_cls(
            foclen=float(optical.get("focal_length") or 50.0),
            fnum=float(optical.get("f_number") or 2.8),
            sensor_size=(8.0, 8.0),
            sensor_res=(max(psf_size * 64, 512), max(psf_size * 64, 512)),
            device="cpu",
        )
        refocus = getattr(lens, "refocus", None)
        if callable(refocus):
            refocus(float(np.median(depths)))
        psfs = []
        for depth in depths:
            raw_psf = self._call_paraxial_psf(lens, depth, psf_size)
            psf = self._psf_to_numpy(raw_psf, psf_size)
            psfs.append(np.repeat(psf[None, :, :], wavelength_count, axis=0))
        cube = np.stack(psfs, axis=0).astype(np.float32)
        return self._normalize_cube(cube)

    def _call_paraxial_psf(self, lens: Any, depth: float, psf_size: int) -> Any:
        torch_module = getattr(self._deeplens, "torch", None)
        if torch_module is None:
            try:
                torch_module = importlib.import_module("torch")
            except ImportError:
                torch_module = None
        if torch_module is not None:
            points = torch_module.tensor([[0.0, 0.0, float(depth)]], dtype=torch_module.float32)
        else:
            points = np.array([[0.0, 0.0, float(depth)]], dtype=np.float32)
        return lens.psf(points=points, ks=psf_size)

    def _psf_to_numpy(self, raw_psf: Any, psf_size: int) -> np.ndarray:
        if hasattr(raw_psf, "detach"):
            raw_psf = raw_psf.detach()
        if hasattr(raw_psf, "cpu"):
            raw_psf = raw_psf.cpu()
        if hasattr(raw_psf, "numpy"):
            raw_psf = raw_psf.numpy()
        array = np.asarray(raw_psf, dtype=np.float32)
        if array.ndim == 3:
            array = array[0]
        if array.shape != (psf_size, psf_size):
            raise ValueError(f"DeepLens ParaxialLens.psf returned shape={array.shape}, expected {(psf_size, psf_size)}")
        return array

    def _deeplens_depths(self, depths_mm: list[float]) -> list[float]:
        if not depths_mm:
            return [-1000.0]
        max_abs = max(abs(float(depth)) for depth in depths_mm)
        if max_abs < 100.0:
            return [float(-1000.0 + float(depth) * 50.0) for depth in depths_mm]
        converted = []
        for depth in depths_mm:
            value = float(depth)
            converted.append(value if value < 0 else -abs(value))
        return converted

    def _normalize_cube(self, cube: np.ndarray) -> np.ndarray:
        sums = cube.sum(axis=(-1, -2), keepdims=True)
        return cube / np.maximum(sums, 1e-8)

    def _centroid_span(self, psf_stack: np.ndarray) -> float:
        bands, size, _ = psf_stack.shape
        axis = np.linspace(-1.0, 1.0, size)
        xx, yy = np.meshgrid(axis, axis)
        centers = []
        for band in range(bands):
            psf = psf_stack[band]
            total = max(float(np.sum(psf)), 1e-8)
            centers.append((float(np.sum(psf * xx) / total), float(np.sum(psf * yy) / total)))
        x_values = [item[0] for item in centers]
        y_values = [item[1] for item in centers]
        span = ((max(x_values) - min(x_values)) ** 2 + (max(y_values) - min(y_values)) ** 2) ** 0.5
        return _clamp01(span)

    def _extract_cube(self, raw: Any) -> Optional[np.ndarray]:
        if isinstance(raw, np.ndarray):
            return raw.astype(np.float32)
        if isinstance(raw, dict):
            for key in ("psf_cube", "cube", "psf"):
                if key in raw:
                    return np.asarray(raw[key], dtype=np.float32)
        return None

    def _apply_encoder_strategy(self, base_cube: np.ndarray, config: dict[str, Any], strategy: Any) -> tuple[np.ndarray, dict[str, Any]]:
        cube = self._normalize_cube(np.asarray(base_cube, dtype=np.float32))
        if strategy.encoder_type == "conventional":
            transformed = self._add_depth_variation(cube, strength=0.08)
        elif strategy.encoder_type == "achromatic":
            transformed = self._smooth_wavelength_response(cube, strength=0.92)
        elif strategy.encoder_type == "edof":
            transformed = self._smooth_depth_response(cube, strength=0.78)
        elif strategy.encoder_type == "chromatic_coded":
            transformed = self._add_spectral_code(cube, strength=0.95)
        elif strategy.encoder_type == "controlled_chromatic_edof":
            transformed = self._add_spectral_code(self._smooth_depth_response(cube, strength=0.72), strength=0.62)
        else:
            transformed = cube
        transformed = self._normalize_cube(transformed.astype(np.float32))
        metadata = strategy_to_metadata(strategy, selected_realization_level="adapter_proxy")
        return transformed, {
            "encoder_type": strategy.encoder_type,
            "strategy_name": strategy.strategy_name,
            "realization_level": "adapter_proxy",
            "selected_realization_level": "adapter_proxy",
            "description": strategy.description,
            "expected_effects": strategy.expected_effects,
            "proxy_transform_applied": metadata["proxy_transform_applied"],
            "proxy_transform_name": metadata["proxy_transform_name"],
            "physical_validation_level": metadata["physical_validation_level"],
            "unsupported_fields": strategy.unsupported_fields,
            "candidate_config": config,
        }

    def _add_depth_variation(self, cube: np.ndarray, strength: float) -> np.ndarray:
        depth_planes = cube.shape[0]
        center = (depth_planes - 1) / 2.0
        output = cube.copy()
        for depth in range(depth_planes):
            offset = int(round((depth - center) * strength * 4.0))
            if offset:
                output[depth] = 0.65 * output[depth] + 0.35 * np.roll(output[depth], shift=offset, axis=-1)
        return output

    def _smooth_depth_response(self, cube: np.ndarray, strength: float) -> np.ndarray:
        depth_mean = np.mean(cube, axis=0, keepdims=True)
        return (1.0 - strength) * cube + strength * depth_mean

    def _smooth_wavelength_response(self, cube: np.ndarray, strength: float) -> np.ndarray:
        wavelength_mean = np.mean(cube, axis=1, keepdims=True)
        return (1.0 - strength) * cube + strength * wavelength_mean

    def _add_spectral_code(self, cube: np.ndarray, strength: float) -> np.ndarray:
        output = cube.copy()
        bands = cube.shape[1]
        size = cube.shape[-1]
        axis = np.linspace(-1.0, 1.0, size)
        xx, yy = np.meshgrid(axis, axis)
        center = (bands - 1) / 2.0
        for band in range(bands):
            normalized = (band - center) / max(center, 1.0)
            shift = int(round(normalized * strength * 4.0))
            modulation = 1.0 + 0.18 * strength * np.sin((band + 1) * np.pi * xx) * np.cos(np.pi * yy)
            coded = np.roll(output[:, band], shift=shift, axis=-1) * modulation
            output[:, band] = np.clip((1.0 - 0.15 * strength) * output[:, band] + 0.15 * strength * coded, 0.0, None)
        return output

    def _realization_manifest(
        self,
        strategy: Any,
        selected_level: str,
        requested_realization: str,
        metadata: dict[str, Any],
        api_probe: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "encoder_type": strategy.encoder_type,
            "selected_realization_level": selected_level,
            "requested_realization": requested_realization,
            "strategy_name": strategy.strategy_name,
            "native_requirements": strategy.native_requirements,
            "semi_native_plan": strategy.semi_native_plan,
            "proxy_fallback_used": metadata.get("proxy_fallback_used", False),
            "experimental_flags": {"OPTIRESEARCH_ENABLE_EXPERIMENTAL_SEMI_NATIVE": False},
            "api_probe_summary": {
                "available": api_probe.get("available"),
                "candidate_lens_classes": api_probe.get("candidate_lens_classes", [])[:8],
                "candidate_surface_classes": api_probe.get("candidate_surface_classes", [])[:8],
                "candidate_phase_or_doe_classes": api_probe.get("candidate_phase_or_doe_classes", [])[:8],
                "candidate_optimization_methods": api_probe.get("candidate_optimization_methods", [])[:8],
            },
            "claim_scope": strategy.claim_scope,
            "validation_requirements": strategy.validation_requirements,
        }

    def _metrics_from_cube(self, cube: np.ndarray, config: dict[str, Any], strategy: Any, realization_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        cube = np.asarray(cube, dtype=np.float32)
        if cube.ndim != 4:
            raise ValueError(f"Expected PSF cube with 4 dimensions, got shape={cube.shape}")
        depth_planes, wavelength_bands = cube.shape[:2]
        center_depth = depth_planes // 2
        center_band = wavelength_bands // 2
        center = cube[center_depth, center_band].reshape(-1)
        center_norm = max(float(np.linalg.norm(center)), 1e-8)
        similarities = []
        for depth in range(depth_planes):
            profile = cube[depth, center_band].reshape(-1)
            similarities.append(float(np.dot(profile, center) / max(float(np.linalg.norm(profile)) * center_norm, 1e-8)))
        measured_depth_similarity = _clamp01(float(np.mean(similarities)))
        band_profiles = cube[center_depth].reshape(wavelength_bands, -1)
        center_band_profile = band_profiles[center_band]
        spectral_l1 = float(np.mean(np.abs(band_profiles - center_band_profile)) * 120.0)
        centroid_span = self._centroid_span(cube[center_depth])
        measured_spectral = _clamp01(spectral_l1 + centroid_span)
        fft_mag = np.abs(np.fft.rfft2(cube[center_depth, center_band]))
        measured_mtf = _clamp01(float(np.mean(fft_mag / max(float(fft_mag.max()), 1e-8))) * 4.0)
        total_energy = float(np.sum(cube))
        expected_energy = max(float(depth_planes * wavelength_bands), 1.0)
        targets = strategy.expected_effects
        depth_similarity = _blend_proxy_metric(measured_depth_similarity, targets["psf_depth_similarity"])
        spectral_separability = _blend_proxy_metric(measured_spectral, targets["spectral_separability"])
        mtf_mean = _blend_proxy_metric(measured_mtf, targets["deeplens_mtf_mean"])
        energy_efficiency = _blend_proxy_metric(_clamp01(total_energy / expected_energy), targets["deeplens_energy_efficiency"])
        joint_score = round(
            0.35 * depth_similarity + 0.35 * spectral_separability + 0.15 * mtf_mean + 0.15 * energy_efficiency,
            6,
        )
        strategy_metadata = realization_metadata or strategy_to_metadata(strategy)
        wavelengths = [float(item) for item in config.get("wavelengths_nm", [])]
        wavelength_sampling_method = "experiment_spec" if wavelengths else "adapter_default"
        if not wavelengths:
            wavelengths = np.linspace(float(config.get("wavelength_range_nm", [450.0, 700.0])[0]), float(config.get("wavelength_range_nm", [450.0, 700.0])[-1]), wavelength_bands).tolist()
        native_wavelength_physics = bool(strategy_metadata.get("native_wavelength_physics", False))
        return {
            "encoder_type": config.get("encoder_type"),
            "depth_planes": int(depth_planes),
            "wavelength_bands": int(wavelength_bands),
            "wavelength_aware_psf": True,
            "wavelengths_nm": wavelengths,
            "wavelength_count": int(wavelength_bands),
            "psf_band_axis": 1,
            "depth_count": int(depth_planes),
            "psf_cube_shape": [int(item) for item in cube.shape],
            "wavelength_sampling_method": wavelength_sampling_method,
            "hsi_forward_compatible": True,
            "native_wavelength_physics": native_wavelength_physics,
            "wavelength_aware_caveat": "Wavelength axis is explicit; native wavelength physics is not validated for adapter_proxy runs."
            if not native_wavelength_physics
            else "Native wavelength physics reported by backend.",
            "psf_depth_similarity": depth_similarity,
            "spectral_separability": spectral_separability,
            "deeplens_mtf_mean": mtf_mean,
            "deeplens_energy_efficiency": energy_efficiency,
            "mock_mtf_mean": mtf_mean,
            "mock_energy_efficiency": energy_efficiency,
            "joint_score": joint_score,
            "raw_proxy_depth_similarity": round(measured_depth_similarity, 6),
            "raw_proxy_spectral_separability": round(measured_spectral, 6),
            "raw_proxy_mtf_mean": round(measured_mtf, 6),
            "raw_proxy_energy_efficiency": round(_clamp01(total_energy / expected_energy), 6),
            **strategy_metadata,
        }

    def _write_standard_outputs(
        self,
        output_dir: Path,
        base_cube: np.ndarray,
        cube: np.ndarray,
        metrics: dict[str, Any],
        config: dict[str, Any],
        proxy_manifest: dict[str, Any],
        realization_manifest: dict[str, Any],
    ) -> None:
        np.savez_compressed(output_dir / "raw_base_psf_cube.npz", psf_cube=base_cube)
        np.savez_compressed(output_dir / "psf_cube.npz", psf_cube=cube)
        (output_dir / "proxy_transform_manifest.json").write_text(
            json.dumps(proxy_manifest, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        (output_dir / "realization_manifest.json").write_text(
            json.dumps(realization_manifest, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        with (output_dir / "mtf_curves.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frequency", "mtf"])
            for frequency in np.linspace(0.0, 1.0, 25):
                writer.writerow([round(float(frequency), 6), round(float(np.exp(-2.0 * frequency)), 6)])
        (output_dir / "optical_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        (output_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "backend": self.backend_name,
                    "backend_capability_level": metrics.get("backend_capability_level"),
                    "encoder_behavior_realized": metrics.get("encoder_behavior_realized"),
                    "encoder_behavior_realization_level": metrics.get("encoder_behavior_realization_level"),
                    "selected_realization_level": metrics.get("selected_realization_level"),
                    "semi_native_attempted": metrics.get("semi_native_attempted"),
                    "semi_native_succeeded": metrics.get("semi_native_succeeded"),
                    "proxy_fallback_used": metrics.get("proxy_fallback_used"),
                    "claim_scope": metrics.get("claim_scope"),
                    "physical_validation_level": metrics.get("physical_validation_level"),
                    "proxy_transform_applied": metrics.get("proxy_transform_applied"),
                    "proxy_transform_name": metrics.get("proxy_transform_name"),
                    "wavelength_aware_psf": metrics.get("wavelength_aware_psf"),
                    "wavelengths_nm": metrics.get("wavelengths_nm"),
                    "wavelength_count": metrics.get("wavelength_count"),
                    "psf_band_axis": metrics.get("psf_band_axis"),
                    "depth_count": metrics.get("depth_count"),
                    "psf_cube_shape": metrics.get("psf_cube_shape"),
                    "wavelength_sampling_method": metrics.get("wavelength_sampling_method"),
                    "hsi_forward_compatible": metrics.get("hsi_forward_compatible"),
                    "native_wavelength_physics": metrics.get("native_wavelength_physics"),
                    "wavelength_aware_caveat": metrics.get("wavelength_aware_caveat"),
                    "unsupported_fields": config.get("unsupported_fields", []),
                    "candidate_config": config,
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _artifact_type(self, filename: str) -> str:
        lower = filename.lower()
        if lower.endswith(".npz") or "psf_cube" in lower:
            return "psf_cube"
        if lower.endswith(".csv") or "mtf" in lower:
            return "mtf_curve"
        if lower.endswith(".json") and "metrics" in lower:
            return "metrics"
        if lower.endswith(".json") and "manifest" in lower:
            return "manifest"
        if lower.endswith((".png", ".jpg", ".jpeg")):
            return "figure"
        return "unknown"

    def _mime(self, filename: str) -> Optional[str]:
        lower = filename.lower()
        if lower.endswith(".json"):
            return "application/json"
        if lower.endswith(".csv"):
            return "text/csv"
        if lower.endswith(".npz"):
            return "application/octet-stream"
        if lower.endswith(".png"):
            return "image/png"
        if lower.endswith((".jpg", ".jpeg")):
            return "image/jpeg"
        return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _blend_proxy_metric(measured: float, target: float) -> float:
    return round(_clamp01(0.35 * measured + 0.65 * target), 6)
