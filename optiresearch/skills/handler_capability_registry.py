"""Handler Capability Registry for Phase 40.

Single source of truth for what each handler can actually produce.
All other components (design generator, evaluator, claim gate, execution loop)
query this registry instead of hardcoding evidence levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HandlerCapability:
    handler_id: str
    design_type: str  # "scientific", "probe", "report", "data_request"
    task_type: str
    supported_execution_modes: list[str] = field(default_factory=lambda: ["dry_run"])
    actual_evidence_level: str = ""
    max_claim_ceiling: str = ""
    synthetic_only: bool = False
    native_backend_required: bool = False
    physical_backend: bool = False
    real_data_required: bool = False
    remote_required: bool = False
    supports_remote: bool = False
    requires_remote_validation: bool = False
    remote_evidence_ceiling: str = ""
    local_evidence_ceiling: str = ""
    metrics_supported: list[str] = field(default_factory=list)
    artifacts_supported: list[str] = field(default_factory=list)
    known_limitations: list[str] = field(default_factory=list)
    compatible_design_ids: list[str] = field(default_factory=list)
    compatible_backend_ids: list[str] = field(default_factory=list)
    remote_worker_requirements: list[str] = field(default_factory=list)
    default_timeout_sec: int = 600
    risk_level: str = "low"
    enabled: bool = True


class HandlerCapabilityRegistry:
    """Central registry of handler capabilities.

    Loads from YAML config by default, with hardcoded fallback.
    Query this to determine what evidence level a handler actually produces,
    whether it's locally executable, and what claims it supports.
    """

    DEFAULT_CONFIG_PATH = str(
        Path(__file__).resolve().parent.parent / "config" / "handler_capabilities.yaml"
    )

    def __init__(self, config_path: str | None = None):
        self._capabilities: dict[str, HandlerCapability] = {}
        self._by_design_id: dict[str, str] = {}  # design_id -> handler_id
        self._schema_version: str = ""
        self._loaded_from_config = False
        self._load(config_path)

    def _load(self, config_path: str | None = None) -> None:
        import os
        path = (
            config_path
            or os.environ.get("OPTIRESEARCH_HANDLER_CAPABILITY_CONFIG")
            or self.DEFAULT_CONFIG_PATH
        )
        try:
            import yaml
            from pathlib import Path
            from optiresearch.skills.handler_capability_schema import (
                validate_handler_capability_config,
            )
            config_path_obj = Path(path)
            if not config_path_obj.exists():
                self._register_builtins()
                return
            data = yaml.safe_load(config_path_obj.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                self._register_builtins()
                return
            errors = validate_handler_capability_config(data)
            if errors:
                # Config invalid — fall back to builtins
                self._register_builtins()
                return
            self._schema_version = data.get("capability_schema_version", "")
            self._capabilities.clear()
            self._by_design_id.clear()
            for h in data.get("handlers", []):
                cap = HandlerCapability(
                    handler_id=h.get("handler_id", ""),
                    design_type=h.get("design_type", ""),
                    task_type=h.get("task_type", ""),
                    supported_execution_modes=h.get("supported_execution_modes", []),
                    actual_evidence_level=h.get("actual_evidence_level", ""),
                    max_claim_ceiling=h.get("max_claim_ceiling", ""),
                    synthetic_only=h.get("synthetic_only", False),
                    native_backend_required=h.get("native_backend_required", False),
                    physical_backend=h.get("physical_backend", False),
                    real_data_required=h.get("real_data_required", False),
                    remote_required=h.get("remote_required", False),
                    supports_remote=h.get("supports_remote", False),
                    requires_remote_validation=h.get("requires_remote_validation", False),
                    remote_evidence_ceiling=h.get("remote_evidence_ceiling", ""),
                    local_evidence_ceiling=h.get("local_evidence_ceiling", ""),
                    metrics_supported=h.get("metrics_supported", []),
                    artifacts_supported=h.get("artifacts_supported", []),
                    known_limitations=h.get("known_limitations", []),
                    compatible_design_ids=h.get("compatible_design_ids", []),
                    compatible_backend_ids=h.get("compatible_backend_ids", []),
                    remote_worker_requirements=h.get("remote_worker_requirements", []),
                    default_timeout_sec=h.get("default_timeout_sec", 600),
                    risk_level=h.get("risk_level", "low"),
                    enabled=h.get("enabled", True),
                )
                self._capabilities[cap.handler_id] = cap
                for did in cap.compatible_design_ids:
                    self._by_design_id[did] = cap.handler_id
            self._loaded_from_config = True
        except Exception:
            self._register_builtins()

    @classmethod
    def from_config(cls, config_path: str) -> "HandlerCapabilityRegistry":
        return cls(config_path=config_path)

    def reload(self) -> None:
        self._load()

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def loaded_from_config(self) -> bool:
        return self._loaded_from_config

    def register(self, cap: HandlerCapability) -> None:
        self._capabilities[cap.handler_id] = cap
        for did in cap.compatible_design_ids:
            self._by_design_id[did] = cap.handler_id

    def get(self, handler_id: str) -> HandlerCapability | None:
        cap = self._capabilities.get(handler_id)
        if cap and not cap.enabled:
            return cap  # Return but caller should check enabled
        return cap

    def list_enabled(self) -> list[HandlerCapability]:
        return [c for c in self._capabilities.values() if c.enabled]

    def list_disabled(self) -> list[HandlerCapability]:
        return [c for c in self._capabilities.values() if not c.enabled]

    def list_all(self) -> list[HandlerCapability]:
        return list(self._capabilities.values())

    def find_by_backend_id(self, backend_id: str) -> list[HandlerCapability]:
        return [c for c in self._capabilities.values() if backend_id in c.compatible_backend_ids]

    def find_by_design_id(self, design_id: str) -> HandlerCapability | None:
        handler_id = self._by_design_id.get(design_id)
        if handler_id:
            return self._capabilities.get(handler_id)
        # Fallback: check compatible_design_ids directly
        for cap in self._capabilities.values():
            if design_id in cap.compatible_design_ids:
                return cap
        return None

    def find_by_task_type(self, task_type: str) -> list[HandlerCapability]:
        return [c for c in self._capabilities.values() if c.task_type == task_type]

    def get_actual_evidence_level(self, design_id: str) -> str | None:
        cap = self.find_by_design_id(design_id)
        if cap:
            return cap.actual_evidence_level
        return None

    def is_locally_executable(self, design_id: str) -> bool:
        cap = self.find_by_design_id(design_id)
        if cap:
            return "local" in cap.supported_execution_modes
        return False

    def inspect(self, handler_id: str) -> dict[str, Any] | None:
        cap = self._capabilities.get(handler_id)
        if cap is None:
            return None
        return {
            "handler_id": cap.handler_id,
            "design_type": cap.design_type,
            "task_type": cap.task_type,
            "supported_execution_modes": cap.supported_execution_modes,
            "actual_evidence_level": cap.actual_evidence_level,
            "max_claim_ceiling": cap.max_claim_ceiling,
            "synthetic_only": cap.synthetic_only,
            "native_backend_required": cap.native_backend_required,
            "physical_backend": cap.physical_backend,
            "real_data_required": cap.real_data_required,
            "remote_required": cap.remote_required,
            "supports_remote": cap.supports_remote,
            "requires_remote_validation": cap.requires_remote_validation,
            "remote_evidence_ceiling": cap.remote_evidence_ceiling,
            "local_evidence_ceiling": cap.local_evidence_ceiling,
            "metrics_supported": cap.metrics_supported,
            "artifacts_supported": cap.artifacts_supported,
            "known_limitations": cap.known_limitations,
            "compatible_design_ids": cap.compatible_design_ids,
            "compatible_backend_ids": cap.compatible_backend_ids,
            "remote_worker_requirements": cap.remote_worker_requirements,
            "risk_level": cap.risk_level,
            "enabled": cap.enabled,
        }

    def _register_builtins(self) -> None:
        builtins = [
            HandlerCapability(
                handler_id="objective_redesign_simpler_metric",
                design_type="scientific",
                task_type="stable_lens_hsi_codesign",
                supported_execution_modes=["dry_run", "local"],
                actual_evidence_level="lightweight_scientific_execution",
                max_claim_ceiling="lightweight_scientific_execution",
                synthetic_only=True,
                native_backend_required=False,
                physical_backend=False,
                real_data_required=False,
                remote_required=False,
                metrics_supported=[
                    "mse_before", "mse_after",
                    "psnr_before", "psnr_after",
                    "reconstruction_loss_before", "reconstruction_loss_after",
                    "best_reconstruction_loss",
                    "improvement_detected", "metrics_valid",
                    "accepted_update_count", "execution_time_sec",
                ],
                artifacts_supported=["result.json"],
                known_limitations=[
                    "Synthetic HSI data only — real HSI performance may differ",
                    "FFT Fraunhofer PSF proxy — not native DeepLens geometric ray-tracing",
                    "MSE-only objective — does not test multi-objective loss stability",
                ],
                compatible_design_ids=["objective_redesign_simpler_metric_mse_only"],
            ),
            HandlerCapability(
                handler_id="param_reduction_sweep",
                design_type="scientific",
                task_type="stable_lens_hsi_codesign",
                supported_execution_modes=["dry_run", "local"],
                actual_evidence_level="lightweight_scientific_execution",
                max_claim_ceiling="lightweight_scientific_execution",
                synthetic_only=True,
                native_backend_required=False,
                physical_backend=False,
                real_data_required=False,
                remote_required=False,
                metrics_supported=[
                    "mse_before", "mse_after",
                    "psnr_before", "psnr_after",
                    "reconstruction_loss_before", "reconstruction_loss_after",
                    "best_reconstruction_loss",
                    "configs_tested", "best_k",
                    "improvement_detected", "metrics_valid",
                    "accepted_update_count",
                ],
                artifacts_supported=["result.json"],
                known_limitations=[
                    "Synthetic HSI data only",
                    "Low-dimensional pseudo-optical parameter sweep — not native lens optimization",
                    "FFT Fraunhofer PSF proxy",
                ],
                compatible_design_ids=["param_reduction_sweep"],
            ),
            HandlerCapability(
                handler_id="backend_switch_waveoptics_coherent",
                design_type="probe",
                task_type="native_waveoptics_codesign",
                supported_execution_modes=["dry_run"],
                actual_evidence_level="structured_unsupported",
                max_claim_ceiling="structured_unsupported",
                synthetic_only=False,
                native_backend_required=True,
                physical_backend=True,
                real_data_required=False,
                remote_required=False,
                metrics_supported=[],
                artifacts_supported=[],
                known_limitations=[
                    "Coherent ASM path has requires_grad=False — PSF tensors do not expose usable gradients",
                    "Probe-only — cannot produce metric evidence",
                ],
                compatible_design_ids=["backend_switch_waveoptics_coherent"],
            ),
            HandlerCapability(
                handler_id="report_negative_result_doc",
                design_type="report",
                task_type="report_generation",
                supported_execution_modes=["dry_run", "local"],
                actual_evidence_level="report_only",
                max_claim_ceiling="report_only",
                synthetic_only=False,
                native_backend_required=False,
                physical_backend=False,
                real_data_required=False,
                remote_required=False,
                metrics_supported=["report_generated"],
                artifacts_supported=["report.md"],
                known_limitations=[
                    "Report-only evidence does not support optical improvement claims",
                ],
                compatible_design_ids=["report_negative_result_doc"],
            ),
            HandlerCapability(
                handler_id="real_data_request",
                design_type="data_request",
                task_type="native_lens_simulation_codesign",
                supported_execution_modes=[],
                actual_evidence_level="requires_user_data",
                max_claim_ceiling="requires_user_data",
                synthetic_only=False,
                native_backend_required=False,
                physical_backend=False,
                real_data_required=True,
                remote_required=False,
                metrics_supported=[],
                artifacts_supported=[],
                known_limitations=[
                    "Requires external real HSI measurement data",
                    "Cannot execute autonomously",
                ],
                compatible_design_ids=["real_data_request_req"],
            ),
        ]
        for cap in builtins:
            self.register(cap)


# Singleton instance for module-level access
_registry: HandlerCapabilityRegistry | None = None


def get_handler_capability_registry() -> HandlerCapabilityRegistry:
    global _registry
    if _registry is None:
        _registry = HandlerCapabilityRegistry()
    return _registry
