"""Strategy-to-ExperimentSpec compiler.

Converts a StrategyRecommendation from the StrategyEngine into a concrete
ExperimentSpecV2 that the ExperimentControllerV2 can execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class MappingError:
    """Structured error when a strategy action cannot map to an experiment."""
    action: str
    reason: str
    suggestion: str = ""


def is_mapping_error(obj: Any) -> bool:
    """Check if an object is a MappingError."""
    return isinstance(obj, MappingError)


def compile_experiment_spec(
    recommendation: "StrategyRecommendation",
    backend_id: str,
    objective: Optional[str] = None,
    prefer_executable: bool = False,
    spec_patch: Optional[dict[str, Any]] = None,
) -> Optional[Union["ExperimentSpecV2", "MappingError"]]:
    """Convert a StrategyRecommendation into an ExperimentSpecV2.

    Maps the recommended_action to a concrete task_type, backend_id,
    and spec_payload. Returns MappingError for unmappable actions when
    prefer_executable is True, or None otherwise.

    All imports are lazy to avoid circular dependencies.
    """
    from optiresearch.runtime.experiment_controller_v2 import ExperimentSpecV2
    from optiresearch.memory.schemas import make_deterministic_id

    action = recommendation.recommended_action
    task_type = _action_to_task_type(action, backend_id)
    if task_type is None:
        if prefer_executable:
            return MappingError(
                action=action,
                reason=f"Action '{action}' cannot be mapped to an executable experiment.",
                suggestion="Try retry_with_smaller_lr, enable_rollback, run_ablation, or probe_waveoptics_path.",
            )
        return None

    spec_payload = _build_payload(
        action, recommendation, backend_id,
        prefer_executable=prefer_executable,
        spec_patch=spec_patch,
    )
    spec_id = make_deterministic_id("autospec", backend_id, action)

    from optiresearch.backends.registry import get_backend_task_evidence_cap
    evidence_cap = get_backend_task_evidence_cap(backend_id, task_type)

    return ExperimentSpecV2(
        spec_id=spec_id,
        task_type=task_type,
        backend_id=backend_id,
        spec_payload=spec_payload,
        metadata={"objective": objective or "", "source": "strategy_engine"},
        expected_evidence_level=evidence_cap,
        max_allowed_claim=evidence_cap,
        task_requirement_level=evidence_cap,
    )


def _action_to_task_type(action: str, backend_id: str) -> Optional[str]:
    """Map strategy action to experiment task_type.

    Returns None for actions that don't map to experiments.
    """
    mapping: dict[str, Optional[str]] = {
        "retry_with_smaller_lr": "stable_lens_hsi_codesign",
        "switch_backend": "stable_lens_hsi_codesign",
        "switch_backend_after_claim_ceiling": _pick_task_for_backend(backend_id),
        "run_ablation": "stable_lens_hsi_codesign",
        "enable_rollback": "stable_lens_hsi_codesign",
        "run_remote_validation": "stable_lens_hsi_codesign",
        "probe_waveoptics_path": "lightweight_psf_probe",
        "stop_and_report": None,
        "downgrade_claim": None,
    }
    return mapping.get(action)


def _pick_task_for_backend(backend_id: str) -> str:
    """Pick the most appropriate task type for a backend when switching."""
    task_map: dict[str, str] = {
        "deeplens_geolens_geometric": "psf_probe",
        "deeplens_fresnel_component": "component_optimization",
        "deeplens_binary2phase_component": "component_optimization",
        "deeplens_coherent_asm": "lightweight_psf_probe",
        "deeplens_blackbox_source_psf": "psf_probe",
        "phase_to_fft_proxy": "stable_lens_hsi_codesign",
        "mock_deeplens": "lightweight_psf_probe",
        "local_synthetic_hsi": "stable_lens_hsi_codesign",
    }
    return task_map.get(backend_id, "lightweight_psf_probe")


def _build_payload(
    action: str,
    recommendation: "StrategyRecommendation",
    backend_id: str,
    prefer_executable: bool = False,
    spec_patch: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build experiment payload specific to the action."""
    max_steps = 5 if prefer_executable else 20

    payload: dict[str, Any] = {}

    if action in ("retry_with_smaller_lr",):
        payload["optical_lr"] = 1e-6
        payload["recon_lr"] = 1e-3
        payload["max_steps"] = max_steps
        payload["rollback_on_loss_increase"] = True
        payload["candidate"] = "GeoLensCooke"
        payload["reconstructor"] = "tiny_cnn"
    elif action == "enable_rollback":
        payload["rollback_on_loss_increase"] = True
        payload["max_steps"] = max_steps
        payload["candidate"] = "GeoLensCooke"
        payload["reconstructor"] = "tiny_cnn"
    elif action == "switch_backend":
        payload["max_steps"] = max_steps
        payload["rollback_on_loss_increase"] = True
        payload["candidate"] = "GeoLensCooke"
        payload["reconstructor"] = "tiny_cnn"
    elif action == "run_ablation":
        payload["max_steps"] = max_steps
        payload["candidate"] = "GeoLensCooke"
        payload["reconstructor"] = "tiny_cnn"
    elif action == "run_remote_validation":
        payload["max_steps"] = 5 if prefer_executable else 10
        payload["candidate"] = "GeoLensCooke"
        payload["reconstructor"] = "tiny_cnn"
        payload["rollback_on_loss_increase"] = True
    elif action == "probe_waveoptics_path":
        payload["max_steps"] = 3
        payload["candidate"] = "GeoLensCooke"
        payload["reconstructor"] = "tiny_cnn"
        payload["device"] = "cpu"
    elif action == "switch_backend_after_claim_ceiling":
        payload["max_steps"] = max_steps
        payload["rollback_on_loss_increase"] = True
        payload["candidate"] = "GeoLensCooke"
        payload["reconstructor"] = "tiny_cnn"
        payload["device"] = "cpu"

    if spec_patch:
        disallowed_keys = {
            "backend_id", "task_type", "execution_target",
            "claim_ceiling", "shell_command", "file_path",
        }
        safe_patch = {k: v for k, v in spec_patch.items() if k not in disallowed_keys}
        if spec_patch.get("execution_target") == "remote":
            safe_patch.pop("execution_target", None)
        payload.update(safe_patch)

    return payload
