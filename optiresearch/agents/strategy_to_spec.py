"""Strategy-to-ExperimentSpec compiler.

Converts a StrategyRecommendation from the StrategyEngine into a concrete
ExperimentSpecV2 that the ExperimentControllerV2 can execute.
"""

from __future__ import annotations

from typing import Any, Optional


def compile_experiment_spec(
    recommendation: "StrategyRecommendation",
    backend_id: str,
    objective: Optional[str] = None,
) -> Optional["ExperimentSpecV2"]:
    """Convert a StrategyRecommendation into an ExperimentSpecV2.

    Maps the recommended_action to a concrete task_type, backend_id,
    and spec_payload. Returns None for actions that don't require
    experiment execution (e.g., stop_and_report, downgrade_claim).

    All imports are lazy to avoid circular dependencies.
    """
    from optiresearch.runtime.experiment_controller_v2 import ExperimentSpecV2
    from optiresearch.memory.schemas import make_deterministic_id

    action = recommendation.recommended_action
    task_type = _action_to_task_type(action, backend_id)
    if task_type is None:
        return None

    spec_payload = _build_payload(action, recommendation, backend_id)
    spec_id = make_deterministic_id("autospec", backend_id, action)

    return ExperimentSpecV2(
        spec_id=spec_id,
        task_type=task_type,
        backend_id=backend_id,
        spec_payload=spec_payload,
        metadata={"objective": objective or "", "source": "strategy_engine"},
    )


def _action_to_task_type(action: str, backend_id: str) -> Optional[str]:
    """Map strategy action to experiment task_type.

    Returns None for actions that don't map to experiments.
    """
    mapping: dict[str, Optional[str]] = {
        "retry_with_smaller_lr": "stable_lens_hsi_codesign",
        "switch_backend": "stable_lens_hsi_codesign",
        "run_ablation": "stable_lens_hsi_codesign",
        "enable_rollback": "stable_lens_hsi_codesign",
        "run_remote_validation": "stable_lens_hsi_codesign",
        "stop_and_report": None,
        "downgrade_claim": None,
    }
    return mapping.get(action)


def _build_payload(
    action: str,
    recommendation: "StrategyRecommendation",
    backend_id: str,
) -> dict[str, Any]:
    """Build experiment payload specific to the action."""
    payload: dict[str, Any] = {}

    if action in ("retry_with_smaller_lr",):
        payload["optical_lr"] = 1e-6
        payload["recon_lr"] = 1e-3
        payload["max_steps"] = 20
        payload["rollback_on_loss_increase"] = True
        payload["candidate"] = "GeoLensCooke"
        payload["reconstructor"] = "tiny_cnn"
    elif action == "enable_rollback":
        payload["rollback_on_loss_increase"] = True
        payload["max_steps"] = 10
        payload["candidate"] = "GeoLensCooke"
        payload["reconstructor"] = "tiny_cnn"
    elif action == "switch_backend":
        payload["max_steps"] = 10
        payload["rollback_on_loss_increase"] = True
        payload["candidate"] = "GeoLensCooke"
        payload["reconstructor"] = "tiny_cnn"
    elif action == "run_ablation":
        payload["max_steps"] = 10
        payload["candidate"] = "GeoLensCooke"
        payload["reconstructor"] = "tiny_cnn"
    elif action == "run_remote_validation":
        payload["max_steps"] = 10
        payload["candidate"] = "GeoLensCooke"
        payload["reconstructor"] = "tiny_cnn"
        payload["rollback_on_loss_increase"] = True

    return payload
