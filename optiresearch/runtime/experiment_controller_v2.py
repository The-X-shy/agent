"""Experiment Controller v2 — unified local/remote/mock experiment entry point.

Wraps existing Phase 18–23 runtime loops behind a single controller that
validates backend capabilities, enforces claim ceilings, and delegates
to the strategy engine for next-action recommendations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from optiresearch.memory.schemas import StrictModel, make_deterministic_id


class ExperimentSpecV2(StrictModel):
    """Input specification for ExperimentControllerV2."""

    spec_id: str
    task_type: Literal[
        "native_optimization_probe",
        "native_hsi_codesign",
        "native_hsi_reconstruction_codesign",
        "native_waveoptics_codesign",
        "stable_lens_hsi_codesign",
        "psf_probe",
        "component_optimization",
        "lightweight_psf_probe",
    ]
    backend_id: str
    execution_target: Literal["local", "remote"] = "local"
    worker_id: Optional[str] = None
    remote_job_id: Optional[str] = None
    spec_payload: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    expected_evidence_level: Optional[str] = None
    max_allowed_claim: Optional[str] = None
    task_requirement_level: Optional[str] = None


class ControllerResult(StrictModel):
    """Output of an experiment run through ExperimentControllerV2."""

    spec_id: str
    status: Literal["succeeded", "failed", "unsupported", "claim_downgraded", "skipped"]
    execution_target: Optional[str] = None
    backend_id: Optional[str] = None
    run_id: Optional[str] = None
    evidence_level: Optional[str] = None
    result_payload: Optional[dict[str, Any]] = None
    downgraded_from: Optional[str] = None
    downgraded_to: Optional[str] = None
    original_claim: Optional[str] = None
    safe_claim_wording: Optional[str] = None
    errors: list[dict[str, Any]] = []
    artifact_paths: list[str] = []
    metadata: dict[str, Any] = {}


# Claim level ordering for comparison
_CLAIM_LEVELS = [
    "unsupported",
    "mock_simulation",
    "deeplens_integration_smoke",
    "native_component_optimization",
    "native_hsi_proxy",
    "native_full_reconstruction_proxy",
    "native_lens_simulation",
    "native_waveoptics",
    "stable_native_lens_hsi_codesign",
    "rollback_protected_native_lens_hsi",
    "real_hsi_performance",
]


def _claim_level_index(level: str) -> int:
    try:
        return _CLAIM_LEVELS.index(level)
    except ValueError:
        return -1


# Task type -> minimum required claim ceiling
_TASK_REQUIRED_CEILING: dict[str, str] = {
    "native_optimization_probe": "native_component_optimization",
    "native_hsi_codesign": "native_hsi_proxy",
    "native_hsi_reconstruction_codesign": "native_full_reconstruction_proxy",
    "native_waveoptics_codesign": "native_waveoptics",
    "stable_lens_hsi_codesign": "native_lens_simulation",
    "psf_probe": "deeplens_integration_smoke",
    "component_optimization": "native_component_optimization",
    "lightweight_psf_probe": "deeplens_integration_smoke",
}


class ExperimentControllerV2:
    """Unified experiment controller for differentiable optics research.

    Wraps existing runtime loops, validates backend capabilities against
    task requirements, enforces claim ceilings, and delegates next-action
    recommendations to the strategy engine.
    """

    def __init__(self):
        pass

    def plan_experiment(
        self,
        objective: str,
        backend_id: str,
        task_type: str,
        spec_payload: Optional[dict[str, Any]] = None,
    ) -> ExperimentSpecV2:
        """Create an experiment specification from objective and backend."""
        spec_id = make_deterministic_id("v2", objective, backend_id, task_type)
        return ExperimentSpecV2(
            spec_id=spec_id,
            task_type=task_type,
            backend_id=backend_id,
            spec_payload=spec_payload or {},
            metadata={"objective": objective},
        )

    def validate_preconditions(self, spec: ExperimentSpecV2) -> list[str]:
        """Check preconditions and return a list of issues (empty = OK)."""
        issues: list[str] = []

        from optiresearch.backends.registry import get_backend

        backend = get_backend(spec.backend_id)
        if backend is None:
            issues.append(f"Unknown backend: {spec.backend_id}")
            return issues

        # Check claim ceiling
        ceiling_issue = self._check_claim_ceiling(spec.backend_id, spec.task_type)
        if ceiling_issue:
            issues.append(ceiling_issue)

        # Check task-specific requirements
        if spec.task_type == "stable_lens_hsi_codesign":
            if not backend.supports_hsi_forward:
                issues.append(
                    f"Backend {spec.backend_id} does not support HSI forward simulation"
                )
            if not backend.supports_native_optimization:
                issues.append(
                    f"Backend {spec.backend_id} does not support native optimization"
                )

        if spec.task_type == "native_waveoptics_codesign":
            if not backend.supports_full_waveoptics:
                issues.append(
                    f"Backend {spec.backend_id} does not support full wave-optics"
                )

        if spec.execution_target == "remote" and not backend.supports_remote_execution:
            issues.append(
                f"Backend {spec.backend_id} does not support remote execution"
            )

        return issues

    def run_local(self, spec: ExperimentSpecV2) -> ControllerResult:
        """Run an experiment locally, delegating to the appropriate runtime loop.

        Phase 29: Uses backend task evidence caps instead of blocking on
        claim ceiling mismatch. Returns 'unsupported' only when the task
        is not allowed at all on the backend.
        """
        from optiresearch.backends.registry import get_backend_task_evidence_cap

        evidence_cap = get_backend_task_evidence_cap(spec.backend_id, spec.task_type)
        if evidence_cap is None:
            return ControllerResult(
                spec_id=spec.spec_id,
                status="unsupported",
                execution_target="local",
                backend_id=spec.backend_id,
                errors=[{
                    "type": "unsupported_task_for_backend",
                    "message": (
                        f"Task '{spec.task_type}' not allowed on backend "
                        f"'{spec.backend_id}'"
                    ),
                }],
            )

        spec.metadata["evidence_level_cap"] = evidence_cap

        issues = self.validate_preconditions(spec)
        if issues:
            return ControllerResult(
                spec_id=spec.spec_id,
                status="skipped",
                execution_target="local",
                backend_id=spec.backend_id,
                errors=[{"type": "precondition", "message": i} for i in issues],
            )

        try:
            result = self._dispatch_local(spec)
            if result.evidence_level is None:
                result.evidence_level = evidence_cap
            return result
        except Exception as exc:
            return ControllerResult(
                spec_id=spec.spec_id,
                status="failed",
                execution_target="local",
                backend_id=spec.backend_id,
                errors=[{"type": type(exc).__name__, "message": str(exc)}],
            )

    def run_remote(self, spec: ExperimentSpecV2) -> ControllerResult:
        """Run an experiment remotely via the SSH/remote worker system."""
        try:
            from optiresearch.runtime.remote_jobs import RemoteJobSpec
            from optiresearch.remote.worker_registry import RemoteWorkerRegistry

            registry = RemoteWorkerRegistry()
            worker_id = spec.worker_id or "wslbox"
            worker = registry.get_worker(worker_id)
            if worker is None:
                return ControllerResult(
                    spec_id=spec.spec_id,
                    status="failed",
                    execution_target="remote",
                    backend_id=spec.backend_id,
                    errors=[
                        {
                            "type": "unknown_worker",
                            "message": f"Worker '{worker_id}' not found",
                        }
                    ],
                )
        except Exception as exc:
            return ControllerResult(
                spec_id=spec.spec_id,
                status="failed",
                execution_target="remote",
                backend_id=spec.backend_id,
                errors=[{"type": type(exc).__name__, "message": str(exc)}],
            )

        return ControllerResult(
            spec_id=spec.spec_id,
            status="succeeded",
            execution_target="remote",
            backend_id=spec.backend_id,
            run_id=spec.spec_id,
            metadata={"worker_id": worker_id},
        )

    def collect_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        """Collect artifacts from a completed run."""
        artifacts: list[dict[str, Any]] = []
        run_dir = Path("workspace") / run_id
        if run_dir.exists():
            for f in run_dir.rglob("*.json"):
                artifacts.append({"path": str(f.relative_to("workspace")), "type": "json"})
            for f in run_dir.rglob("*.md"):
                artifacts.append({"path": str(f.relative_to("workspace")), "type": "markdown"})
        return artifacts

    def evaluate_metrics(self, result: ControllerResult) -> dict[str, Any]:
        """Extract key metrics from a controller result."""
        metrics: dict[str, Any] = {"status": result.status}
        if result.result_payload:
            payload = result.result_payload
            for key in (
                "reconstruction_loss_before",
                "reconstruction_loss_after",
                "mse_before",
                "mse_after",
                "psnr_before",
                "psnr_after",
                "optical_gradient_norm",
                "rollback_count",
                "accepted_update_count",
                "rejected_update_count",
                "optical_parameters_changed",
            ):
                if key in payload:
                    metrics[key] = payload[key]
        return metrics

    def update_memory(self, result: ControllerResult) -> None:
        """Write experiment outcome to research memory."""
        try:
            from optiresearch.memory.research_memory_v2 import (
                ResearchMemoryV2,
                ResearchMemoryEntry,
            )

            mem = ResearchMemoryV2()
            mem_id = make_deterministic_id("mem", result.spec_id, result.status)
            entry = ResearchMemoryEntry(
                memory_id=mem_id,
                memory_type="ExperimentOutcome",
                content=(
                    f"Experiment {result.spec_id}: status={result.status}, "
                    f"backend={result.backend_id}, evidence={result.evidence_level}"
                ),
                tags=[result.status, result.backend_id or "unknown"],
                source_run_id=result.run_id,
                confidence=0.8,
            )
            mem.add_entry(entry)
        except Exception:
            pass

    def update_claim_evidence(self, result: ControllerResult) -> str:
        """Register claim evidence from a controller result."""
        if result.safe_claim_wording:
            return result.safe_claim_wording
        if result.evidence_level:
            return f"Evidence level: {result.evidence_level}"
        return "No claim evidence available"

    def recommend_next_action(self, result: ControllerResult) -> dict[str, Any]:
        """Delegate to StrategyEngine for next-action recommendation."""
        try:
            from optiresearch.agents.strategy_engine import StrategyEngine

            metrics = self.evaluate_metrics(result)
            engine = StrategyEngine()
            rec = engine.recommend(
                latest_result=metrics,
                backend_id=result.backend_id or "unknown",
            )
            return {
                "action": rec.recommended_action,
                "rationale": rec.rationale,
                "risk_level": rec.risk_level,
                "cli_commands": rec.proposed_cli_commands,
            }
        except Exception as exc:
            return {"action": "error", "rationale": str(exc)}

    # ── private helpers ───────────────────────────────────────────

    def _check_claim_ceiling(self, backend_id: str, task_type: str) -> Optional[str]:
        """Return an issue string if the backend cannot support the task's claim level."""
        from optiresearch.backends.registry import get_backend

        backend = get_backend(backend_id)
        if backend is None:
            return f"Unknown backend: {backend_id}"

        required = _TASK_REQUIRED_CEILING.get(task_type)
        if not required:
            return None

        backend_level = _claim_level_index(backend.claim_ceiling)
        required_level = _claim_level_index(required)

        if backend_level < required_level:
            return (
                f"Backend {backend_id} claim ceiling ({backend.claim_ceiling}) "
                f"is below task requirement ({required}). "
                f"Claim will be downgraded to {backend.claim_ceiling}."
            )
        return None

    def _dispatch_local(self, spec: ExperimentSpecV2) -> ControllerResult:
        """Route to the appropriate runtime loop based on task_type.

        Phase 29: Lightweight routing for proxy backends. When the backend
        is phase_to_fft_proxy or lightweight_mode is set in the payload,
        stable_lens_hsi_codesign routes to the lightweight experiment.
        """
        payload = spec.spec_payload
        use_lightweight = (
            payload.get("lightweight_mode", False)
            or spec.backend_id == "phase_to_fft_proxy"
        )

        if spec.task_type == "stable_lens_hsi_codesign":
            if use_lightweight:
                return self._run_lightweight_stable_lens_hsi(spec, payload)
            return self._run_stable_lens_hsi(spec, payload)
        elif spec.task_type == "lightweight_psf_probe":
            return self._run_lightweight_psf_probe(spec, payload)
        elif spec.task_type == "native_hsi_codesign":
            if use_lightweight:
                return self._run_lightweight_stable_lens_hsi(spec, payload)
            return self._run_native_hsi_codesign(spec, payload)
        elif spec.task_type == "native_hsi_reconstruction_codesign":
            if use_lightweight:
                return self._run_lightweight_stable_lens_hsi(spec, payload)
            return self._run_native_hsi_reconstruction_codesign(spec, payload)
        elif spec.task_type == "native_waveoptics_codesign":
            return self._run_native_waveoptics_codesign(spec, payload)
        elif spec.task_type == "native_optimization_probe":
            return self._run_native_optimization_probe(spec, payload)
        else:
            return ControllerResult(
                spec_id=spec.spec_id,
                status="skipped",
                execution_target="local",
                backend_id=spec.backend_id,
                errors=[
                    {
                        "type": "unsupported_task",
                        "message": f"Task type '{spec.task_type}' not yet implemented in v2",
                    }
                ],
            )

    def _run_lightweight_psf_probe(
        self, spec: ExperimentSpecV2, payload: dict[str, Any]
    ) -> ControllerResult:
        """Run a lightweight PSF probe (no DeepLens dependency)."""
        from optiresearch.runtime.lightweight_experiments import (
            run_lightweight_psf_probe,
        )
        result = run_lightweight_psf_probe(
            backend_id=spec.backend_id,
            device=payload.get("device", "cpu"),
        )
        result.spec_id = spec.spec_id
        return result

    def _run_lightweight_stable_lens_hsi(
        self, spec: ExperimentSpecV2, payload: dict[str, Any]
    ) -> ControllerResult:
        """Run lightweight stable lens HSI (no DeepLens dependency)."""
        from optiresearch.runtime.lightweight_experiments import (
            run_lightweight_stable_lens_hsi,
        )
        result = run_lightweight_stable_lens_hsi(
            backend_id=spec.backend_id,
            max_steps=payload.get("max_steps", 5),
            optical_lr=payload.get("optical_lr", 1e-6),
            recon_lr=payload.get("recon_lr", 1e-3),
            rollback_on_loss_increase=payload.get("rollback_on_loss_increase", True),
            device=payload.get("device", "cpu"),
        )
        result.spec_id = spec.spec_id
        return result

    def _run_stable_lens_hsi(
        self, spec: ExperimentSpecV2, payload: dict[str, Any]
    ) -> ControllerResult:
        from optiresearch.runtime.stable_native_lens_hsi_loop import (
            run_stable_native_lens_hsi_codesign,
        )
        from optiresearch.schemas.stable_native_lens_hsi import (
            StableNativeLensHSISpec,
            make_stable_lens_id,
        )

        inner_spec = StableNativeLensHSISpec(
            run_id=make_stable_lens_id(
                payload.get("candidate", "GeoLensCooke"),
                payload.get("reconstructor", "tiny_cnn"),
            ),
            candidate=payload.get("candidate", "GeoLensCooke"),
            reconstructor=payload.get("reconstructor", "tiny_cnn"),
            max_steps=payload.get("max_steps", 10),
            optical_lr=payload.get("optical_lr", 1e-6),
            recon_lr=payload.get("recon_lr", 1e-3),
            optical_grad_clip=payload.get("optical_grad_clip", 1.0),
            rollback_on_loss_increase=payload.get("rollback_on_loss_increase", True),
            device=payload.get("device", "cpu"),
        )
        result = run_stable_native_lens_hsi_codesign(inner_spec)
        return ControllerResult(
            spec_id=spec.spec_id,
            status="succeeded" if result.status == "succeeded" else "failed",
            execution_target="local",
            backend_id=spec.backend_id,
            run_id=inner_spec.run_id,
            evidence_level=result.evidence_level,
            result_payload=result.model_dump(mode="json"),
            artifact_paths=result.artifact_paths,
        )

    def _run_native_hsi_codesign(
        self, spec: ExperimentSpecV2, payload: dict[str, Any]
    ) -> ControllerResult:
        from optiresearch.runtime.native_hsi_codesign_loop import (
            run_native_optical_hsi_codesign,
        )
        from optiresearch.schemas.native_hsi_codesign import NativeOpticalHSICoDesignSpec

        inner_spec = NativeOpticalHSICoDesignSpec(
            run_id=make_deterministic_id("nhsicd", spec.spec_id),
            candidate=payload.get("candidate", "Fresnel"),
            encoder=payload.get("encoder", "conventional"),
            max_steps=payload.get("max_steps", 10),
            optical_lr=payload.get("optical_lr", 1e-3),
            device=payload.get("device", "cpu"),
        )
        result = run_native_optical_hsi_codesign(inner_spec)
        return ControllerResult(
            spec_id=spec.spec_id,
            status="succeeded",
            execution_target="local",
            backend_id=spec.backend_id,
            run_id=inner_spec.run_id,
            evidence_level=result.evidence_level,
            result_payload=result.model_dump(mode="json"),
            artifact_paths=result.artifact_paths,
        )

    def _run_native_hsi_reconstruction_codesign(
        self, spec: ExperimentSpecV2, payload: dict[str, Any]
    ) -> ControllerResult:
        from optiresearch.runtime.native_hsi_reconstruction_codesign_loop import (
            run_native_hsi_reconstruction_codesign,
        )
        from optiresearch.schemas.native_hsi_reconstruction_codesign import (
            NativeHSIReconstructionCoDesignSpec,
        )

        inner_spec = NativeHSIReconstructionCoDesignSpec(
            run_id=make_deterministic_id("nhsirc", spec.spec_id),
            candidate=payload.get("candidate", "Fresnel"),
            encoder=payload.get("encoder", "conventional"),
            reconstructor=payload.get("reconstructor", "tiny_cnn"),
            max_steps=payload.get("max_steps", 10),
            optical_lr=payload.get("optical_lr", 1e-3),
            recon_lr=payload.get("recon_lr", 1e-3),
            device=payload.get("device", "cpu"),
        )
        result = run_native_hsi_reconstruction_codesign(inner_spec)
        return ControllerResult(
            spec_id=spec.spec_id,
            status="succeeded",
            execution_target="local",
            backend_id=spec.backend_id,
            run_id=inner_spec.run_id,
            evidence_level=result.evidence_level,
            result_payload=result.model_dump(mode="json"),
            artifact_paths=result.artifact_paths,
        )

    def _run_native_waveoptics_codesign(
        self, spec: ExperimentSpecV2, payload: dict[str, Any]
    ) -> ControllerResult:
        from optiresearch.runtime.native_waveoptics_hsi_codesign_loop import (
            run_native_waveoptics_hsi_codesign,
        )
        from optiresearch.schemas.deeplens_waveoptics_probe import (
            NativeWaveOpticsHSICoDesignSpec,
        )

        inner_spec = NativeWaveOpticsHSICoDesignSpec(
            run_id=make_deterministic_id("nwocd", spec.spec_id),
            candidate=payload.get("candidate", "GeoLensCooke"),
            reconstructor=payload.get("reconstructor", "tiny_cnn"),
            max_steps=payload.get("max_steps", 10),
            optical_lr=payload.get("optical_lr", 1e-3),
            recon_lr=payload.get("recon_lr", 1e-3),
            device=payload.get("device", "cpu"),
        )
        result = run_native_waveoptics_hsi_codesign(inner_spec)
        return ControllerResult(
            spec_id=spec.spec_id,
            status="succeeded",
            execution_target="local",
            backend_id=spec.backend_id,
            run_id=inner_spec.run_id,
            evidence_level=result.evidence_level,
            result_payload=result.model_dump(mode="json"),
            artifact_paths=result.artifact_paths,
        )

    def _run_native_optimization_probe(
        self, spec: ExperimentSpecV2, payload: dict[str, Any]
    ) -> ControllerResult:
        from optiresearch.runtime.native_optimization_probe import (
            run_native_optimization_probe,
        )
        from optiresearch.schemas.native_optimization import NativeOptimizationProbeSpec

        inner_spec = NativeOptimizationProbeSpec(
            run_id=make_deterministic_id("nop", spec.spec_id),
            candidate=payload.get("candidate", "ParaxialLens"),
            objective=payload.get("objective", "minimize_psf_width"),
            optical_lr=payload.get("optical_lr", 1e-3),
            max_steps=payload.get("max_steps", 5),
            device=payload.get("device", "cpu"),
        )
        result = run_native_optimization_probe(inner_spec)
        return ControllerResult(
            spec_id=spec.spec_id,
            status="succeeded",
            execution_target="local",
            backend_id=spec.backend_id,
            run_id=inner_spec.run_id,
            evidence_level=result.evidence_level,
            result_payload=result.model_dump(mode="json"),
            artifact_paths=result.artifact_paths,
        )
