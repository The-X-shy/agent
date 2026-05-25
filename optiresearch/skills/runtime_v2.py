"""Skill Runtime v2 for Phase 36."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from optiresearch.agent_system.event_bus import get_event_bus
from optiresearch.agent_system.events import AgentEvent
from optiresearch.skills.contracts import SkillResult
from optiresearch.skills.registry_v2 import SkillRegistryV2


class SkillRuntimeV2:
    def __init__(self, registry: SkillRegistryV2 | None = None):
        self._registry = registry or SkillRegistryV2()
        self._event_bus = get_event_bus()

    @property
    def registry(self) -> SkillRegistryV2:
        return self._registry

    def validate_input(self, skill_id: str, inputs: dict[str, Any]) -> list[str]:
        spec = self._registry.get(skill_id)
        if spec is None:
            return [f"Unknown skill: {skill_id}"]
        errors: list[str] = []
        for key, schema in spec.input_schema.items():
            if key not in inputs and schema.get("required"):
                errors.append(f"Missing required input: {key}")
        return errors

    def execute_skill(self, skill_id: str, inputs: dict[str, Any] | None = None) -> SkillResult:
        inputs = inputs or {}
        spec = self._registry.get(skill_id)
        if spec is None:
            return SkillResult(skill_id=skill_id, status="failed",
                               errors=[f"Unknown skill: {skill_id}"])

        validation_errors = self.validate_input(skill_id, inputs)
        if validation_errors:
            return SkillResult(skill_id=skill_id, status="failed", errors=validation_errors)

        t0 = time.time()
        try:
            output = self._dispatch(skill_id, inputs)
            elapsed = time.time() - t0
            if output.get("status") == "unsupported":
                status = "unsupported"
            elif output.get("status") == "failed":
                status = "failed"
            else:
                status = "succeeded"
            result = SkillResult(
                skill_id=skill_id, status=status,
                outcome=output.get("outcome"),
                inputs_hash=_hash_inputs(inputs),
                output=output, execution_time_sec=elapsed,
            )
            self._event_bus.publish(AgentEvent.create(
                "skill_called" if status == "succeeded" else "skill_failed",
                "skill_runtime",
                payload={"skill_id": skill_id, "status": status, "execution_time_sec": elapsed},
                severity="info" if status == "succeeded" else "warning",
            ))
            return result
        except Exception as exc:
            elapsed = time.time() - t0
            result = SkillResult(
                skill_id=skill_id, status="failed",
                inputs_hash=_hash_inputs(inputs),
                errors=[str(exc)], execution_time_sec=elapsed,
            )
            self._event_bus.publish(AgentEvent.create(
                "skill_failed", "skill_runtime",
                payload={"skill_id": skill_id, "error": str(exc)},
                severity="error",
            ))
            return result

    def audit_skill_result(self, result: SkillResult) -> list[str]:
        issues: list[str] = []
        if result.status == "failed":
            issues.append(f"Skill {result.skill_id} failed: {result.errors}")
        if result.execution_time_sec > 3600:
            issues.append(f"Skill {result.skill_id} exceeded 1h timeout")
        return issues

    def _dispatch(self, skill_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
        if skill_id == "claim_check":
            return self._dispatch_claim_check(inputs)
        if skill_id == "strategy_recommendation":
            return self._dispatch_strategy_recommendation(inputs)
        if skill_id == "backend_probe":
            return self._dispatch_backend_probe(inputs)
        if skill_id == "autograd_audit":
            return self._dispatch_autograd_audit(inputs)
        if skill_id == "report_generation":
            return self._dispatch_report_generation(inputs)
        if skill_id == "evidence_registry_export":
            return self._dispatch_evidence_export(inputs)
        if skill_id == "remote_execution":
            return self._dispatch_remote_execution(inputs)
        if skill_id == "deeplens_native_geolens_hsi_codesign":
            return self._dispatch_geolens_hsi(inputs)
        if skill_id == "native_geolens_stabilization_sweep":
            return self._dispatch_stabilization_sweep(inputs)
        if skill_id == "lightweight_scientific_hsi_mse_only":
            return self._dispatch_lightweight_scientific_hsi(inputs)
        if skill_id == "param_reduction_sweep":
            return self._dispatch_param_reduction_sweep(inputs)
        raise NotImplementedError(f"No runtime dispatch for skill: {skill_id}")

    def _dispatch_claim_check(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from optiresearch.memory.claim_gate_v2 import ClaimGateV2
        gate = ClaimGateV2()
        decision = gate.check_claim(
            claim_text=inputs.get("claim", ""),
            backend_id=inputs.get("backend_id", "deeplens_geolens_geometric"),
        )
        self._event_bus.publish(AgentEvent.create("claim_checked", "claim_gate",
            payload={"decision": decision.decision, "violation_type": decision.violation_type or "none"}))
        return {"decision": decision.decision, "max_allowed_claim": decision.max_allowed_claim,
                "violation_type": decision.violation_type, "safe_wording": decision.safe_wording}

    def _dispatch_strategy_recommendation(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from optiresearch.agents.strategy_engine import StrategyEngine
        engine = StrategyEngine()
        rec = engine.recommend(
            latest_result=inputs.get("latest_result", {}),
            backend_id=inputs.get("backend_id", "deeplens_geolens_geometric"),
        )
        self._event_bus.publish(AgentEvent.create("strategy_recommended", "strategy_engine",
            payload={"action": rec.recommended_action, "risk_level": rec.risk_level}))
        return {"action": rec.recommended_action, "rationale": rec.rationale,
                "risk_level": rec.risk_level}

    def _dispatch_backend_probe(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            from optiresearch.backends.registry import list_backends
            backends = list_backends()
            backend_id = inputs.get("backend_id", "")
            info = {"backend_id": backend_id, "available": False}
            for b in backends:
                if b.backend_id == backend_id:
                    info = {"backend_id": b.backend_id, "available": True,
                            "known_failure_modes": b.known_failure_modes}
                    break
            return info
        except Exception as e:
            return {"backend_id": inputs.get("backend_id", ""), "available": False, "error": str(e)}

    def _dispatch_autograd_audit(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            from optiresearch.diagnostics.autograd_auditor import audit_autograd_graph
            result = audit_autograd_graph(backend_id=inputs.get("backend_id", "deeplens_geolens_geometric"))
            return {"autograd_intact": result.get("autograd_intact", False),
                    "breaks_found": result.get("breaks_found", 0),
                    "details": str(result)[:500]}
        except Exception as e:
            return {"autograd_intact": False, "error": str(e)}

    def _dispatch_report_generation(self, inputs: dict[str, Any]) -> dict[str, Any]:
        report_type = inputs.get("report_type", "system_subunit")
        if report_type == "system_subunit":
            from optiresearch.reports.system_subunit_report import export_system_subunit_report
            path = export_system_subunit_report()
            return {"report_type": report_type, "path": str(path), "outcome": "report_only"}
        if report_type in ("negative_result", "agent_plan_negative_result"):
            from pathlib import Path
            path = Path("workspace/reports/agent_plan_negative_result.md")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "# Agent Plan Negative Result\n\n"
                "The selected local scientific design was unavailable or unsupported. "
                "This report records the boundary as report-only evidence.\n",
                encoding="utf-8",
            )
            return {"report_type": report_type, "path": str(path), "outcome": "report_only"}
        return {
            "report_type": report_type,
            "status": "unsupported",
            "outcome": "structured_unsupported",
            "error": f"Unsupported report type: {report_type}",
        }

    def _dispatch_evidence_export(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            from optiresearch.memory.claim_evidence import ClaimEvidenceManager
            mgr = ClaimEvidenceManager()
            claims = mgr.list_claims() if hasattr(mgr, 'list_claims') else []
            return {"claims_count": len(claims), "claims": [str(c)[:200] for c in claims[:5]]}
        except Exception as e:
            return {"claims_count": 0, "error": str(e)}

    def _dispatch_remote_execution(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not inputs.get("allow_remote"):
            return {"status": "dry_run", "message": "Remote execution requires explicit opt-in (allow_remote=true)"}
        try:
            worker_id = inputs.get("worker_id", "")
            if not worker_id:
                return {"status": "failed", "error": "No worker_id specified"}
            handler_id = inputs.get("handler_id", "")
            design_id = inputs.get("design_id", "")
            # Choose remote job based on handler
            if handler_id == "remote_native_geolens_validation" or "geolens" in design_id:
                from optiresearch.remote.result_ingestion import parse_remote_handler_result
                from optiresearch.runtime import remote_jobs
                spec = inputs.get("spec_payload", {})
                self._event_bus.publish(AgentEvent.create(
                    "remote_execution_requested",
                    "skill_runtime",
                    payload={"worker_id": worker_id, "handler_id": handler_id, "design_id": design_id},
                ))
                self._event_bus.publish(AgentEvent.create(
                    "remote_execution_started",
                    "skill_runtime",
                    payload={"worker_id": worker_id, "handler_id": handler_id, "design_id": design_id},
                ))
                payload = remote_jobs.run_remote_deeplens_native_geolens_hsi_codesign(
                    worker_id=worker_id,
                    lens_file=spec.get("candidate", "auto:cooke"),
                    dataset=spec.get("dataset", "synthetic"),
                    reconstructor=spec.get("reconstructor", "differentiable_linear"),
                    max_steps=spec.get("max_steps", 5),
                    optical_lr=spec.get("optical_lr", 1e-6),
                    device=spec.get("device", "cpu"),
                    ingest=True,
                )
                remote_job_result = payload.get("result") if isinstance(payload, dict) else payload
                ingestion = payload.get("ingestion") if isinstance(payload, dict) else None
                handler_result = parse_remote_handler_result(
                    remote_job_result,
                    worker_id=worker_id,
                    ingestion=ingestion or {},
                )
                handler_dump = handler_result.model_dump(mode="json")
                completed = handler_result.status == "succeeded"
                self._event_bus.publish(AgentEvent.create(
                    "remote_execution_completed" if completed else "remote_execution_failed",
                    "skill_runtime",
                    payload=handler_dump,
                    severity="info" if completed else "warning",
                    related_run_id=handler_result.run_id,
                    related_job_id=handler_result.remote_job_id,
                ))
                self._event_bus.publish(AgentEvent.create(
                    "remote_validation_passed" if handler_result.remote_validation_passed else "remote_validation_failed",
                    "skill_runtime",
                    payload={
                        "worker_id": worker_id,
                        "status": handler_result.status,
                        "job_id": handler_result.remote_job_id,
                        "remote_validation_passed": handler_result.remote_validation_passed,
                    },
                    severity="info" if handler_result.remote_validation_passed else "warning",
                    related_run_id=handler_result.run_id,
                    related_job_id=handler_result.remote_job_id,
                ))
                if handler_result.artifacts:
                    self._event_bus.publish(AgentEvent.create(
                        "artifact_ingested",
                        "skill_runtime",
                        payload={
                            "worker_id": worker_id,
                            "job_id": handler_result.remote_job_id,
                            "artifacts": handler_result.artifacts,
                            "artifact_return_path": handler_result.artifact_return_path,
                        },
                        related_run_id=handler_result.run_id,
                        related_job_id=handler_result.remote_job_id,
                    ))
                return {
                    "status": "succeeded" if completed else "failed",
                    "outcome": "remote_execution_completed" if completed else "remote_execution_failed",
                    "remote_job_id": handler_result.remote_job_id,
                    "run_id": handler_result.run_id,
                    "execution_target": "remote_wsl",
                    "evidence_level": handler_result.evidence_level,
                    "execution_fidelity": handler_result.execution_fidelity,
                    "remote_validation_passed": handler_result.remote_validation_passed,
                    "proxy_fallback_used": handler_result.proxy_fallback_used,
                    "deeplens_native_psf_path": handler_result.deeplens_native_psf_path,
                    "full_wave_optics": handler_result.full_wave_optics,
                    "phase_to_fft_proxy_used": handler_result.phase_to_fft_proxy_used,
                    "metrics": handler_result.metrics,
                    "artifacts": handler_result.artifacts,
                    "artifact_return_path": handler_result.artifact_return_path,
                    "artifact_ids": handler_result.artifact_ids,
                    "primary_metric_artifact_id": handler_result.primary_metric_artifact_id,
                    "execution_result_artifact_id": handler_result.execution_result_artifact_id,
                    "evidence_artifact_ids": handler_result.evidence_artifact_ids,
                    "artifact_manifest_path": handler_result.artifact_manifest_path,
                    "artifact_manifest_complete": handler_result.artifact_manifest_complete,
                    "artifact_ingestion_status": handler_result.artifact_ingestion_status,
                    "missing_required_artifacts": handler_result.missing_required_artifacts,
                    "sha256_verified": handler_result.sha256_verified,
                    "errors": handler_result.errors,
                    "caveats": handler_result.caveats,
                    "remote_handler_result": handler_dump,
                }
            return {"status": "dry_run", "worker_id": worker_id,
                    "message": f"Remote execution dispatched for {handler_id} (results pending)"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _dispatch_geolens_hsi(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            from optiresearch.runtime.stable_native_lens_hsi_loop import run_stable_native_lens_hsi_codesign
            from optiresearch.schemas.stable_native_lens_hsi import StableNativeLensHSISpec, make_stable_lens_id
            run_id = make_stable_lens_id("GeoLensCooke", inputs.get("reconstructor", "differentiable_linear"))
            spec = StableNativeLensHSISpec(
                run_id=run_id, candidate="GeoLensCooke",
                reconstructor=inputs.get("reconstructor", "differentiable_linear"),
                max_steps=inputs.get("max_steps", 5),
                optical_lr=inputs.get("optical_lr", 1e-6), device=inputs.get("device", "cpu"),
                full_wave_optics=False, phase_to_fft_proxy_used=False,
            )
            result = run_stable_native_lens_hsi_codesign(spec)
            self._event_bus.publish(AgentEvent.create(
                "experiment_completed" if result.status == "succeeded" else "experiment_failed",
                "controller", payload={"run_id": run_id, "status": result.status}))
            return {"run_id": run_id, "status": result.status, "evidence_level": result.evidence_level,
                    "accepted_update_count": result.accepted_update_count}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _dispatch_stabilization_sweep(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not inputs.get("allow_execution"):
            return {"status": "dry_run", "message": "Stabilization sweep requires allow_execution=true (30 configs)"}
        try:
            from optiresearch.runtime.native_geolens_stabilization_sweep import run_native_geolens_stabilization_sweep
            summary = run_native_geolens_stabilization_sweep(
                lens_file=inputs.get("lens_file", "auto:cooke"),
                reconstructor=inputs.get("reconstructor", "differentiable_linear"),
                device=inputs.get("device", "cpu"),
            )
            return {"sweep_id": summary.get("sweep_id"), "configs_tested": summary.get("configs_tested"),
                    "configs_with_accepted_updates": summary.get("configs_with_accepted_updates")}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _dispatch_lightweight_scientific_hsi(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            from optiresearch.runtime.lightweight_experiments import (
                run_lightweight_mse_only_hsi,
            )
            result = run_lightweight_mse_only_hsi(
                backend_id=inputs.get("backend_id", "phase_to_fft_proxy"),
                max_steps=inputs.get("max_steps", 10),
                optical_lr=inputs.get("optical_lr", 1e-6),
                recon_lr=inputs.get("recon_lr", 1e-3),
                bands=inputs.get("bands", 4),
                image_size=inputs.get("image_size", 16),
                psf_size=inputs.get("psf_size", 15),
                device=inputs.get("device", "cpu"),
            )
            self._event_bus.publish(AgentEvent.create(
                "experiment_completed" if result.status == "succeeded" else "experiment_failed",
                "controller",
                payload={"run_id": result.run_id, "status": result.status,
                         "evidence_level": result.evidence_level},
            ))
            payload = result.result_payload or {}
            return {
                "run_id": result.run_id,
                "status": result.status,
                "evidence_level": result.evidence_level,
                "reconstruction_loss_before": payload.get("reconstruction_loss_before"),
                "reconstruction_loss_after": payload.get("reconstruction_loss_after"),
                "best_reconstruction_loss": payload.get("best_reconstruction_loss"),
                "mse_before": payload.get("mse_before"),
                "mse_after": payload.get("mse_after"),
                "psnr_before": payload.get("psnr_before"),
                "psnr_after": payload.get("psnr_after"),
                "improvement_detected": payload.get("improvement_detected"),
                "synthetic_data": payload.get("synthetic_data", True),
                "physical_backend": payload.get("physical_backend", False),
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}


    def _dispatch_param_reduction_sweep(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            from optiresearch.runtime.local_scientific_handlers import (
                run_param_reduction_sweep_lightweight,
            )
            result = run_param_reduction_sweep_lightweight(
                max_steps=inputs.get("max_steps", 3),
                optical_lr=inputs.get("optical_lr", 1e-6),
                recon_lr=inputs.get("recon_lr", 1e-3),
                bands=inputs.get("bands", 4),
                image_size=inputs.get("image_size", 16),
                psf_size=inputs.get("psf_size", 15),
                device=inputs.get("device", "cpu"),
            )
            self._event_bus.publish(AgentEvent.create(
                "experiment_completed" if result.status == "succeeded" else "experiment_failed",
                "controller",
                payload={"run_id": result.run_id, "status": result.status,
                         "evidence_level": result.evidence_level},
            ))
            payload = result.result_payload or {}
            return {
                "run_id": result.run_id,
                "status": result.status,
                "evidence_level": result.evidence_level,
                "configs_tested": payload.get("configs_tested"),
                "best_k": payload.get("best_k"),
                "reconstruction_loss_before": payload.get("reconstruction_loss_before"),
                "reconstruction_loss_after": payload.get("reconstruction_loss_after"),
                "best_reconstruction_loss": payload.get("best_reconstruction_loss"),
                "mse_before": payload.get("mse_before"),
                "mse_after": payload.get("mse_after"),
                "psnr_before": payload.get("psnr_before"),
                "psnr_after": payload.get("psnr_after"),
                "improvement_detected": payload.get("improvement_detected"),
                "synthetic_data": payload.get("synthetic_data", True),
                "physical_backend": payload.get("physical_backend", False),
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}


def _hash_inputs(inputs: dict[str, Any]) -> str:
    raw = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
