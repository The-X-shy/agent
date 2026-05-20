"""Rule-based Simulation Experimentalist."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from optiresearch.memory.meta_trace import MetaTraceWriter
from optiresearch.memory.schemas import MetaTrace, make_trace_id
from optiresearch.runtime.backend_metadata import backend_metadata
from optiresearch.schemas.experiment import ExperimentSpec
from optiresearch.skills.executor import SkillExecutor
from optiresearch.skills.validator import SkillValidator
from optiresearch.storage.file_artifact_store import FileArtifactStore


class SimulationExperimentalist:
    """Run allowlisted optical simulation skills and register artifacts."""

    def __init__(self, validator: SkillValidator | None = None) -> None:
        self.validator = validator or SkillValidator()

    def run_simulation(
        self,
        plan: dict[str, Any],
        method: ExperimentSpec,
        skill_runtime: SkillExecutor,
        artifact_store: FileArtifactStore,
        trace_writer: MetaTraceWriter,
        output_dir: Path,
    ) -> dict[str, Any]:
        backend = method.backend
        command = "run_deeplens_psf" if backend == "deeplens" else "run_mock_psf"
        result = skill_runtime.execute(
            "deeplens-adapter",
            command,
            {
                "spec": method,
                "sweep": None,
                "output_dir": str(output_dir),
                "seed": plan.get("first_run", {}).get("seed", 42),
                "realization": plan.get("first_run", {}).get("realization", "auto"),
            },
        )
        errors = list(result.get("errors", []))
        errors.extend(self.validator.validate_artifacts(result))
        errors.extend(self.validator.validate_metrics(result))
        status = "failed" if errors or result.get("status") != "succeeded" else "succeeded"
        refs = []
        if status == "succeeded":
            result_metadata = result.get("metadata", {})
            common_metadata = backend_metadata(
                backend,
                {
                    "backend_capability_level": result_metadata.get(
                        "backend_capability_level",
                        result.get("metrics", {}).get("backend_capability_level"),
                    ),
                    "encoder_behavior_realized": result_metadata.get(
                        "encoder_behavior_realized",
                        result.get("metrics", {}).get("encoder_behavior_realized"),
                    ),
                    "encoder_behavior_realization_level": result_metadata.get(
                        "encoder_behavior_realization_level",
                        result.get("metrics", {}).get("encoder_behavior_realization_level"),
                    ),
                    "physical_validation_level": result_metadata.get(
                        "physical_validation_level",
                        result.get("metrics", {}).get("physical_validation_level"),
                    ),
                    "proxy_transform_applied": result_metadata.get(
                        "proxy_transform_applied",
                        result.get("metrics", {}).get("proxy_transform_applied"),
                    ),
                    "proxy_transform_name": result_metadata.get(
                        "proxy_transform_name",
                        result.get("metrics", {}).get("proxy_transform_name"),
                    ),
                    "selected_realization_level": result_metadata.get(
                        "selected_realization_level",
                        result.get("metrics", {}).get("selected_realization_level"),
                    ),
                    "semi_native_attempted": result_metadata.get(
                        "semi_native_attempted",
                        result.get("metrics", {}).get("semi_native_attempted"),
                    ),
                    "semi_native_succeeded": result_metadata.get(
                        "semi_native_succeeded",
                        result.get("metrics", {}).get("semi_native_succeeded"),
                    ),
                    "proxy_fallback_used": result_metadata.get(
                        "proxy_fallback_used",
                        result.get("metrics", {}).get("proxy_fallback_used"),
                    ),
                    "claim_scope": result_metadata.get(
                        "claim_scope",
                        result.get("metrics", {}).get("claim_scope"),
                    ),
                    "deeplens_version": result_metadata.get("deeplens_version"),
                    "python_executable": result_metadata.get("python_executable"),
                },
            )
            for raw_path in result["artifacts"]:
                path = Path(raw_path)
                producer = "DeepLensAdapter.simulate_psf_cube" if backend == "deeplens" else "MockDeepLensAdapter.simulate_psf_cube"
                refs.append(
                    artifact_store.register_file(
                        path,
                        workspace_id=plan["workspace_id"],
                        run_id=plan["run_id"],
                        trace_id=None,
                        producer=producer,
                        metadata={
                            "filename": path.name,
                            "producer_skill_id": "deeplens-adapter",
                            "producer_skill_version": "0.1.0",
                            **common_metadata,
                        },
                        metrics=result["metrics"] if path.name == "optical_metrics.json" else {},
                    )
                )
        trace = self._trace(plan, result, refs, errors, status)
        written = trace_writer.write_trace(trace)
        refs_with_trace = []
        for ref in refs:
            ref.trace_id = written.trace_id
            artifact_store.store.upsert(
                "artifacts",
                ref.artifact_id,
                ref,
                workspace_id=ref.workspace_id,
                run_id=ref.run_id,
            )
            refs_with_trace.append(ref)
        return {"trace": written, "artifacts": refs_with_trace, "result": result, "errors": errors}

    def _trace(
        self,
        plan: dict[str, Any],
        result: dict[str, Any],
        refs: list[Any],
        errors: list[Any],
        status: str,
    ) -> MetaTrace:
        backend = plan.get("first_run", {}).get("backend", "mock_deeplens")
        result_metadata = result.get("metadata", {})
        trace_backend_metadata = backend_metadata(
            backend,
            {
                "backend_capability_level": result_metadata.get(
                    "backend_capability_level",
                    result.get("metrics", {}).get("backend_capability_level"),
                ),
                "encoder_behavior_realized": result_metadata.get(
                    "encoder_behavior_realized",
                    result.get("metrics", {}).get("encoder_behavior_realized"),
                ),
                "encoder_behavior_realization_level": result_metadata.get(
                    "encoder_behavior_realization_level",
                    result.get("metrics", {}).get("encoder_behavior_realization_level"),
                ),
                "physical_validation_level": result_metadata.get(
                    "physical_validation_level",
                    result.get("metrics", {}).get("physical_validation_level"),
                ),
                "proxy_transform_applied": result_metadata.get(
                    "proxy_transform_applied",
                    result.get("metrics", {}).get("proxy_transform_applied"),
                ),
                "proxy_transform_name": result_metadata.get(
                    "proxy_transform_name",
                    result.get("metrics", {}).get("proxy_transform_name"),
                ),
                "selected_realization_level": result_metadata.get(
                    "selected_realization_level",
                    result.get("metrics", {}).get("selected_realization_level"),
                ),
                "semi_native_attempted": result_metadata.get(
                    "semi_native_attempted",
                    result.get("metrics", {}).get("semi_native_attempted"),
                ),
                "semi_native_succeeded": result_metadata.get(
                    "semi_native_succeeded",
                    result.get("metrics", {}).get("semi_native_succeeded"),
                ),
                "proxy_fallback_used": result_metadata.get(
                    "proxy_fallback_used",
                    result.get("metrics", {}).get("proxy_fallback_used"),
                ),
                "claim_scope": result_metadata.get(
                    "claim_scope",
                    result.get("metrics", {}).get("claim_scope"),
                ),
                "deeplens_version": result_metadata.get("deeplens_version"),
                "python_executable": result_metadata.get("python_executable"),
            },
        )
        task = f"run {backend} PSF simulation"
        now = datetime.now(timezone.utc)
        findings = [
            f"decision: use {backend} backend for reproducible optical simulation",
            f"metrics: {result.get('metrics', {})}",
        ]
        tool = "DeepLensAdapter.simulate_psf_cube" if backend == "deeplens" else "MockDeepLensAdapter.simulate_psf_cube"
        return MetaTrace(
            trace_id=make_trace_id(plan["workspace_id"], plan["run_id"], "simulation", "SimulationExperimentalist", task),
            workspace_id=plan["workspace_id"],
            run_id=plan["run_id"],
            branch_id=None,
            step_id="simulation",
            actor="SimulationExperimentalist",
            phase="Execute",
            task=task,
            skill_id="deeplens-adapter",
            skill_version="0.1.0",
            tool=tool,
            input_refs=[],
            output_refs=[ref.artifact_id for ref in refs],
            findings=findings,
            limitations=[self._error_text(error) for error in errors],
            next_action="compile run memory",
            status=status,
            timestamp_start=now,
            timestamp_end=now,
            parents=[],
            content_hash=None,
            metadata={
                "objective": plan["objective"],
                "command": "run_deeplens_psf" if backend == "deeplens" else "run_mock_psf",
                "structured_errors": [error for error in errors if isinstance(error, dict)],
                "artifact_types": sorted({ref.metadata.get("artifact_type", "unknown") for ref in refs}),
                **trace_backend_metadata,
            },
        )

    def _error_text(self, error: Any) -> str:
        if isinstance(error, dict):
            code = error.get("code", "ERROR")
            message = error.get("message", "")
            return f"{code}: {message}".strip()
        return str(error)
