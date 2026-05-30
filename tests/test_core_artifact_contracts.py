"""Define 7 core artifact contracts."""
from __future__ import annotations

from optiresearch.schemas.artifact_contract import ArtifactContract


AC_DIAGNOSTIC = ArtifactContract(
    contract_id="ac_diagnostic",
    handler_id="diagnostic",
    required_artifacts=["result.json", "metrics.json"],
    optional_artifacts=["diagnosis.json", "autograd_audit.json", "stability_trace.json"],
    artifact_roles={
        "result.json": "execution_result",
        "metrics.json": "primary_metric",
        "diagnosis.json": "diagnostic_metric",
    },
    sha256_required=True,
    artifactstore_registration_required=True,
    evidence_binding_required=False,
    missing_artifact_policy="structured_warning",
)

AC_COMPONENT_PROBE = ArtifactContract(
    contract_id="ac_component_probe",
    handler_id="component_probe",
    required_artifacts=["result.json", "metrics.json", "artifact_manifest.json"],
    optional_artifacts=["psf_stats.json", "component_trace.json"],
    artifact_roles={
        "result.json": "execution_result",
        "metrics.json": "primary_metric",
        "psf_stats.json": "psf_artifact",
        "artifact_manifest.json": "report",
    },
    sha256_required=True,
    artifactstore_registration_required=True,
    evidence_binding_required=True,
    missing_artifact_policy="partial_evidence",
)

AC_NATIVE_GEOLENS_STABILITY = ArtifactContract(
    contract_id="ac_native_geolens_stability",
    handler_id="deeplens_native_geolens_hsi_codesign",
    required_artifacts=["result.json", "metrics.json", "artifact_manifest.json", "report.md"],
    optional_artifacts=["psf_stats.json", "stability_trace.json"],
    artifact_roles={
        "result.json": "execution_result",
        "metrics.json": "primary_metric",
        "report.md": "report",
        "psf_stats.json": "psf_artifact",
        "stability_trace.json": "diagnostic_metric",
    },
    sha256_required=True,
    artifactstore_registration_required=True,
    evidence_binding_required=True,
    missing_artifact_policy="needs_followup",
)

AC_NATIVE_GEOLENS_BENCHMARK = ArtifactContract(
    contract_id="ac_native_geolens_benchmark",
    handler_id="native_geolens_stability_benchmark",
    required_artifacts=["result.json", "metrics.json", "artifact_manifest.json",
                        "benchmark_summary.json", "benchmark_results.csv", "report.md"],
    optional_artifacts=["benchmark_failure_records.json", "stability_trace.json"],
    artifact_roles={
        "result.json": "execution_result",
        "metrics.json": "primary_metric",
        "benchmark_summary.json": "benchmark_summary",
        "benchmark_results.csv": "benchmark_table",
        "report.md": "report",
        "benchmark_failure_records.json": "diagnostic_metric",
    },
    sha256_required=True,
    artifactstore_registration_required=True,
    evidence_binding_required=True,
    missing_artifact_policy="needs_followup",
)

AC_BENCHMARK_FAILURE_ANALYSIS = ArtifactContract(
    contract_id="ac_benchmark_failure_analysis",
    handler_id="benchmark_failure_analysis",
    required_artifacts=["failure_analysis.json", "metrics.json", "report.md"],
    optional_artifacts=["benchmark_failure_records.json"],
    artifact_roles={
        "failure_analysis.json": "execution_result",
        "metrics.json": "primary_metric",
        "report.md": "report",
    },
    sha256_required=True,
    artifactstore_registration_required=True,
    evidence_binding_required=True,
    missing_artifact_policy="partial_evidence",
)

AC_REMOTE_JOB = ArtifactContract(
    contract_id="ac_remote_job",
    handler_id="remote_job",
    required_artifacts=["result.json", "metrics.json", "artifact_manifest.json"],
    optional_artifacts=["report.md", "psf_stats.json", "stability_trace.json"],
    artifact_roles={
        "result.json": "execution_result",
        "metrics.json": "primary_metric",
        "artifact_manifest.json": "report",
        "report.md": "report",
    },
    sha256_required=True,
    artifactstore_registration_required=True,
    evidence_binding_required=True,
    missing_artifact_policy="needs_followup",
)

AC_AGENT_PLAN = ArtifactContract(
    contract_id="ac_agent_plan",
    handler_id="agent_plan_execution",
    required_artifacts=["result.json", "metrics.json", "report.md"],
    optional_artifacts=["plan_trace.json", "artifacts_manifest.json"],
    artifact_roles={
        "result.json": "execution_result",
        "metrics.json": "primary_metric",
        "report.md": "report",
    },
    sha256_required=True,
    artifactstore_registration_required=True,
    evidence_binding_required=True,
    missing_artifact_policy="partial_evidence",
)


_CONTRACTS = {
    c.contract_id: c for c in [
        AC_DIAGNOSTIC, AC_COMPONENT_PROBE, AC_NATIVE_GEOLENS_STABILITY,
        AC_NATIVE_GEOLENS_BENCHMARK, AC_BENCHMARK_FAILURE_ANALYSIS,
        AC_REMOTE_JOB, AC_AGENT_PLAN,
    ]
}


def get_all_artifact_contracts() -> dict[str, ArtifactContract]:
    return dict(_CONTRACTS)


def get_artifact_contract(contract_id: str) -> ArtifactContract | None:
    return _CONTRACTS.get(contract_id)
