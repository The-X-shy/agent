"""Define core artifact contracts (Phase 69 — reconciled handler_ids)."""
from __future__ import annotations

from optiresearch.schemas.artifact_contract import ArtifactContract


AC_DIAGNOSTIC = ArtifactContract(
    contract_id="ac_diagnostic",
    handler_id="deeplens_trainable_parameter_inspection",
    required_artifacts=["result.json", "metrics_summary.json"],
    optional_artifacts=["diagnosis.json", "autograd_audit.json", "stability_trace.json"],
    artifact_roles={
        "result.json": "execution_result",
        "metrics_summary.json": "primary_metric",
        "diagnosis.json": "auxiliary",
    },
    sha256_required=True,
    artifactstore_registration_required=True,
    evidence_binding_required=False,
    missing_artifact_policy="structured_warning",
)

AC_COMPONENT_PROBE = ArtifactContract(
    contract_id="ac_component_probe",
    handler_id="deeplens_component_first_probe",
    required_artifacts=["result.json", "metrics_summary.json", "artifact_manifest.json"],
    optional_artifacts=["psf_stats.json", "component_trace.json"],
    artifact_roles={
        "result.json": "execution_result",
        "metrics_summary.json": "primary_metric",
        "psf_stats.json": "optical_artifact",
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
    required_artifacts=["result.json", "metrics_summary.json", "artifact_manifest.json", "report.md"],
    optional_artifacts=["psf_stats.json", "stability_trace.json"],
    artifact_roles={
        "result.json": "execution_result",
        "metrics_summary.json": "primary_metric",
        "report.md": "report",
        "psf_stats.json": "optical_artifact",
        "stability_trace.json": "trace",
    },
    sha256_required=True,
    artifactstore_registration_required=True,
    evidence_binding_required=True,
    missing_artifact_policy="needs_followup",
)

AC_NATIVE_GEOLENS_BENCHMARK = ArtifactContract(
    contract_id="ac_native_geolens_benchmark",
    handler_id="native_geolens_stability_benchmark",
    required_artifacts=["result.json", "metrics_summary.json", "artifact_manifest.json",
                        "benchmark_summary.json", "benchmark_results.csv", "report.md"],
    optional_artifacts=["benchmark_failure_records.json", "stability_trace.json"],
    artifact_roles={
        "result.json": "execution_result",
        "metrics_summary.json": "primary_metric",
        "benchmark_summary.json": "execution_result",
        "benchmark_results.csv": "auxiliary",
        "report.md": "report",
        "benchmark_failure_records.json": "auxiliary",
    },
    sha256_required=True,
    artifactstore_registration_required=True,
    evidence_binding_required=True,
    missing_artifact_policy="needs_followup",
)

AC_BENCHMARK_FAILURE_ANALYSIS = ArtifactContract(
    contract_id="ac_benchmark_failure_analysis",
    handler_id="deeplens_native_geolens_hsi_codesign",
    required_artifacts=["failure_analysis.json", "metrics_summary.json", "report.md"],
    optional_artifacts=["benchmark_failure_records.json"],
    artifact_roles={
        "failure_analysis.json": "execution_result",
        "metrics_summary.json": "primary_metric",
        "report.md": "report",
    },
    sha256_required=True,
    artifactstore_registration_required=True,
    evidence_binding_required=True,
    missing_artifact_policy="partial_evidence",
)

AC_REMOTE_JOB = ArtifactContract(
    contract_id="ac_remote_job",
    handler_id="remote_native_geolens_validation",
    required_artifacts=["result.json", "metrics_summary.json", "artifact_manifest.json"],
    optional_artifacts=["report.md", "psf_stats.json", "stability_trace.json"],
    artifact_roles={
        "result.json": "execution_result",
        "metrics_summary.json": "primary_metric",
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
    handler_id="report_negative_result_doc",
    required_artifacts=["result.json", "metrics_summary.json", "report.md"],
    optional_artifacts=["plan_trace.json", "artifact_manifest.json"],
    artifact_roles={
        "result.json": "execution_result",
        "metrics_summary.json": "primary_metric",
        "report.md": "report",
    },
    sha256_required=True,
    artifactstore_registration_required=True,
    evidence_binding_required=True,
    missing_artifact_policy="partial_evidence",
)

# New Phase 69 contracts
AC_REMOTE_DIAGNOSTIC_JOB = ArtifactContract(
    contract_id="ac_remote_diagnostic_job",
    handler_id="deeplens_autograd_audit",
    required_artifacts=["result.json", "diagnostic_metrics.json", "artifact_manifest.json", "report.md"],
    optional_artifacts=["stability_trace.json"],
    artifact_roles={
        "result.json": "execution_result",
        "diagnostic_metrics.json": "primary_metric",
        "artifact_manifest.json": "report",
        "report.md": "report",
    },
    sha256_required=True,
    artifactstore_registration_required=True,
    evidence_binding_required=True,
    missing_artifact_policy="structured_warning",
)

AC_SYSTEM_CAPABILITY = ArtifactContract(
    contract_id="ac_system_capability",
    handler_id="system_capability",
    required_artifacts=[
        "system_capability_registry.json",
        "contract_coverage.json",
        "claim_policy_matrix.json",
    ],
    optional_artifacts=["system_capability_registry.md", "contract_coverage.md", "claim_policy_matrix.md"],
    artifact_roles={
        "system_capability_registry.json": "execution_result",
        "contract_coverage.json": "auxiliary",
        "claim_policy_matrix.json": "auxiliary",
    },
    sha256_required=False,
    artifactstore_registration_required=False,
    evidence_binding_required=False,
    missing_artifact_policy="structured_warning",
)

_CONTRACTS = {
    c.contract_id: c for c in [
        AC_DIAGNOSTIC, AC_COMPONENT_PROBE, AC_NATIVE_GEOLENS_STABILITY,
        AC_NATIVE_GEOLENS_BENCHMARK, AC_BENCHMARK_FAILURE_ANALYSIS,
        AC_REMOTE_JOB, AC_AGENT_PLAN, AC_REMOTE_DIAGNOSTIC_JOB,
        AC_SYSTEM_CAPABILITY,
    ]
}


def get_all_artifact_contracts() -> dict[str, ArtifactContract]:
    return dict(_CONTRACTS)


def get_artifact_contract(contract_id: str) -> ArtifactContract | None:
    return _CONTRACTS.get(contract_id)
