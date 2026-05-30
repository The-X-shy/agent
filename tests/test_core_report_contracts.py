"""Define 8 core report contracts."""
from __future__ import annotations

from optiresearch.schemas.report_contract import ReportContract


RC_AGENT_PLAN_EXECUTION = ReportContract(
    report_contract_id="rc_agent_plan",
    report_type="agent_plan_execution_report",
    exporter_cli="export-agent-plan-execution-report",
    required_sections=["Plan Summary", "Execution Steps", "Artifacts", "Claim Boundary"],
    optional_sections=["Error Details", "Retry History"],
    required_tables=["step_results"],
    required_fields=["plan_id", "status", "handler_id", "evidence_level", "final_claim_ceiling"],
    linked_artifacts=["result.json", "metrics.json", "report.md"],
    safe_wording_required=True,
    blocked_claims_section_required=True,
    evidence_level_section_required=True,
)

RC_REMOTE_DIAGNOSTIC = ReportContract(
    report_contract_id="rc_remote_diagnostic",
    report_type="remote_diagnostic_report",
    exporter_cli="export-remote-diagnostic-report",
    required_sections=["Diagnostic Summary", "Artifact Manifest", "Evidence Level", "Claim Boundary"],
    optional_sections=["PSF Statistics", "Error Details"],
    required_tables=["diagnostic_results"],
    required_fields=["diagnostic_type", "handler_id", "remote_job_id", "evidence_level"],
    linked_artifacts=["result.json", "metrics.json", "artifact_manifest.json"],
    safe_wording_required=True,
    blocked_claims_section_required=True,
    evidence_level_section_required=True,
)

RC_COMPONENT_PROBE = ReportContract(
    report_contract_id="rc_component_probe",
    report_type="component_probe_report",
    exporter_cli="export-component-probe-report",
    required_sections=["Component Probe Summary", "Component Metrics", "Claim Boundary",
                       "Blocked Claims", "Evidence Level"],
    optional_sections=["PSF Comparison", "Gradient Analysis"],
    required_tables=["component_results"],
    required_fields=["component_id", "backend_id", "evidence_level", "claim_ceiling"],
    linked_artifacts=["result.json", "metrics.json", "psf_stats.json"],
    safe_wording_required=True,
    blocked_claims_section_required=True,
    evidence_level_section_required=True,
)

RC_NATIVE_GEOLENS_STABILITY = ReportContract(
    report_contract_id="rc_native_geolens_stability",
    report_type="native_geolens_stability_report",
    exporter_cli="export-native-geolens-stabilization-report",
    required_sections=["Stability Summary", "Optimization Metrics", "Rollback History",
                       "Claim Boundary", "Blocked Claims", "Evidence Level", "Safe Wording"],
    optional_sections=["PSF Statistics", "Gradient Trace"],
    required_tables=["optimization_results", "stability_metrics"],
    required_fields=["stability_score", "rollback_count", "mse", "psnr", "sam", "evidence_level"],
    linked_artifacts=["result.json", "metrics.json", "report.md"],
    safe_wording_required=True,
    blocked_claims_section_required=True,
    evidence_level_section_required=True,
)

RC_NATIVE_GEOLENS_BENCHMARK = ReportContract(
    report_contract_id="rc_native_geolens_benchmark",
    report_type="native_geolens_benchmark_report",
    exporter_cli="export-native-geolens-benchmark-report",
    required_sections=[
        "Benchmark Summary",
        "Completed Configurations (Completed-Only)",
        "Improvement Rates (Completed-Only)",
        "Full-Grid Improvement Rates",
        "Claim Boundary",
        "Blocked Claims",
        "Evidence Level",
        "Safe Wording",
    ],
    optional_sections=["PSF Statistics", "Stability Trace", "Failure Analysis"],
    required_tables=["improvement_rates_completed_only", "improvement_rates_full_grid", "metric_statistics"],
    required_fields=["completed_count", "unsupported_count", "failed_count",
                     "all_metrics_improved_rate", "all_metrics_improved_rate_full_grid",
                     "evidence_level"],
    linked_artifacts=["benchmark_summary.json", "benchmark_results.csv", "report.md"],
    linked_claims=["reproducible_synthetic_stability"],
    safe_wording_required=True,
    blocked_claims_section_required=True,
    evidence_level_section_required=True,
)

RC_BENCHMARK_FAILURE = ReportContract(
    report_contract_id="rc_benchmark_failure",
    report_type="native_geolens_benchmark_failure_report",
    exporter_cli="export-native-geolens-benchmark-failure-report",
    required_sections=["Failure Summary", "Failure Records", "Root Cause Analysis",
                       "Evidence Level", "Claim Boundary"],
    optional_sections=["Recovery Recommendations"],
    required_tables=["failure_records"],
    required_fields=["failure_count", "failure_modes", "evidence_level"],
    linked_artifacts=["failure_analysis.json", "benchmark_failure_records.json"],
    safe_wording_required=True,
    blocked_claims_section_required=True,
    evidence_level_section_required=True,
)

RC_DESIGN_STRATEGY = ReportContract(
    report_contract_id="rc_design_strategy",
    report_type="deeplens_design_strategy_report",
    exporter_cli="export-deeplens-design-strategy-report",
    required_sections=["Design Strategies", "Strategy Families", "Claim Boundary", "Evidence Level"],
    optional_sections=["Failure Compatibility Matrix"],
    required_tables=["strategy_list"],
    required_fields=["strategy_id", "strategy_family", "evidence_level", "claim_ceiling"],
    linked_artifacts=[],
    safe_wording_required=True,
    blocked_claims_section_required=False,
    evidence_level_section_required=True,
)

RC_EVIDENCE_TABLES = ReportContract(
    report_contract_id="rc_evidence_tables",
    report_type="evidence_tables",
    exporter_cli="export-evidence-tables",
    required_sections=["Evidence Distribution", "Evidence by Level", "Claim Boundary"],
    optional_sections=["Artifact Coverage"],
    required_tables=["evidence_distribution"],
    required_fields=["evidence_level", "claim_count", "artifact_count"],
    linked_artifacts=["evidence_distribution.json"],
    safe_wording_required=True,
    blocked_claims_section_required=True,
    evidence_level_section_required=True,
)

_CONTRACTS = {
    c.report_contract_id: c for c in [
        RC_AGENT_PLAN_EXECUTION, RC_REMOTE_DIAGNOSTIC, RC_COMPONENT_PROBE,
        RC_NATIVE_GEOLENS_STABILITY, RC_NATIVE_GEOLENS_BENCHMARK,
        RC_BENCHMARK_FAILURE, RC_DESIGN_STRATEGY, RC_EVIDENCE_TABLES,
    ]
}


def get_all_report_contracts() -> dict[str, ReportContract]:
    return dict(_CONTRACTS)


def get_report_contract(contract_id: str) -> ReportContract | None:
    return _CONTRACTS.get(contract_id)
