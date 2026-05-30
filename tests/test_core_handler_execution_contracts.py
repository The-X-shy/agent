"""Define and validate 12 core handler execution contracts."""
from __future__ import annotations

from optiresearch.schemas.execution_contract import ExecutionContract


_CONTRACTS: dict[str, ExecutionContract] = {}


def _register(c: ExecutionContract) -> ExecutionContract:
    _CONTRACTS[c.contract_id] = c
    return c


EC_DEEPLENS_NATIVE_GEOLENS_HSI = _register(ExecutionContract(
    contract_id="ec_deeplens_native_geolens_hsi",
    handler_id="deeplens_native_geolens_hsi_codesign",
    skill_id="deeplens_native_geolens_hsi_codesign",
    design_ids=["geolens_curriculum_probe", "geolens_regularized_probe"],
    backend_ids=["deeplens_geolens_geometric"],
    execution_modes=["local", "remote_opt_in"],
    required_inputs=["lens_file", "dataset_spec"],
    required_outputs=["result.json", "metrics.json", "artifact_manifest.json", "report.md"],
    required_metrics=["mse", "psnr", "sam", "stability_score"],
    optional_metrics=["grad_norm_max", "rollback_count"],
    status_values=["succeeded", "unsupported", "failed"],
    evidence_level_mapping={"local": "native_lens_simulation", "remote_opt_in": "native_lens_simulation"},
    claim_ceiling_mapping={"local": "native_lens_simulation", "remote_opt_in": "native_lens_simulation"},
    failure_modes=["gradient_instability", "rollback_triggered", "no_parameter_change"],
    retry_policy={"max_retries": 3, "backoff_sec": 60},
    timeout_policy={"local": 1200, "remote_opt_in": 3600},
    artifact_contract_id="ac_native_geolens_stability",
    report_contract_id="rc_native_geolens_stability",
))

EC_STABLE_NATIVE_LENS_HSI = _register(ExecutionContract(
    contract_id="ec_stable_native_lens_hsi",
    handler_id="stable_native_lens_hsi_codesign",
    skill_id="stable_native_lens_hsi_codesign",
    design_ids=["geolens_regularized_probe"],
    backend_ids=["deeplens_geolens_geometric"],
    execution_modes=["local", "remote_opt_in"],
    required_inputs=["lens_file", "dataset_spec", "stability_config"],
    required_outputs=["result.json", "metrics.json", "artifact_manifest.json", "report.md"],
    required_metrics=["mse", "psnr", "sam", "stability_score"],
    status_values=["succeeded", "unsupported", "failed"],
    evidence_level_mapping={"local": "stable_native_lens_hsi_codesign", "remote_opt_in": "stable_native_lens_hsi_codesign"},
    claim_ceiling_mapping={"local": "stable_native_lens_hsi_codesign", "remote_opt_in": "stable_native_lens_hsi_codesign"},
    failure_modes=["gradient_instability", "rollback_triggered"],
    retry_policy={"max_retries": 3, "backoff_sec": 60},
    timeout_policy={"local": 1200, "remote_opt_in": 3600},
    artifact_contract_id="ac_native_geolens_stability",
    report_contract_id="rc_native_geolens_stability",
))

EC_NATIVE_GEOLENS_BENCHMARK = _register(ExecutionContract(
    contract_id="ec_native_geolens_benchmark",
    handler_id="native_geolens_stability_benchmark",
    skill_id="native_geolens_stabilization_sweep",
    design_ids=["geolens_regularized_probe"],
    backend_ids=["deeplens_geolens_geometric"],
    execution_modes=["local", "remote_opt_in"],
    required_inputs=["lens_file", "dataset_spec", "seeds", "step_grid"],
    required_outputs=["result.json", "metrics.json", "artifact_manifest.json", "benchmark_summary.json", "report.md"],
    required_metrics=["completed_count", "unsupported_count", "failed_count", "completion_rate",
                      "all_metrics_improved_rate", "all_metrics_improved_rate_full_grid"],
    optional_metrics=["rollback_rate", "mean_mse_delta", "std_mse_delta"],
    status_values=["succeeded", "unsupported", "failed"],
    evidence_level_mapping={"local": "native_geolens_stability_benchmark", "remote_opt_in": "native_geolens_stability_benchmark"},
    claim_ceiling_mapping={"local": "native_geolens_stability_benchmark", "remote_opt_in": "native_geolens_stability_benchmark"},
    failure_modes=["low_completion_rate", "inconsistent_results", "rollback_rate_high"],
    retry_policy={"max_retries": 1, "backoff_sec": 120},
    timeout_policy={"local": 3600, "remote_opt_in": 7200},
    artifact_contract_id="ac_native_geolens_benchmark",
    report_contract_id="rc_native_geolens_benchmark",
))

EC_COMPONENT_FIRST_PROBE = _register(ExecutionContract(
    contract_id="ec_component_first_probe",
    handler_id="deeplens_component_first_probe",
    skill_id="deeplens_component_first_probe",
    design_ids=["component_first_fresnel_probe", "component_first_binary2phase_probe"],
    backend_ids=["deeplens_fresnel_component", "deeplens_binary2phase_component"],
    execution_modes=["local", "remote_opt_in"],
    required_inputs=["lens_file", "component_spec"],
    required_outputs=["result.json", "metrics.json"],
    required_metrics=["mse", "psnr"],
    status_values=["succeeded", "unsupported", "failed"],
    evidence_level_mapping={"local": "native_component_optimization", "remote_opt_in": "native_component_optimization"},
    claim_ceiling_mapping={"local": "native_component_optimization", "remote_opt_in": "native_component_optimization"},
    failure_modes=["component_not_differentiable", "gradient_flow_blocked"],
    retry_policy={"max_retries": 2, "backoff_sec": 30},
    timeout_policy={"local": 600, "remote_opt_in": 1800},
    artifact_contract_id="ac_component_probe",
    report_contract_id="rc_component_probe",
))

EC_COMPONENT_SURROGATE_HSI = _register(ExecutionContract(
    contract_id="ec_component_surrogate_hsi",
    handler_id="component_surrogate_hsi_codesign",
    skill_id="component_surrogate_hsi_codesign",
    design_ids=["component_first_fresnel_probe", "component_first_binary2phase_probe"],
    backend_ids=["component_surrogate_psf"],
    execution_modes=["local", "remote_opt_in"],
    required_inputs=["lens_file", "dataset_spec", "component_spec"],
    required_outputs=["result.json", "metrics.json", "artifact_manifest.json", "report.md"],
    required_metrics=["mse", "psnr", "sam"],
    status_values=["succeeded", "unsupported", "failed"],
    evidence_level_mapping={"local": "component_surrogate_hsi_codesign", "remote_opt_in": "component_surrogate_hsi_codesign"},
    claim_ceiling_mapping={"local": "component_surrogate_hsi_codesign", "remote_opt_in": "component_surrogate_hsi_codesign"},
    failure_modes=["surrogate_psf_insufficient", "component_not_differentiable"],
    retry_policy={"max_retries": 2, "backoff_sec": 60},
    timeout_policy={"local": 1200, "remote_opt_in": 3600},
    artifact_contract_id="ac_component_surrogate_hsi",
    report_contract_id="rc_component_surrogate_hsi",
))

EC_TRAINABLE_PARAM_INSPECTION = _register(ExecutionContract(
    contract_id="ec_trainable_param_inspection",
    handler_id="deeplens_trainable_parameter_inspection",
    skill_id="deeplens_trainable_parameter_inspection",
    design_ids=[],
    backend_ids=["deeplens_geolens_geometric"],
    execution_modes=["local", "remote_opt_in"],
    required_inputs=["lens_file"],
    required_outputs=["result.json", "metrics.json"],
    required_metrics=["trainable_parameter_count", "parameter_names"],
    optional_metrics=["parameter_shapes", "parameter_devices"],
    status_values=["succeeded", "unsupported", "failed"],
    evidence_level_mapping={"local": "diagnostic_evidence", "remote_opt_in": "diagnostic_evidence"},
    claim_ceiling_mapping={"local": "diagnostic_evidence", "remote_opt_in": "diagnostic_evidence"},
    failure_modes=["no_trainable_parameters", "native_api_error"],
    retry_policy={"max_retries": 1, "backoff_sec": 30},
    timeout_policy={"local": 300, "remote_opt_in": 900},
    artifact_contract_id="ac_diagnostic",
    report_contract_id="rc_remote_diagnostic",
))

EC_AUTOGRAD_AUDIT = _register(ExecutionContract(
    contract_id="ec_autograd_audit",
    handler_id="deeplens_autograd_audit",
    skill_id="autograd_audit",
    design_ids=[],
    backend_ids=["deeplens_geolens_geometric"],
    execution_modes=["local", "remote_opt_in"],
    required_inputs=["lens_file"],
    required_outputs=["result.json", "metrics.json"],
    required_metrics=["gradient_flow_verified", "autograd_graph_size"],
    optional_metrics=["gradient_norms", "parameter_gradients"],
    status_values=["succeeded", "unsupported", "failed"],
    evidence_level_mapping={"local": "diagnostic_evidence", "remote_opt_in": "diagnostic_evidence"},
    claim_ceiling_mapping={"local": "diagnostic_evidence", "remote_opt_in": "diagnostic_evidence"},
    failure_modes=["gradient_flow_blocked", "autograd_error"],
    retry_policy={"max_retries": 1, "backoff_sec": 30},
    timeout_policy={"local": 300, "remote_opt_in": 900},
    artifact_contract_id="ac_diagnostic",
    report_contract_id="rc_remote_diagnostic",
))

EC_CURRICULUM_PROBE = _register(ExecutionContract(
    contract_id="ec_curriculum_probe",
    handler_id="deeplens_curriculum_probe",
    skill_id="deeplens_curriculum_probe",
    design_ids=["geolens_curriculum_probe"],
    backend_ids=["deeplens_geolens_geometric"],
    execution_modes=["local", "remote_opt_in"],
    required_inputs=["lens_file", "curriculum_config"],
    required_outputs=["result.json", "metrics.json"],
    required_metrics=["stage_count", "mse_per_stage", "stability_per_stage"],
    status_values=["succeeded", "unsupported", "failed"],
    evidence_level_mapping={"local": "diagnostic_evidence", "remote_opt_in": "diagnostic_evidence"},
    claim_ceiling_mapping={"local": "diagnostic_evidence", "remote_opt_in": "diagnostic_evidence"},
    failure_modes=["no_parameter_change", "unstable_training"],
    retry_policy={"max_retries": 2, "backoff_sec": 30},
    timeout_policy={"local": 600, "remote_opt_in": 1800},
    artifact_contract_id="ac_diagnostic",
    report_contract_id="rc_remote_diagnostic",
))

EC_REGULARIZED_PROBE = _register(ExecutionContract(
    contract_id="ec_regularized_probe",
    handler_id="deeplens_regularized_probe",
    skill_id="deeplens_regularized_probe",
    design_ids=["geolens_regularized_probe"],
    backend_ids=["deeplens_geolens_geometric"],
    execution_modes=["local", "remote_opt_in"],
    required_inputs=["lens_file", "regularization_config"],
    required_outputs=["result.json", "metrics.json"],
    required_metrics=["regularization_terms", "mse", "regularization_loss"],
    status_values=["succeeded", "unsupported", "failed"],
    evidence_level_mapping={"local": "diagnostic_evidence", "remote_opt_in": "diagnostic_evidence"},
    claim_ceiling_mapping={"local": "diagnostic_evidence", "remote_opt_in": "diagnostic_evidence"},
    failure_modes=["unstable_training", "regularization_too_strong"],
    retry_policy={"max_retries": 2, "backoff_sec": 30},
    timeout_policy={"local": 600, "remote_opt_in": 1800},
    artifact_contract_id="ac_diagnostic",
    report_contract_id="rc_remote_diagnostic",
))

EC_OBJECTIVE_REDESIGN = _register(ExecutionContract(
    contract_id="ec_objective_redesign",
    handler_id="objective_redesign_simpler_metric",
    skill_id="lightweight_scientific_hsi_mse_only",
    design_ids=[],
    backend_ids=["phase_to_fft_proxy", "local_synthetic_hsi"],
    execution_modes=["dry_run", "local"],
    required_inputs=["objective_spec"],
    required_outputs=["redesign_proposal.json"],
    required_metrics=["simpler_metric_identified"],
    optional_metrics=["metric_complexity_reduction"],
    status_values=["succeeded", "needs_followup", "unsupported"],
    evidence_level_mapping={"dry_run": "lightweight_scientific_execution", "local": "lightweight_scientific_execution"},
    claim_ceiling_mapping={"dry_run": "lightweight_scientific_execution", "local": "lightweight_scientific_execution"},
    failure_modes=["no_simpler_alternative"],
    retry_policy={"max_retries": 1, "backoff_sec": 10},
    timeout_policy={"dry_run": 60, "local": 300},
    artifact_contract_id="ac_agent_plan",
    report_contract_id="rc_agent_plan",
))

EC_PARAM_REDUCTION = _register(ExecutionContract(
    contract_id="ec_param_reduction",
    handler_id="param_reduction_sweep",
    skill_id="param_reduction_sweep",
    design_ids=[],
    backend_ids=["phase_to_fft_proxy", "local_synthetic_hsi"],
    execution_modes=["dry_run", "local"],
    required_inputs=["param_sweep_config"],
    required_outputs=["sweep_result.json", "metrics.json"],
    required_metrics=["param_reduction_achieved", "mse_retained"],
    status_values=["succeeded", "needs_followup", "unsupported"],
    evidence_level_mapping={"dry_run": "lightweight_scientific_execution", "local": "lightweight_scientific_execution"},
    claim_ceiling_mapping={"dry_run": "lightweight_scientific_execution", "local": "lightweight_scientific_execution"},
    failure_modes=["param_reduction_degrades_mse"],
    retry_policy={"max_retries": 1, "backoff_sec": 10},
    timeout_policy={"dry_run": 60, "local": 600},
    artifact_contract_id="ac_agent_plan",
    report_contract_id="rc_agent_plan",
))

EC_REPORT_NEGATIVE_RESULT = _register(ExecutionContract(
    contract_id="ec_report_negative_result",
    handler_id="report_negative_result_doc",
    skill_id="report_generation",
    design_ids=["report_geolens_negative_result"],
    backend_ids=[],
    execution_modes=["dry_run", "local"],
    required_inputs=["result_data", "failure_mode"],
    required_outputs=["negative_result_report.md"],
    required_metrics=[],
    status_values=["succeeded", "needs_followup", "unsupported"],
    evidence_level_mapping={"dry_run": "report_only", "local": "report_only"},
    claim_ceiling_mapping={"dry_run": "report_only", "local": "report_only"},
    failure_modes=["insufficient_data_for_report"],
    retry_policy={"max_retries": 1, "backoff_sec": 10},
    timeout_policy={"dry_run": 60, "local": 300},
    artifact_contract_id="ac_agent_plan",
    report_contract_id="rc_agent_plan",
))


def get_all_contracts() -> dict[str, ExecutionContract]:
    return dict(_CONTRACTS)


def get_contract(contract_id: str) -> ExecutionContract | None:
    return _CONTRACTS.get(contract_id)
