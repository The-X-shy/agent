"""Define 8 core remote execution contracts."""
from __future__ import annotations

from optiresearch.schemas.remote_execution_contract import RemoteExecutionContract


REC_TRAINABLE_PARAM_INSPECTION = RemoteExecutionContract(
    remote_contract_id="rec_trainable_param_inspection",
    command_name="run-remote-deeplens-trainable-parameter-inspection",
    handler_id="deeplens_trainable_parameter_inspection",
    allowed_args=["--lens-file", "--output-dir", "--remote-job-id"],
    forbidden_args=["--allow-adapter-proxy"],
    required_worker_capabilities=["deeplens_available", "windows_wsl"],
    required_env_vars=["DEEPLENS_PATH"],
    propagated_env_vars=["OPTIRESEARCH_DB_PATH", "OPTIRESEARCH_ARTIFACT_ROOT"],
    timeout_sec=1800,
    output_dir_policy="required",
    artifact_return_policy="required",
    allowlist_entry_required=True,
    workspace_write_policy="restricted",
    remote_job_id_required=True,
    result_parser="result.json",
    failure_parser="error_log.txt",
    retry_policy={"max_retries": 2, "backoff_sec": 120},
)

REC_AUTOGRAD_AUDIT = RemoteExecutionContract(
    remote_contract_id="rec_autograd_audit",
    command_name="run-remote-deeplens-autograd-audit",
    handler_id="deeplens_autograd_audit",
    allowed_args=["--lens-file", "--output-dir", "--remote-job-id", "--strict-native"],
    forbidden_args=["--allow-adapter-proxy"],
    required_worker_capabilities=["deeplens_available", "windows_wsl"],
    required_env_vars=["DEEPLENS_PATH"],
    propagated_env_vars=["OPTIRESEARCH_DB_PATH"],
    timeout_sec=900,
    output_dir_policy="required",
    artifact_return_policy="required",
    allowlist_entry_required=True,
    workspace_write_policy="restricted",
    remote_job_id_required=True,
    result_parser="result.json",
    failure_parser="error_log.txt",
    retry_policy={"max_retries": 1, "backoff_sec": 60},
)

REC_COMPONENT_FIRST_PROBE = RemoteExecutionContract(
    remote_contract_id="rec_component_first_probe",
    command_name="run-remote-deeplens-component-probe",
    handler_id="deeplens_component_first_probe",
    allowed_args=["--lens-file", "--component-id", "--output-dir", "--remote-job-id", "--strict-native"],
    forbidden_args=["--allow-adapter-proxy"],
    required_worker_capabilities=["deeplens_available", "windows_wsl"],
    required_env_vars=["DEEPLENS_PATH"],
    propagated_env_vars=["OPTIRESEARCH_DB_PATH"],
    timeout_sec=1800,
    output_dir_policy="required",
    artifact_return_policy="required",
    allowlist_entry_required=True,
    workspace_write_policy="restricted",
    remote_job_id_required=True,
    result_parser="result.json",
    failure_parser="error_log.txt",
    retry_policy={"max_retries": 2, "backoff_sec": 120},
)

REC_STABILIZED_NATIVE_GEOLENS_HSI = RemoteExecutionContract(
    remote_contract_id="rec_stabilized_native_geolens_hsi",
    command_name="run-remote-stabilized-native-geolens-hsi",
    handler_id="stable_native_lens_hsi_codesign",
    allowed_args=["--lens-file", "--dataset", "--max-steps", "--steps",
                  "--spectral-angle-weight", "--grad-clip", "--rollback-patience",
                  "--output-dir", "--remote-job-id", "--strict-native"],
    forbidden_args=["--allow-adapter-proxy"],
    required_worker_capabilities=["deeplens_available", "windows_wsl"],
    required_env_vars=["DEEPLENS_PATH"],
    propagated_env_vars=["OPTIRESEARCH_DB_PATH", "OPTIRESEARCH_ARTIFACT_ROOT"],
    timeout_sec=7200,
    output_dir_policy="required",
    artifact_return_policy="required",
    allowlist_entry_required=True,
    workspace_write_policy="restricted",
    remote_job_id_required=True,
    result_parser="result.json",
    failure_parser="error_log.txt",
    retry_policy={"max_retries": 3, "backoff_sec": 300},
)

REC_NATIVE_GEOLENS_BENCHMARK = RemoteExecutionContract(
    remote_contract_id="rec_native_geolens_benchmark",
    command_name="run-remote-native-geolens-stability-benchmark",
    handler_id="native_geolens_stability_benchmark",
    allowed_args=["--lens-file", "--dataset", "--seeds", "--steps",
                  "--seed-start", "--max-steps", "--output-dir", "--remote-job-id",
                  "--opt-in-write", "--strict-native"],
    forbidden_args=["--allow-adapter-proxy"],
    required_worker_capabilities=["deeplens_available", "windows_wsl"],
    required_env_vars=["DEEPLENS_PATH"],
    propagated_env_vars=["OPTIRESEARCH_DB_PATH", "OPTIRESEARCH_ARTIFACT_ROOT"],
    timeout_sec=28800,
    output_dir_policy="required",
    artifact_return_policy="required",
    allowlist_entry_required=True,
    workspace_write_policy="restricted",
    remote_job_id_required=True,
    result_parser="result.json",
    failure_parser="error_log.txt",
    retry_policy={"max_retries": 1, "backoff_sec": 600},
)

REC_BENCHMARK_FAILURE_ANALYSIS = RemoteExecutionContract(
    remote_contract_id="rec_benchmark_failure_analysis",
    command_name="run-remote-native-geolens-benchmark-failure-analysis",
    handler_id="benchmark_failure_analysis",
    allowed_args=["--benchmark-dir", "--output-dir", "--remote-job-id"],
    forbidden_args=[],
    required_worker_capabilities=["windows_wsl"],
    required_env_vars=[],
    propagated_env_vars=["OPTIRESEARCH_DB_PATH"],
    timeout_sec=1800,
    output_dir_policy="required",
    artifact_return_policy="required",
    allowlist_entry_required=True,
    workspace_write_policy="restricted",
    remote_job_id_required=True,
    result_parser="failure_analysis.json",
    failure_parser="error_log.txt",
    retry_policy={"max_retries": 2, "backoff_sec": 60},
    is_known_gap=True,
    reason="No CLI command or remote_jobs function exists for benchmark failure analysis yet",
)

REC_RESUME_BENCHMARK = RemoteExecutionContract(
    remote_contract_id="rec_resume_benchmark",
    command_name="run-remote-resume-native-geolens-benchmark",
    handler_id="native_geolens_stability_benchmark",
    allowed_args=["--benchmark-dir", "--output-dir", "--remote-job-id",
                  "--opt-in-write", "--strict-native"],
    forbidden_args=["--allow-adapter-proxy"],
    required_worker_capabilities=["deeplens_available", "windows_wsl"],
    required_env_vars=["DEEPLENS_PATH"],
    propagated_env_vars=["OPTIRESEARCH_DB_PATH"],
    timeout_sec=28800,
    output_dir_policy="required",
    artifact_return_policy="required",
    allowlist_entry_required=True,
    workspace_write_policy="restricted",
    remote_job_id_required=True,
    result_parser="result.json",
    failure_parser="error_log.txt",
    retry_policy={"max_retries": 1, "backoff_sec": 600},
    is_known_gap=True,
    reason="No CLI command or remote_jobs function exists for resume benchmark yet",
)

REC_COMPONENT_SURROGATE_HSI_CODESIGN = RemoteExecutionContract(
    remote_contract_id="rec_component_surrogate_hsi_codesign",
    command_name="run-remote-component-surrogate-hsi-codesign",
    handler_id="component_surrogate_hsi_codesign",
    allowed_args=["--lens-file", "--dataset", "--component-id",
                  "--max-steps", "--steps", "--output-dir", "--remote-job-id",
                  "--strict-native"],
    forbidden_args=["--allow-adapter-proxy"],
    required_worker_capabilities=["deeplens_available", "windows_wsl"],
    required_env_vars=["DEEPLENS_PATH"],
    propagated_env_vars=["OPTIRESEARCH_DB_PATH", "OPTIRESEARCH_ARTIFACT_ROOT"],
    timeout_sec=7200,
    output_dir_policy="required",
    artifact_return_policy="required",
    allowlist_entry_required=True,
    workspace_write_policy="restricted",
    remote_job_id_required=True,
    result_parser="result.json",
    failure_parser="error_log.txt",
    retry_policy={"max_retries": 2, "backoff_sec": 300},
)

REC_DEEPLENS_CURRICULUM_PROBE = RemoteExecutionContract(
    remote_contract_id="rec_deeplens_curriculum_probe",
    command_name="run-remote-deeplens-curriculum-probe",
    handler_id="deeplens_curriculum_probe",
    allowed_args=["--lens-file", "--backend-id", "--max-steps", "--device", "--remote-job-id"],
    forbidden_args=["--allow-adapter-proxy"],
    required_worker_capabilities=["deeplens_available", "windows_wsl"],
    required_env_vars=["DEEPLENS_PATH"],
    propagated_env_vars=["OPTIRESEARCH_DB_PATH"],
    timeout_sec=1800,
    output_dir_policy="required",
    artifact_return_policy="required",
    allowlist_entry_required=True,
    workspace_write_policy="restricted",
    remote_job_id_required=True,
    result_parser="result.json",
    failure_parser="error_log.txt",
    retry_policy={"max_retries": 2, "backoff_sec": 120},
)

REC_DEEPLENS_REGULARIZED_PROBE = RemoteExecutionContract(
    remote_contract_id="rec_deeplens_regularized_probe",
    command_name="run-remote-deeplens-regularized-probe",
    handler_id="deeplens_regularized_probe",
    allowed_args=["--lens-file", "--backend-id", "--max-steps", "--device", "--remote-job-id"],
    forbidden_args=["--allow-adapter-proxy"],
    required_worker_capabilities=["deeplens_available", "windows_wsl"],
    required_env_vars=["DEEPLENS_PATH"],
    propagated_env_vars=["OPTIRESEARCH_DB_PATH"],
    timeout_sec=1800,
    output_dir_policy="required",
    artifact_return_policy="required",
    allowlist_entry_required=True,
    workspace_write_policy="restricted",
    remote_job_id_required=True,
    result_parser="result.json",
    failure_parser="error_log.txt",
    retry_policy={"max_retries": 2, "backoff_sec": 120},
)

KNOWN_GAP_CONTRACT_IDS = {"rec_benchmark_failure_analysis", "rec_resume_benchmark"}

_ALL_CONTRACTS = {
    c.remote_contract_id: c for c in [
        REC_TRAINABLE_PARAM_INSPECTION, REC_AUTOGRAD_AUDIT,
        REC_COMPONENT_FIRST_PROBE, REC_STABILIZED_NATIVE_GEOLENS_HSI,
        REC_NATIVE_GEOLENS_BENCHMARK, REC_BENCHMARK_FAILURE_ANALYSIS,
        REC_RESUME_BENCHMARK, REC_COMPONENT_SURROGATE_HSI_CODESIGN,
        REC_DEEPLENS_CURRICULUM_PROBE, REC_DEEPLENS_REGULARIZED_PROBE,
    ]
}

_CONTRACTS = _ALL_CONTRACTS  # for existing code using _CONTRACTS


def get_all_remote_contracts(exclude_known_gaps: bool = True) -> dict[str, RemoteExecutionContract]:
    """Return all contracts, optionally excluding known gaps."""
    if exclude_known_gaps:
        return {k: v for k, v in _ALL_CONTRACTS.items() if k not in KNOWN_GAP_CONTRACT_IDS}
    return dict(_ALL_CONTRACTS)


def get_known_gaps() -> dict[str, RemoteExecutionContract]:
    """Return only known gap contracts."""
    return {k: v for k, v in _ALL_CONTRACTS.items() if k in KNOWN_GAP_CONTRACT_IDS}


def get_remote_contract(contract_id: str) -> RemoteExecutionContract | None:
    return _ALL_CONTRACTS.get(contract_id)

