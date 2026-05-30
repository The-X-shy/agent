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
    command_name="run-remote-deeplens-component-first-probe",
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

_CONTRACTS = {
    c.remote_contract_id: c for c in [
        REC_TRAINABLE_PARAM_INSPECTION, REC_AUTOGRAD_AUDIT,
        REC_COMPONENT_FIRST_PROBE, REC_STABILIZED_NATIVE_GEOLENS_HSI,
        REC_NATIVE_GEOLENS_BENCHMARK, REC_BENCHMARK_FAILURE_ANALYSIS,
        REC_RESUME_BENCHMARK, REC_COMPONENT_SURROGATE_HSI_CODESIGN,
    ]
}


def get_all_remote_contracts() -> dict[str, RemoteExecutionContract]:
    return dict(_CONTRACTS)


def get_remote_contract(contract_id: str) -> RemoteExecutionContract | None:
    return _CONTRACTS.get(contract_id)
