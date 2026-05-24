from optiresearch.remote.worker_registry import (
    RemoteWorkerRegistry,
    validate_remote_worker_requirements,
)
from optiresearch.schemas.remote import RemoteWorkerSpec
from optiresearch.skills.handler_capability_registry import (
    get_handler_capability_registry,
)


def _worker(*, tags: list[str], max_runtime_seconds: int = 3600) -> RemoteWorkerSpec:
    return RemoteWorkerSpec(
        worker_id="windows_wsl",
        host="wslbox",
        username="ysl",
        remote_project_dir="/mnt/d/agent",
        remote_workspace_dir="/mnt/d/agent/workspace",
        python_executable="/mnt/d/agent/run_agent_python.sh",
        max_runtime_seconds=max_runtime_seconds,
        backend_tags=tags,
        capabilities={
            "allowed_commands": ["run-deeplens-native-geolens-hsi-codesign"],
            "artifact_return_path": "/mnt/d/agent/workspace/remote_jobs",
        },
    )


def test_remote_worker_requirements_validation_passes_for_matching_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REMOTE_WORKER_ROOT", str(tmp_path / "remote_workers"))
    RemoteWorkerRegistry().add_worker(
        _worker(tags=["windows_wsl", "deeplens_available", "geolens_psf_geometric"])
    )
    cap = get_handler_capability_registry().get("remote_native_geolens_validation")

    result = validate_remote_worker_requirements(cap, "windows_wsl")

    assert result["worker_id"] == "windows_wsl"
    assert result["requirements_met"] is True
    assert result["missing_requirements"] == []
    assert result["command_allowlisted"] is True
    assert result["allowed_command_valid"] is True
    assert result["artifact_return_path_valid"] is True


def test_remote_worker_requirements_validation_reports_missing_requirements(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIRESEARCH_REMOTE_WORKER_ROOT", str(tmp_path / "remote_workers"))
    RemoteWorkerRegistry().add_worker(_worker(tags=["windows_wsl"]))
    cap = get_handler_capability_registry().get("remote_native_geolens_validation")

    result = validate_remote_worker_requirements(cap, "windows_wsl")

    assert result["requirements_met"] is False
    assert "deeplens_available" in result["missing_requirements"]
    assert "geolens_psf_geometric" in result["missing_requirements"]
    assert result["final_claim_ceiling"] == "needs_followup"


def test_remote_worker_requirements_validation_reports_unknown_worker():
    cap = get_handler_capability_registry().get("remote_native_geolens_validation")

    result = validate_remote_worker_requirements(cap, "missing_worker")

    assert result["requirements_met"] is False
    assert result["missing_requirements"] == ["worker_exists"]
    assert result["stop_reason"] == "remote_worker_requirements_not_met"
