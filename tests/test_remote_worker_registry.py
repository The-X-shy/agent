from optiresearch.remote.worker_registry import RemoteWorkerRegistry
from optiresearch.schemas.remote import RemoteWorkerSpec


def test_registry_adds_lists_and_loads_worker(tmp_path):
    registry = RemoteWorkerRegistry(root=tmp_path / "remote_workers")
    worker = RemoteWorkerSpec(
        worker_id="windows_wsl",
        host="wslbox",
        username="ysl",
        remote_project_dir="/mnt/d/agent",
        remote_workspace_dir="/mnt/d/agent/workspace",
        python_executable="/mnt/d/agent/run_agent_python.sh",
        backend_tags=["wsl", "deeplens", "torch", "remote"],
        capabilities={"gpu": False},
    )

    registry.add_worker(worker)

    loaded = registry.get_worker("windows_wsl")
    assert loaded == worker
    assert registry.config_path.exists()
    assert [w.worker_id for w in registry.list_workers()] == ["windows_wsl"]
