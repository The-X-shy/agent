import os

import pytest

from optiresearch.remote.worker_registry import RemoteWorkerRegistry
from optiresearch.runtime.remote_jobs import run_remote_deeplens_source_smoke


pytestmark = pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_REMOTE_TESTS") != "1",
    reason="real WSL worker tests are opt-in",
)


def test_real_remote_wsl_worker_smoke():
    worker_id = os.getenv("OPTIRESEARCH_REMOTE_WORKER_ID", "windows_wsl")
    worker = RemoteWorkerRegistry().get_worker(worker_id)

    result = run_remote_deeplens_source_smoke(worker.worker_id, ingest=True)["result"]

    assert result.status in {"succeeded", "failed"}
    assert result.error_code is None or result.error_code.startswith("REMOTE_") or "DEEPLENS" in result.error_code
