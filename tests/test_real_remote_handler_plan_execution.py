import json
import os
from pathlib import Path

import pytest

from optiresearch.runtime.agent_plan_execution_loop import run_agent_plan_execution
from optiresearch.schemas.agent_plan_execution import AgentPlanExecutionSpec


pytestmark = pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_REMOTE_TESTS") != "1"
    or os.getenv("OPTIRESEARCH_REMOTE_WORKER_ID") != "windows_wsl",
    reason="real remote handler plan execution is opt-in",
)


def test_real_remote_handler_plan_execution_opt_in(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    worker_id = os.environ["OPTIRESEARCH_REMOTE_WORKER_ID"]

    result = run_agent_plan_execution(
        AgentPlanExecutionSpec(
            execution_id="real_remote_handler_plan_execution",
            objective="validate native GeoLens HSI path on WSL through remote-aware handler",
            mode="remote_opt_in",
            allow_remote=True,
            remote_worker_id=worker_id,
            execute_top_k=1,
        )
    )

    assert result.selected_design == "remote_native_geolens_validation"
    ex = result.execution_result
    assert ex["execution_target"] == "remote_wsl"
    assert ex["remote_worker_id"] == "windows_wsl"
    assert ex["remote_job_id"]

    if ex.get("remote_validation_passed") is True:
        assert result.claim_gate_decision["final_claim_ceiling"] == "native_lens_simulation"
        assert ex["artifact_return_path"]
        assert Path(ex["artifact_return_path"]).exists()
    else:
        assert ex["status"] in {"failed", "stopped"}
        assert ex.get("errors")
        assert result.claim_gate_decision.get("final_claim_ceiling") in {
            "needs_followup",
            "structured_unsupported",
            "unsupported",
            None,
        }

    serialized = json.dumps(result.model_dump(mode="json"), ensure_ascii=False, default=str)
    assert "sk-" not in serialized
    assert "OPENAI_API_KEY" not in serialized
