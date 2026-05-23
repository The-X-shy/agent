from optiresearch.memory.claim_gate_v2 import ClaimGateV2


def test_report_only_supports_negative_result_documentation_only():
    gate = ClaimGateV2()

    supported = gate.check_claim(
        "The negative result is documented",
        backend_id="",
        experiment_result={
            "status": "completed",
            "task_type": "report_generation",
            "evidence_level": "report_only",
        },
    )
    overreach = gate.check_claim(
        "Optical improvement was achieved",
        backend_id="",
        experiment_result={
            "status": "completed",
            "task_type": "report_generation",
            "evidence_level": "report_only",
        },
    )

    assert supported.decision == "supported"
    assert supported.max_allowed_claim == "report_only"
    assert overreach.decision == "unsupported"
    assert overreach.violation_type == "report_only_as_improvement"


def test_structured_unsupported_supports_boundary_not_task_success():
    gate = ClaimGateV2()

    boundary = gate.check_claim(
        "Boundary detected for local DiffractiveLens execution",
        backend_id="deeplens_geolens_geometric",
        experiment_result={
            "status": "unsupported",
            "evidence_level": "structured_unsupported",
        },
    )
    success = gate.check_claim(
        "The local DiffractiveLens execution succeeded",
        backend_id="deeplens_geolens_geometric",
        experiment_result={
            "status": "unsupported",
            "evidence_level": "structured_unsupported",
        },
    )

    assert boundary.decision == "supported"
    assert boundary.max_allowed_claim == "structured_unsupported"
    assert success.decision == "unsupported"
    assert success.violation_type == "structured_unsupported_as_success"


def test_local_execution_completed_uses_backend_claim_ceiling():
    gate = ClaimGateV2()

    decision = gate.check_claim(
        "Local native lens simulation completed",
        backend_id="deeplens_geolens_geometric",
        experiment_result={
            "status": "completed",
            "evidence_level": "local_execution_completed",
            "metrics": {"accepted_update_count": 1},
        },
        evidence_scope={"execution_target": "local"},
    )

    assert decision.decision == "supported"
    assert decision.max_allowed_claim == "native_lens_simulation"
