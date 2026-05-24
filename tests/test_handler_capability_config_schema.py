"""Test handler capability config schema validation."""

from optiresearch.skills.handler_capability_schema import (
    validate_handler_capability_config,
    KNOWN_EVIDENCE_LEVELS,
    KNOWN_RISK_LEVELS,
)


def test_valid_config_passes():
    data = {
        "capability_schema_version": "0.1",
        "handlers": [
            {
                "handler_id": "test_handler",
                "display_name": "Test Handler",
                "design_type": "scientific",
                "actual_evidence_level": "lightweight_scientific_execution",
                "max_claim_ceiling": "lightweight_scientific_execution",
                "synthetic_only": True,
                "native_backend_required": False,
                "physical_backend": False,
                "real_data_required": False,
                "supported_execution_modes": ["local"],
                "metrics_supported": ["mse_after"],
                "compatible_design_ids": ["test_design"],
                "risk_level": "low",
                "enabled": True,
            }
        ],
    }
    errors = validate_handler_capability_config(data)
    assert errors == []


def test_missing_version():
    data = {"handlers": []}
    errors = validate_handler_capability_config(data)
    assert any("version" in e.lower() for e in errors)


def test_duplicate_handler_id():
    data = {
        "capability_schema_version": "0.1",
        "handlers": [
            {
                "handler_id": "dup",
                "display_name": "Dup",
                "design_type": "scientific",
                "actual_evidence_level": "lightweight_scientific_execution",
                "max_claim_ceiling": "lightweight_scientific_execution",
                "synthetic_only": False,
                "native_backend_required": False,
                "physical_backend": False,
                "real_data_required": False,
                "supported_execution_modes": ["local"],
                "metrics_supported": [],
                "compatible_design_ids": [],
                "risk_level": "low",
                "enabled": True,
            },
            {
                "handler_id": "dup",
                "display_name": "Dup2",
                "design_type": "probe",
                "actual_evidence_level": "report_only",
                "max_claim_ceiling": "report_only",
                "synthetic_only": False,
                "native_backend_required": False,
                "physical_backend": False,
                "real_data_required": False,
                "supported_execution_modes": [],
                "metrics_supported": [],
                "compatible_design_ids": [],
                "risk_level": "low",
                "enabled": True,
            },
        ],
    }
    errors = validate_handler_capability_config(data)
    assert any("duplicate" in e.lower() for e in errors)


def test_unknown_evidence_level():
    data = {
        "capability_schema_version": "0.1",
        "handlers": [
            {
                "handler_id": "bad_ev",
                "display_name": "Bad Evidence",
                "design_type": "scientific",
                "actual_evidence_level": "super_native_mega_simulation",
                "max_claim_ceiling": "super_native_mega_simulation",
                "synthetic_only": False,
                "native_backend_required": False,
                "physical_backend": False,
                "real_data_required": False,
                "supported_execution_modes": [],
                "metrics_supported": [],
                "compatible_design_ids": [],
                "risk_level": "low",
                "enabled": False,
            },
        ],
    }
    errors = validate_handler_capability_config(data)
    assert any("evidence" in e.lower() for e in errors)


def test_known_evidence_levels_includes_phase42_levels():
    assert "lightweight_scientific_execution" in KNOWN_EVIDENCE_LEVELS
    assert "structured_unsupported" in KNOWN_EVIDENCE_LEVELS
    assert "report_only" in KNOWN_EVIDENCE_LEVELS
    assert "native_lens_simulation" in KNOWN_EVIDENCE_LEVELS
