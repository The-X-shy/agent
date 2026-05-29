"""Schema validation for handler capability config (Phase 42)."""

from __future__ import annotations

from typing import Any

KNOWN_EVIDENCE_LEVELS = {
    "unsupported", "structured_unsupported", "needs_followup",
    "requires_user_data", "report_only", "negative_result",
    "mock_simulation", "deeplens_integration_smoke",
    "native_component_optimization", "native_hsi_proxy",
    "native_full_reconstruction_proxy", "lightweight_scientific_execution",
    "component_surrogate_hsi_codesign",
    "synthetic_lightweight_metric_experiment", "synthetic_hsi_simulation",
    "sweep_analysis", "native_lens_simulation",
    "native_waveoptics_simulation", "native_waveoptics",
    "stable_native_lens_hsi_codesign",
    "rollback_protected_native_lens_hsi",
    "real_hsi_performance", "real_hsi", "real_hsi_validation",
    "diagnostic_evidence",
}

KNOWN_RISK_LEVELS = {"low", "medium", "high"}
KNOWN_EXECUTION_MODES = {"dry_run", "local", "remote_opt_in"}
KNOWN_DESIGN_TYPES = {"scientific", "probe", "report", "data_request", "remote_validation"}

REQUIRED_FIELDS = [
    "handler_id", "display_name", "design_type",
    "actual_evidence_level", "max_claim_ceiling",
    "synthetic_only", "native_backend_required",
    "physical_backend", "real_data_required",
    "supported_execution_modes", "metrics_supported",
    "compatible_design_ids", "risk_level", "enabled",
]

OPTIONAL_FIELDS = [
    "task_type", "remote_required", "supports_remote",
    "requires_remote_validation", "remote_evidence_ceiling",
    "local_evidence_ceiling", "artifacts_supported",
    "known_limitations", "compatible_backend_ids",
    "remote_worker_requirements", "default_timeout_sec",
]


def validate_handler_capability_config(data: dict[str, Any]) -> list[str]:
    """Validate the entire handler capabilities config dict.

    Returns a list of error messages. Empty list = valid.
    """
    errors: list[str] = []

    # Top-level version
    version = data.get("capability_schema_version", "")
    if not version:
        errors.append("Missing capability_schema_version")

    handlers = data.get("handlers")
    if not isinstance(handlers, list):
        errors.append("'handlers' must be a list")
        return errors

    if not handlers:
        errors.append("'handlers' list is empty")
        return errors

    seen_ids: set[str] = set()

    for i, h in enumerate(handlers):
        if not isinstance(h, dict):
            errors.append(f"Handler[{i}] is not a dict")
            continue

        handler_id = h.get("handler_id", f"<index {i}>")
        prefix = f"Handler '{handler_id}'"

        # Required fields
        for field in REQUIRED_FIELDS:
            if field not in h:
                errors.append(f"{prefix}: missing required field '{field}'")

        # Unique handler_id
        hid = h.get("handler_id", "")
        if hid and hid in seen_ids:
            errors.append(f"{prefix}: duplicate handler_id")
        if hid:
            seen_ids.add(hid)

        # Evidence level
        ev = h.get("actual_evidence_level", "")
        if ev and ev not in KNOWN_EVIDENCE_LEVELS:
            errors.append(f"{prefix}: unknown evidence_level '{ev}'")

        ceiling = h.get("max_claim_ceiling", "")
        if ceiling and ceiling not in KNOWN_EVIDENCE_LEVELS:
            errors.append(f"{prefix}: unknown max_claim_ceiling '{ceiling}'")

        # Risk level
        risk = h.get("risk_level", "")
        if risk and risk not in KNOWN_RISK_LEVELS:
            errors.append(f"{prefix}: unknown risk_level '{risk}'")

        # Design type
        dtype = h.get("design_type", "")
        if dtype and dtype not in KNOWN_DESIGN_TYPES:
            errors.append(f"{prefix}: unknown design_type '{dtype}'")

        # Execution modes
        modes = h.get("supported_execution_modes", [])
        if isinstance(modes, list):
            for mode in modes:
                if mode not in KNOWN_EXECUTION_MODES:
                    errors.append(f"{prefix}: unknown execution mode '{mode}'")

        # enabled must be bool
        enabled = h.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            errors.append(f"{prefix}: 'enabled' must be boolean")

        # compatible_design_ids must be list
        design_ids = h.get("compatible_design_ids")
        if design_ids is not None and not isinstance(design_ids, list):
            errors.append(f"{prefix}: 'compatible_design_ids' must be a list")

        # metrics_supported must be list
        metrics = h.get("metrics_supported")
        if metrics is not None and not isinstance(metrics, list):
            errors.append(f"{prefix}: 'metrics_supported' must be a list")

        requirements = h.get("remote_worker_requirements")
        if requirements is not None and not isinstance(requirements, list):
            errors.append(f"{prefix}: 'remote_worker_requirements' must be a list")

        # Backward compat: remote-only checks
        if h.get("remote_required") and not h.get("supports_remote"):
            errors.append(f"{prefix}: remote_required=true but supports_remote=false")
        if h.get("requires_remote_validation") and not h.get("supports_remote"):
            errors.append(f"{prefix}: requires_remote_validation=true but supports_remote=false")

    return errors
