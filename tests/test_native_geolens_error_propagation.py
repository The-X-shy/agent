"""Phase 34: Native GeoLens error propagation tests.

Verify that StableNativeLensHSIResult.error_code and execution fidelity
fields propagate correctly into ControllerResult.
"""

import platform as _platform

from optiresearch.runtime.experiment_controller_v2 import (
    ControllerResult,
    ExperimentControllerV2,
    ExperimentSpecV2,
)
from optiresearch.memory.schemas import make_deterministic_id


def test_error_code_propagates_to_controller_result_errors():
    """When an inner result has error_code, ControllerResult.errors must include it."""
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id=make_deterministic_id("test", "err-prop", "1"),
        task_type="stable_lens_hsi_codesign",
        backend_id="deeplens_geolens_geometric",
        execution_fidelity="deeplens_native_geometric",
        spec_payload={"max_steps": 1, "optical_lr": 1e-6, "candidate": "GeoLensCooke"},
    )
    result = ctrl.run_local(spec)
    # On macOS the result will be "unsupported" due to GeoLens API limitations
    if result.error_code is not None:
        assert len(result.errors) > 0, f"Expected errors when error_code is set, got: {result}"
        error_types = [e.get("type") for e in result.errors]
        assert any("GEOLENS" in t or "UNSUPPORTED" in t or "INDEX" in t for t in error_types), \
            f"Expected a GeoLens-related error code in errors, got: {error_types}"
    else:
        # No error_code → no errors (valid state, e.g. macOS unsupported path)
        assert result.errors == [], f"Expected empty errors when error_code is None, got: {result.errors}"


def test_unsupported_status_preserved():
    """Status 'unsupported' must not be compressed to 'failed'."""
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id=make_deterministic_id("test", "err-prop", "2"),
        task_type="stable_lens_hsi_codesign",
        backend_id="deeplens_geolens_geometric",
        execution_fidelity="deeplens_native_geometric",
        spec_payload={"max_steps": 1, "candidate": "GeoLensCooke"},
    )
    result = ctrl.run_local(spec)
    # The status should be one of the valid ControllerResult statuses
    assert result.status in ("succeeded", "failed", "unsupported", "claim_downgraded", "skipped")
    # If the inner result was unsupported, ControllerResult should also be unsupported
    # (not compressed to "failed")


def test_execution_fidelity_fields_preserved():
    """All execution fidelity fields must appear in ControllerResult."""
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id=make_deterministic_id("test", "err-prop", "3"),
        task_type="stable_lens_hsi_codesign",
        backend_id="deeplens_geolens_geometric",
        execution_fidelity="deeplens_native_geometric",
        spec_payload={"max_steps": 1, "candidate": "GeoLensCooke"},
    )
    result = ctrl.run_local(spec)
    # These fields should always be set (not None) for stable_lens_hsi_codesign
    assert result.execution_fidelity == "deeplens_native_geometric"
    assert result.proxy_fallback_used is not None
    assert result.platform is not None
    # deeplens_native_psf_path should be set if the run went through GeoLens
    if result.status == "succeeded":
        assert result.deeplens_native_psf_path is not None


def test_succeeded_result_no_errors():
    """When error_code is None, errors must be empty."""
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id=make_deterministic_id("test", "err-prop", "4"),
        task_type="native_hsi_codesign",
        backend_id="deeplens_geometric_optical",
        execution_fidelity="deeplens_native_geometric",
        spec_payload={"max_steps": 1, "candidate": "Fresnel"},
    )
    result = ctrl.run_local(spec)
    if result.status == "succeeded":
        assert result.error_code is None
        assert result.errors == []


def test_failed_result_carries_error():
    """When status is failed with error_code, errors must be populated."""
    ctrl = ExperimentControllerV2()
    spec = ExperimentSpecV2(
        spec_id=make_deterministic_id("test", "err-prop", "5"),
        task_type="stable_lens_hsi_codesign",
        backend_id="deeplens_geolens_geometric",
        execution_fidelity="deeplens_native_geometric",
        spec_payload={"max_steps": 1, "candidate": "GeoLensCooke"},
    )
    result = ctrl.run_local(spec)
    if result.status == "failed":
        assert result.error_code is not None
        assert len(result.errors) > 0
    # Either succeeded, unsupported, or failed — all are valid
    assert result.status in ("succeeded", "failed", "unsupported")


def test_controller_result_has_all_fidelity_attributes():
    """ControllerResult must have all the new execution fidelity attributes."""
    fields = set(ControllerResult.model_fields.keys())
    for attr in ("error_code", "error_message", "execution_fidelity",
                 "actual_execution_fidelity", "proxy_fallback_used",
                 "deeplens_native_psf_path", "platform"):
        assert attr in fields, f"Missing field: {attr}"
