"""Validate artifact contracts against run directories (Phase 68)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from optiresearch.schemas.artifact_contract import ArtifactContract


def validate_artifact_contract_for_run(
    run_dir: str | Path,
    contract: ArtifactContract,
) -> dict[str, Any]:
    """Check that a run directory satisfies an artifact contract."""
    run_path = Path(run_dir)
    issues: list[str] = []
    present: list[str] = []
    missing: list[str] = []

    # Check required artifacts exist
    for artifact_name in contract.required_artifacts:
        artifact_path = run_path / artifact_name
        if artifact_path.exists():
            present.append(artifact_name)
        else:
            missing.append(artifact_name)
            issues.append(f"Missing required artifact: {artifact_name}")

    # Check optional artifacts
    optional_present: list[str] = []
    for artifact_name in contract.optional_artifacts:
        if (run_path / artifact_name).exists():
            optional_present.append(artifact_name)

    # Check artifact manifest completeness if present
    manifest_path = run_path / "artifact_manifest.json"
    manifest_ok = True
    if manifest_path.exists():
        try:
            import json
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_artifacts = manifest.get("artifacts", [])
            manifest_names = {a.get("artifact_name", a.get("name", "")) for a in manifest_artifacts}
            for required in contract.required_artifacts:
                if required not in manifest_names:
                    issues.append(f"Required artifact '{required}' not in manifest")
                    manifest_ok = False
        except Exception as exc:
            issues.append(f"Cannot read manifest: {exc}")
            manifest_ok = False
    elif contract.artifactstore_registration_required:
        issues.append("artifact_manifest.json missing but registration required")

    # Evaluate policy
    policy = contract.missing_artifact_policy
    status = "passed"
    if missing:
        if policy == "needs_followup":
            status = "needs_followup"
        elif policy == "partial_evidence":
            status = "partial_evidence"
        elif policy == "structured_warning":
            status = "structured_warning"

    return {
        "contract_id": contract.contract_id,
        "run_dir": str(run_path),
        "status": status,
        "required_total": len(contract.required_artifacts),
        "required_present": len(present),
        "required_missing": len(missing),
        "present": present,
        "missing": missing,
        "optional_present": optional_present,
        "manifest_ok": manifest_ok,
        "missing_artifact_policy": policy,
        "issues": issues,
    }
