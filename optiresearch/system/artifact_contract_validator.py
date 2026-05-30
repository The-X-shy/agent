"""Validate artifact contracts against run directories (Phase 69 — upgraded)."""
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
    sha256_missing: list[str] = []
    evidence_role_mismatches: list[str] = []

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

    # Check artifact manifest completeness and SHA256
    manifest_path = run_path / "artifact_manifest.json"
    manifest_ok = True
    sha256_verified = 0
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

            # Phase 69: SHA256 verification
            if contract.sha256_required:
                for art in manifest_artifacts:
                    art_name = art.get("artifact_name", art.get("name", ""))
                    art_sha = art.get("sha256", "")
                    if not art_sha:
                        sha256_missing.append(art_name)
                    else:
                        sha256_verified += 1

            # Phase 69: Evidence role validation
            if contract.artifact_roles:
                for art in manifest_artifacts:
                    art_name = art.get("artifact_name", art.get("name", ""))
                    art_role = art.get("evidence_role", "")
                    expected_role = contract.artifact_roles.get(art_name)
                    if expected_role and art_role and art_role != expected_role:
                        evidence_role_mismatches.append(
                            f"{art_name}: expected '{expected_role}', got '{art_role}'"
                        )
        except Exception as exc:
            issues.append(f"Cannot read manifest: {exc}")
            manifest_ok = False
    elif contract.artifactstore_registration_required:
        issues.append("artifact_manifest.json missing but registration required")

    if sha256_missing:
        issues.append(f"SHA256 missing for: {sha256_missing}")

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
        "handler_id": contract.handler_id,
        "run_dir": str(run_path),
        "status": status,
        "required_total": len(contract.required_artifacts),
        "required_present": len(present),
        "required_missing": len(missing),
        "present": present,
        "missing": missing,
        "optional_present": optional_present,
        "manifest_ok": manifest_ok,
        "sha256_verified": sha256_verified,
        "sha256_missing": sha256_missing,
        "evidence_role_mismatches": evidence_role_mismatches,
        "missing_artifact_policy": policy,
        "issues": issues,
    }


def validate_artifact_contracts_against_registry(
    contracts: dict[str, ArtifactContract],
) -> dict[str, Any]:
    """Validate artifact contracts against the handler capability registry."""
    invalid_handler_ids: list[str] = []
    evidence_role_coverage: dict[str, int] = {}
    missing_artifact_policies: dict[str, str] = {}

    # Resolve valid handler IDs
    valid_handler_ids: set[str] = set()
    try:
        from optiresearch.skills.handler_capability_registry import get_handler_capability_registry
        reg = get_handler_capability_registry()
        valid_handler_ids = {h.handler_id for h in reg.list_enabled()}
    except Exception:
        pass

    for cid, c in contracts.items():
        # Skip system-level contracts
        if c.handler_id == "system_capability":
            continue

        # Check handler_id validity
        if valid_handler_ids and c.handler_id and c.handler_id not in valid_handler_ids:
            invalid_handler_ids.append(f"{cid}: handler_id '{c.handler_id}' not in registry")

        # Count evidence roles
        for role in c.artifact_roles.values():
            evidence_role_coverage[role] = evidence_role_coverage.get(role, 0) + 1

        # Track missing artifact policies
        missing_artifact_policies[cid] = c.missing_artifact_policy

    valid = len(contracts) - len(invalid_handler_ids)

    return {
        "artifact_contract_count": len(contracts),
        "valid_artifact_contracts": valid,
        "invalid_handler_ids": invalid_handler_ids,
        "evidence_role_coverage": evidence_role_coverage,
        "missing_artifact_policies": missing_artifact_policies,
        "critical_issues": len(invalid_handler_ids),
    }
