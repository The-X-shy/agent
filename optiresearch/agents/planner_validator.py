"""Planner Validator — validates LLM-generated research proposals.

Ensures every LLM proposal passes safety, schema, claim ceiling,
and backend compatibility checks before execution.
"""

from __future__ import annotations

from typing import Any, Optional

FORBIDDEN_KEYWORDS = [
    "git", "rm ", "sudo", "pip install", "curl", "wget",
    "ssh ", "scp ", "chmod", "chown", "kill", "reboot",
    "shutdown", "docker", "systemctl",
]

FORBIDDEN_SHELL_PATTERNS = [";", "&&", "||", "|", "`", "$(", ">"]


def validate_proposal(
    proposal: "LLMPlannerProposal",
    allowed_backends: list[str],
    allowed_task_types: list[str],
    execution_mode: str,
    allow_remote: bool = False,
) -> dict[str, Any]:
    """Run all validation checks on a proposal.

    Returns a dict with 'valid' (bool) and 'errors' (list of error messages).
    """
    errors: list[str] = []

    errors.extend(validate_schema(proposal))
    errors.extend(validate_backend_exists(proposal.backend_id, allowed_backends))
    errors.extend(validate_task_type_supported(proposal.task_type, allowed_task_types))
    errors.extend(validate_claim_ceiling(proposal.backend_id, proposal.task_type))
    errors.extend(validate_execution_mode(proposal, execution_mode, allow_remote))
    errors.extend(validate_no_shell_commands(proposal.proposed_claim))
    errors.extend(validate_no_shell_commands(proposal.rationale))
    errors.extend(validate_no_shell_commands(proposal.hypothesis))
    errors.extend(validate_no_forbidden_actions(proposal.proposed_claim))
    errors.extend(validate_no_forbidden_actions(proposal.rationale))
    errors.extend(validate_dataset_claim(proposal.proposed_claim, proposal.backend_id))
    errors.extend(validate_waveoptics_claim(proposal.proposed_claim, proposal.backend_id))

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "proposal_id": proposal.proposal_id,
    }


def validate_schema(proposal: "LLMPlannerProposal") -> list[str]:
    errors: list[str] = []
    if not proposal.proposal_id:
        errors.append("proposal_id is empty")
    if not proposal.recommended_action:
        errors.append("recommended_action is empty")
    if proposal.risk_level not in ("low", "medium", "high"):
        errors.append(f"invalid risk_level: {proposal.risk_level}")
    return errors


def validate_backend_exists(backend_id: str, allowed_backends: list[str]) -> list[str]:
    if not backend_id:
        return ["backend_id is empty"]
    if backend_id not in allowed_backends:
        return [f"backend '{backend_id}' not in allowed_backends: {allowed_backends}"]
    return []


def validate_task_type_supported(task_type: str, allowed_task_types: list[str]) -> list[str]:
    if not task_type:
        return ["task_type is empty"]
    if task_type not in allowed_task_types:
        return [f"task_type '{task_type}' not in allowed_task_types: {allowed_task_types}"]
    return []


def validate_claim_ceiling(backend_id: str, task_type: str) -> list[str]:
    try:
        from optiresearch.backends.registry import get_backend
        backend = get_backend(backend_id)
        if backend is None:
            return [f"Unknown backend: {backend_id}"]
        if backend.claim_ceiling == "unsupported":
            return [f"Backend {backend_id} has claim_ceiling=unsupported"]
    except Exception as e:
        return [f"Claim ceiling check failed: {e}"]
    return []


def validate_execution_mode(
    proposal: "LLMPlannerProposal",
    execution_mode: str,
    allow_remote: bool,
) -> list[str]:
    errors: list[str] = []
    if execution_mode == "remote_opt_in" and not allow_remote:
        errors.append(
            "Remote execution proposed but allow_remote=False"
        )
    return errors


def validate_no_shell_commands(text: str) -> list[str]:
    if not text:
        return []
    errors: list[str] = []
    text_lower = text.lower()
    for pattern in FORBIDDEN_SHELL_PATTERNS:
        if pattern in text:
            errors.append(f"Text contains shell pattern: '{pattern}'")
    return errors


def validate_no_forbidden_actions(text: str) -> list[str]:
    if not text:
        return []
    errors: list[str] = []
    text_lower = text.lower()
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in text_lower:
            errors.append(f"Text contains forbidden action: '{keyword}'")
    return errors


def validate_dataset_claim(claim_text: str, backend_id: str) -> list[str]:
    if not claim_text:
        return []
    claim_lower = claim_text.lower()
    if backend_id in ("local_synthetic_hsi", "mock_deeplens"):
        if "real" in claim_lower or "physical" in claim_lower:
            return [f"Synthetic backend '{backend_id}' cannot claim real/physical HSI performance"]
    return []


def validate_waveoptics_claim(claim_text: str, backend_id: str) -> list[str]:
    if not claim_text:
        return []
    claim_lower = claim_text.lower()
    if backend_id in ("deeplens_geolens_geometric", "phase_to_fft_proxy"):
        if "full wave" in claim_lower or "coherent" in claim_lower:
            return [f"Backend '{backend_id}' cannot claim coherent/full wave-optics"]
    return []
