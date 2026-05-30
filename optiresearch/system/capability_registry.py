"""Build SystemCapabilityRegistry from all existing configs (Phase 68)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from optiresearch.schemas.system_capability import SystemCapabilityEntry, SystemCapabilityRegistry


def build_system_capability_registry() -> SystemCapabilityRegistry:
    """Collect capabilities from all existing registries into a unified view."""
    entries: list[SystemCapabilityEntry] = []
    source_files: list[str] = []

    entries.extend(_collect_handler_entries(source_files))
    entries.extend(_collect_skill_entries(source_files))
    entries.extend(_collect_design_strategy_entries(source_files))
    entries.extend(_collect_backend_entries(source_files))
    entries.extend(_collect_claim_policy_entries(source_files))

    validation_summary = _build_validation_summary(entries)

    return SystemCapabilityRegistry.create(
        entries=entries,
        source_files=source_files,
        validation_summary=validation_summary,
    )


def _collect_handler_entries(source_files: list[str]) -> list[SystemCapabilityEntry]:
    """Collect entries from HandlerCapabilityRegistry."""
    from optiresearch.skills.handler_capability_registry import get_handler_capability_registry

    registry = get_handler_capability_registry()
    entries: list[SystemCapabilityEntry] = []
    source_files.append("optiresearch/config/handler_capabilities.yaml")

    for cap in registry.list_enabled():
        entry = SystemCapabilityEntry(
            capability_id=cap.handler_id,
            capability_type="handler",
            name=getattr(cap, "display_name", cap.handler_id),
            enabled=cap.enabled,
            maturity_level=_infer_maturity(cap.actual_evidence_level, cap.supports_remote),
            supported_execution_modes=list(cap.supported_execution_modes),
            evidence_level=cap.actual_evidence_level,
            max_claim_ceiling=cap.max_claim_ceiling,
            synthetic_only=cap.synthetic_only,
            native_backend_required=cap.native_backend_required,
            physical_backend=cap.physical_backend,
            real_data_required=cap.real_data_required,
            supports_remote=cap.supports_remote,
            requires_remote=cap.remote_required,
            requires_deeplens=cap.native_backend_required,
            requires_wsl=cap.supports_remote,
            known_limitations=list(cap.known_limitations),
            owner_module=f"optiresearch.skills.handler_capability_registry::{cap.handler_id}",
        )
        entries.append(entry)

    return entries


def _collect_skill_entries(source_files: list[str]) -> list[SystemCapabilityEntry]:
    """Collect entries from SkillRegistryV2."""
    from optiresearch.skills.registry_v2 import SkillRegistryV2

    registry = SkillRegistryV2()
    entries: list[SystemCapabilityEntry] = []
    source_files.append("optiresearch/skills/registry_v2.py")

    for skill in registry.list_skills():
        evidence_level = skill.evidence_level or "unsupported"
        entry = SystemCapabilityEntry(
            capability_id=skill.skill_id,
            capability_type="skill",
            name=skill.name or skill.skill_id,
            enabled=True,
            maturity_level=_infer_maturity(evidence_level, "remote" in skill.allowed_execution_targets),
            supported_execution_modes=list(skill.allowed_execution_targets),
            evidence_level=evidence_level,
            max_claim_ceiling=evidence_level,
            synthetic_only=True,
            native_backend_required=bool(skill.required_backends),
            supports_remote="remote" in skill.allowed_execution_targets,
            requires_remote=False,
            requires_deeplens=any("deeplens" in b for b in skill.required_backends),
            known_limitations=[],
            owner_module="optiresearch.skills.registry_v2",
        )
        entries.append(entry)

    return entries


def _collect_design_strategy_entries(source_files: list[str]) -> list[SystemCapabilityEntry]:
    """Collect entries from DeepLensDesignStrategyRegistry."""
    from optiresearch.optics.deeplens_design_strategy_registry import get_deeplens_design_strategy_registry

    registry = get_deeplens_design_strategy_registry()
    entries: list[SystemCapabilityEntry] = []
    source_files.append("optiresearch/optics/deeplens_design_strategy_registry.py")

    for strategy in registry.list_all():
        entry = SystemCapabilityEntry(
            capability_id=strategy.strategy_id,
            capability_type="design",
            name=strategy.name or strategy.strategy_id,
            enabled=strategy.enabled,
            maturity_level=_infer_maturity(strategy.evidence_level, strategy.execution_target == "remote_opt_in"),
            supported_execution_modes=[strategy.execution_target] if strategy.execution_target else [],
            evidence_level=strategy.evidence_level,
            max_claim_ceiling=strategy.claim_ceiling,
            synthetic_only=True,
            native_backend_required=bool(strategy.required_backend),
            supports_remote=strategy.execution_target == "remote_opt_in",
            requires_deeplens="deeplens" in strategy.required_backend,
            known_limitations=list(strategy.caveats),
            owner_module="optiresearch.optics.deeplens_design_strategy_registry",
        )
        entries.append(entry)

    return entries


def _collect_backend_entries(source_files: list[str]) -> list[SystemCapabilityEntry]:
    """Collect entries from the optical backend registry."""
    from optiresearch.backends.registry import list_backends

    entries: list[SystemCapabilityEntry] = []
    source_files.append("optiresearch/backends/registry.py")

    for backend in list_backends():
        entry = SystemCapabilityEntry(
            capability_id=backend.backend_id,
            capability_type="backend",
            name=backend.label,
            enabled=True,
            maturity_level=_backend_maturity(backend.backend_id),
            supported_execution_modes=["local"] + (["remote_opt_in"] if backend.supports_remote_execution else []),
            evidence_level=backend.claim_ceiling,
            max_claim_ceiling=backend.claim_ceiling,
            synthetic_only=backend.backend_type in ("mock", "proxy", "synthetic"),
            native_backend_required=backend.backend_type == "deeplens",
            physical_backend=backend.supports_real_hardware,
            supports_remote=backend.supports_remote_execution,
            requires_deeplens=backend.backend_type == "deeplens",
            known_limitations=list(backend.known_failure_modes),
            owner_module="optiresearch.backends.registry",
        )
        entries.append(entry)

    return entries


def _collect_claim_policy_entries(source_files: list[str]) -> list[SystemCapabilityEntry]:
    """Collect claim policy entries from ClaimGateV2 evidence ladder."""
    from optiresearch.memory.claim_gate_v2 import _evidence_rank

    entries: list[SystemCapabilityEntry] = []
    source_files.append("optiresearch/memory/claim_gate_v2.py")

    known_levels = [
        "unsupported", "report_only", "negative_result", "mock_simulation",
        "deeplens_integration_smoke", "native_component_optimization",
        "component_surrogate_hsi_codesign", "native_hsi_proxy",
        "native_full_reconstruction_proxy", "lightweight_scientific_execution",
        "sweep_analysis", "native_lens_simulation", "native_waveoptics_simulation",
        "stable_native_lens_hsi_codesign", "rollback_protected_native_lens_hsi",
        "real_hsi_performance",
    ]

    for level in known_levels:
        rank = _evidence_rank(level)
        entry = SystemCapabilityEntry(
            capability_id=f"claim_policy_{level}",
            capability_type="claim_policy",
            name=f"Evidence Level: {level}",
            enabled=rank >= 0,
            maturity_level="production_ready" if rank >= 8 else ("validated_local" if rank >= 4 else "experimental"),
            supported_execution_modes=["dry_run"],
            evidence_level=level,
            max_claim_ceiling=level,
            synthetic_only=rank < 12,
            real_data_required=rank >= 12,
            native_backend_required=rank >= 4,
            known_limitations=[],
            owner_module="optiresearch.memory.claim_gate_v2",
        )
        entries.append(entry)

    return entries


def _build_validation_summary(entries: list[SystemCapabilityEntry]) -> dict[str, Any]:
    """Analyze entries for missing fields, orphans, and inconsistencies."""
    missing_evidence_level = sum(1 for e in entries if not e.evidence_level or e.evidence_level == "unsupported")
    missing_claim_ceiling = sum(1 for e in entries if not e.max_claim_ceiling or e.max_claim_ceiling == "unsupported")
    missing_owner_module = sum(1 for e in entries if not e.owner_module)

    # Detect orphan references
    handler_ids = {e.capability_id for e in entries if e.capability_type == "handler"}
    skill_ids = {e.capability_id for e in entries if e.capability_type == "skill"}
    design_ids = {e.capability_id for e in entries if e.capability_type == "design"}
    backend_ids = {e.capability_id for e in entries if e.capability_type == "backend"}

    # Check for handlers with evidence_level > max_claim_ceiling inconsistency
    from optiresearch.memory.claim_gate_v2 import _evidence_rank
    inconsistent_ceilings = 0
    for e in entries:
        if e.capability_type == "handler":
            ev_rank = _evidence_rank(e.evidence_level)
            ceil_rank = _evidence_rank(e.max_claim_ceiling)
            if ev_rank > ceil_rank > 0:
                inconsistent_ceilings += 1

    by_type = {}
    for e in entries:
        by_type.setdefault(e.capability_type, 0)
        by_type[e.capability_type] += 1

    return {
        "total_entries": len(entries),
        "by_type": by_type,
        "handler_ids": sorted(handler_ids),
        "skill_ids": sorted(skill_ids),
        "design_ids": sorted(design_ids),
        "backend_ids": sorted(backend_ids),
        "missing_evidence_level": missing_evidence_level,
        "missing_claim_ceiling": missing_claim_ceiling,
        "missing_owner_module": missing_owner_module,
        "inconsistent_ceilings": inconsistent_ceilings,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _infer_maturity(evidence_level: str, supports_remote: bool) -> str:
    from optiresearch.memory.claim_gate_v2 import _evidence_rank
    rank = _evidence_rank(evidence_level)
    if rank >= 10:
        return "benchmarked"
    if rank >= 7:
        return "validated_remote" if supports_remote else "validated_local"
    if rank >= 4:
        return "validated_local"
    return "experimental"


def _backend_maturity(backend_id: str) -> str:
    validated = {"deeplens_geolens_geometric", "deeplens_fresnel_component", "deeplens_binary2phase_component"}
    benchmarked = {"deeplens_geolens_geometric"}
    if backend_id in benchmarked:
        return "benchmarked"
    if backend_id in validated:
        return "validated_local"
    return "experimental"
