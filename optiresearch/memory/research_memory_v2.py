"""Research Memory v2 — queryable long-term memory for agentic research.

Encodes Phase 18–23 experience as structured memory entries that the
strategy engine and experiment controller can query at runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

MemoryType = Literal[
    "BackendCapability",
    "FailureMode",
    "OptimizationPolicy",
    "ClaimBoundary",
    "ExperimentOutcome",
    "RemoteExecution",
    "NegativeResult",
]


@dataclass
class ResearchMemoryEntry:
    """A single structured memory record."""

    memory_id: str
    memory_type: MemoryType
    content: str
    tags: list[str] = field(default_factory=list)
    source_run_id: Optional[str] = None
    confidence: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    related_entries: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Phase 18–23 hardcoded rules ─────────────────────────────────────

SEEDED_RULES: list[ResearchMemoryEntry] = [
    ResearchMemoryEntry(
        memory_id="rule_geolens_geometric_not_coherent",
        memory_type="ClaimBoundary",
        content=(
            "GeoLens geometric PSF path is differentiable but is NOT full coherent "
            "wave-optics. The geometric model traces rays, not wavefronts. "
            "Claim ceiling: native_lens_simulation."
        ),
        tags=["geolens", "geometric", "waveoptics", "claim_ceiling"],
        confidence=0.95,
    ),
    ResearchMemoryEntry(
        memory_id="rule_coherent_asm_nograd",
        memory_type="FailureMode",
        content=(
            "Coherent ASM path (GeoLens.psf(model='coherent')) produces "
            "requires_grad=False PSF tensors. Ray sampling in the coherent path "
            "uses no_grad / breaks autograd. Cannot support native_waveoptics claim. "
            "Recommend DiffractiveLens or pure wave propagation probe instead."
        ),
        tags=["coherent", "asm", "waveoptics", "requires_grad", "autograd"],
        confidence=0.90,
    ),
    ResearchMemoryEntry(
        memory_id="rule_phase_to_fft_proxy_ceiling",
        memory_type="ClaimBoundary",
        content=(
            "phase-to-FFT proxy can support native_full_reconstruction_proxy claim "
            "but NOT full wave-optics claim. The FFT of a scalar phase map is a "
            "far-field proxy, not a full wave propagation model."
        ),
        tags=["phase_to_fft", "proxy", "claim_ceiling", "waveoptics"],
        confidence=0.95,
    ),
    ResearchMemoryEntry(
        memory_id="rule_large_gradients_small_lr",
        memory_type="OptimizationPolicy",
        content=(
            "Large GeoLens optical gradients (can exceed 1700 at default LR=1e-3) "
            "require small optical_lr (1e-6 to 1e-5) for stable training. "
            "Gradient clipping alone is insufficient — LR reduction is essential."
        ),
        tags=["geolens", "gradient", "lr", "stability", "training"],
        confidence=0.90,
    ),
    ResearchMemoryEntry(
        memory_id="rule_rollback_protection_not_improvement",
        memory_type="OptimizationPolicy",
        content=(
            "Rollback protects against harmful optical updates but does NOT prove "
            "optical improvement. If accepted_update_count=0 and rollback_count>0, "
            "the system has NOT demonstrated that optical optimization helps HSI. "
            "Do not claim optical improvement when all updates were rejected."
        ),
        tags=["rollback", "stability", "claim", "optimization"],
        confidence=0.95,
    ),
    ResearchMemoryEntry(
        memory_id="rule_synthetic_not_real_hsi",
        memory_type="NegativeResult",
        content=(
            "Synthetic HSI results cannot support real HSI performance claims. "
            "Synthetic data lacks real sensor noise, calibration errors, and "
            "environmental variability. Real HSI dataset is absent — "
            "real HSI performance = unsupported."
        ),
        tags=["hsi", "synthetic", "real", "dataset", "unsupported"],
        confidence=0.95,
    ),
    ResearchMemoryEntry(
        memory_id="rule_remote_validation_required",
        memory_type="RemoteExecution",
        content=(
            "Remote validation is required before declaring execution robustness. "
            "Local results alone are insufficient — environment differences, "
            "stochasticity, and DeepLens version can change outcomes. "
            "Every claim above native_component_optimization needs remote validation."
        ),
        tags=["remote", "validation", "claim", "robustness"],
        confidence=0.90,
    ),
    ResearchMemoryEntry(
        memory_id="rule_optical_warmup_before_joint",
        memory_type="OptimizationPolicy",
        content=(
            "Optical warmup (reconstructor-only training for 3+ steps) before joint "
            "finetune improves stability. This allows the reconstructor to adapt to "
            "the initial PSF before optics start changing."
        ),
        tags=["warmup", "training", "stability", "reconstructor"],
        confidence=0.85,
    ),
    ResearchMemoryEntry(
        memory_id="rule_gradient_clipping_helps",
        memory_type="OptimizationPolicy",
        content=(
            "Gradient clipping (clip_grad_norm_ at 1.0) reduces but does not eliminate "
            "optical gradient instability. Combine with small LR for best results."
        ),
        tags=["gradient", "clipping", "stability", "training"],
        confidence=0.85,
    ),
]


class ResearchMemoryV2:
    """Queryable research memory for the agentic differentiable optics framework."""

    def __init__(self):
        self._entries: dict[str, ResearchMemoryEntry] = {}
        self._seed()

    def _seed(self) -> None:
        """Load hardcoded Phase 18–23 rules. Idempotent."""
        if not self._entries:
            for entry in SEEDED_RULES:
                self._entries[entry.memory_id] = entry

    def add_entry(self, entry: ResearchMemoryEntry) -> str:
        self._entries[entry.memory_id] = entry
        return entry.memory_id

    def query(
        self,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[list[str]] = None,
        content_contains: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> list[ResearchMemoryEntry]:
        """Query memory entries with optional filters."""
        results = list(self._entries.values())

        if memory_type is not None:
            results = [e for e in results if e.memory_type == memory_type]
        if tags:
            tag_set = set(tags)
            results = [e for e in results if tag_set & set(e.tags)]
        if content_contains is not None:
            q = content_contains.lower()
            results = [e for e in results if q in e.content.lower()]
        if min_confidence > 0.0:
            results = [e for e in results if e.confidence >= min_confidence]

        return results

    def compile_snapshot(self) -> dict[str, list[ResearchMemoryEntry]]:
        """Group all entries by memory type."""
        snapshot: dict[str, list[ResearchMemoryEntry]] = {}
        for entry in self._entries.values():
            snapshot.setdefault(entry.memory_type, []).append(entry)
        return snapshot

    def export_markdown(self, path: Path) -> Path:
        """Write compiled research memory as a markdown file."""
        snapshot = self.compile_snapshot()
        lines = [
            "# Research Memory v2",
            "",
            f"**Total entries:** {len(self._entries)}",
            f"**Memory types:** {len(snapshot)}",
            "",
        ]

        for mtype in sorted(snapshot.keys()):
            entries = snapshot[mtype]
            lines.append(f"## {mtype} ({len(entries)} entries)")
            lines.append("")
            for entry in entries:
                lines.append(f"### {entry.memory_id}")
                lines.append(f"- **Confidence:** {entry.confidence:.2f}")
                lines.append(f"- **Tags:** {', '.join(entry.tags)}")
                lines.append(f"- **Content:** {entry.content}")
                if entry.source_run_id:
                    lines.append(f"- **Source:** {entry.source_run_id}")
                lines.append("")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def to_json(self) -> str:
        """Serialize all entries as JSON."""
        payload = {
            eid: {
                "memory_id": e.memory_id,
                "memory_type": e.memory_type,
                "content": e.content,
                "tags": e.tags,
                "confidence": e.confidence,
                "source_run_id": e.source_run_id,
            }
            for eid, e in sorted(self._entries.items())
        }
        return json.dumps(payload, indent=2, ensure_ascii=False, default=str)
