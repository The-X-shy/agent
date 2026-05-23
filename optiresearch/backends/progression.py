"""Backend progression graph for multi-backend autonomous loops.

Defines valid backend-to-backend transitions with expected claim gains,
runtime costs, and preconditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class BackendProgressionEdge:
    from_backend: str
    to_backend: str
    reason: str
    expected_claim_gain: str
    runtime_cost: str
    required_preconditions: list[str] = field(default_factory=list)
    fallback_backend: Optional[str] = None
    allowed_task_types: list[str] = field(default_factory=list)


_DEFAULT_EDGES: list[BackendProgressionEdge] = [
    BackendProgressionEdge(
        from_backend="mock_deeplens",
        to_backend="phase_to_fft_proxy",
        reason="Move from mock simulation to differentiable proxy",
        expected_claim_gain="mock_simulation -> native_full_reconstruction_proxy",
        runtime_cost="low",
        allowed_task_types=["stable_lens_hsi_codesign", "lightweight_psf_probe"],
    ),
    BackendProgressionEdge(
        from_backend="phase_to_fft_proxy",
        to_backend="deeplens_geolens_geometric",
        reason="Move from FFT proxy to DeepLens native lens simulation",
        expected_claim_gain="native_full_reconstruction_proxy -> native_lens_simulation",
        runtime_cost="requires_deeplens",
        required_preconditions=["deeplens_available"],
        fallback_backend="phase_to_fft_proxy",
        allowed_task_types=["psf_probe", "stable_lens_hsi_codesign"],
    ),
    BackendProgressionEdge(
        from_backend="phase_to_fft_proxy",
        to_backend="deeplens_fresnel_component",
        reason="Move from FFT proxy to native differentiable component",
        expected_claim_gain="native_full_reconstruction_proxy -> native_component_optimization",
        runtime_cost="requires_deeplens",
        required_preconditions=["deeplens_available"],
        fallback_backend="phase_to_fft_proxy",
        allowed_task_types=["native_optimization_probe", "component_optimization"],
    ),
    BackendProgressionEdge(
        from_backend="deeplens_fresnel_component",
        to_backend="deeplens_geolens_geometric",
        reason="Move from component-level to lens-file simulation",
        expected_claim_gain="native_component_optimization -> native_lens_simulation",
        runtime_cost="requires_deeplens",
        required_preconditions=["deeplens_available"],
        allowed_task_types=["psf_probe", "stable_lens_hsi_codesign"],
    ),
    BackendProgressionEdge(
        from_backend="deeplens_geolens_geometric",
        to_backend="deeplens_coherent_asm",
        reason="Probe full wave-optics path (non-differentiable)",
        expected_claim_gain="native_lens_simulation -> waveoptics_probe",
        runtime_cost="requires_deeplens",
        required_preconditions=["deeplens_available"],
        allowed_task_types=["psf_probe"],
    ),
    BackendProgressionEdge(
        from_backend="deeplens_geolens_geometric",
        to_backend="deeplens_geolens_geometric",
        reason="Remote validation on WSL worker",
        expected_claim_gain="local_native_lens_simulation -> remote_validated",
        runtime_cost="requires_remote",
        required_preconditions=["deeplens_available", "remote_worker_available"],
        fallback_backend="deeplens_geolens_geometric",
        allowed_task_types=["stable_lens_hsi_codesign"],
    ),
    BackendProgressionEdge(
        from_backend="local_synthetic_hsi",
        to_backend="phase_to_fft_proxy",
        reason="Move from synthetic data to differentiable proxy",
        expected_claim_gain="synthetic_hsi_simulation -> native_full_reconstruction_proxy",
        runtime_cost="low",
        allowed_task_types=["stable_lens_hsi_codesign", "lightweight_psf_probe"],
    ),
]


class BackendProgressionGraph:
    def __init__(self, edges: Optional[list[BackendProgressionEdge]] = None):
        self._edges = edges or list(_DEFAULT_EDGES)
        self._from_map: dict[str, list[BackendProgressionEdge]] = {}
        for edge in self._edges:
            self._from_map.setdefault(edge.from_backend, []).append(edge)

    @property
    def nodes(self) -> set[str]:
        nodes: set[str] = set()
        for edge in self._edges:
            nodes.add(edge.from_backend)
            nodes.add(edge.to_backend)
        return nodes

    def get_edges_from(self, backend_id: str) -> list[BackendProgressionEdge]:
        return self._from_map.get(backend_id, [])

    def get_next(
        self, backend_id: str, reason: str = "", prefer_local: bool = True
    ) -> Optional[BackendProgressionEdge]:
        edges = self.get_edges_from(backend_id)
        if not edges:
            return None
        if prefer_local:
            local = [e for e in edges if e.runtime_cost == "low"]
            if local:
                return local[0]
        return edges[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": sorted(self.nodes),
            "edges": [
                {
                    "from": e.from_backend,
                    "to": e.to_backend,
                    "reason": e.reason,
                    "expected_claim_gain": e.expected_claim_gain,
                    "runtime_cost": e.runtime_cost,
                }
                for e in self._edges
            ],
        }


_default_graph: Optional[BackendProgressionGraph] = None


def _get_graph() -> BackendProgressionGraph:
    global _default_graph
    if _default_graph is None:
        _default_graph = BackendProgressionGraph()
    return _default_graph


def get_next_backend(
    current_backend: str,
    reason: str = "",
    prefer_local: bool = True,
) -> Optional[dict[str, Any]]:
    graph = _get_graph()
    edge = graph.get_next(current_backend, reason=reason, prefer_local=prefer_local)
    if edge is None:
        return None
    return {
        "next_backend": edge.to_backend,
        "reason": edge.reason,
        "expected_claim_gain": edge.expected_claim_gain,
        "runtime_cost": edge.runtime_cost,
        "required_preconditions": edge.required_preconditions,
        "fallback_backend": edge.fallback_backend,
        "allowed_task_types": edge.allowed_task_types,
    }


def get_all_edges_from(
    backend_id: str,
) -> list[dict[str, Any]]:
    """Return ALL edges from a given backend, including non-preferred ones.

    Unlike get_next_backend() which returns only the first edge, this
    returns all edges so the loop can try alternatives when the primary
    fails.
    """
    graph = _get_graph()
    edges = graph.get_edges_from(backend_id)
    return [
        {
            "next_backend": e.to_backend,
            "reason": e.reason,
            "expected_claim_gain": e.expected_claim_gain,
            "runtime_cost": e.runtime_cost,
            "required_preconditions": e.required_preconditions,
            "fallback_backend": e.fallback_backend,
            "allowed_task_types": e.allowed_task_types,
        }
        for e in edges
    ]


def list_progression_from(backend_id: str) -> list[str]:
    graph = _get_graph()
    return [e.to_backend for e in graph.get_edges_from(backend_id)]


def export_progression_graph_markdown(path: Path) -> Path:
    graph = _get_graph()
    lines = [
        "# Backend Progression Graph",
        "",
        "| From | To | Reason | Claim Gain | Runtime Cost |",
        "|---|---|---|---|---|",
    ]
    for e in graph._edges:
        lines.append(
            f"| {e.from_backend} | {e.to_backend} | {e.reason} | "
            f"{e.expected_claim_gain} | {e.runtime_cost} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
