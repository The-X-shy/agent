"""Shared adapter result contract for optical simulation backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Protocol

from pydantic import Field

from optiresearch.memory.schemas import StrictModel


class AdapterArtifact(StrictModel):
    """Backend-produced artifact before registration in ArtifactStore."""

    path: str = Field(min_length=1)
    artifact_type: str = Field(default="unknown", min_length=1)
    mime: Optional[str] = None
    content_hash: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdapterMetricBundle(StrictModel):
    """Metrics emitted by an adapter run."""

    metrics: dict[str, Any] = Field(default_factory=dict)
    primary_metric: Optional[str] = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdapterRunResult(StrictModel):
    """Stable adapter output shape shared by mock and real backends."""

    status: Literal["succeeded", "failed", "skipped"]
    artifacts: list[str] = Field(default_factory=list)
    artifact_refs: list[AdapterArtifact] = Field(default_factory=list)
    metric_bundle: AdapterMetricBundle = Field(default_factory=AdapterMetricBundle)
    logs: list[str] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def metrics(self) -> dict[str, Any]:
        return self.metric_bundle.metrics

    def __getitem__(self, key: str) -> Any:
        if key == "metrics":
            return self.metrics
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except AttributeError:
            return default

    def __iter__(self) -> Iterator[str]:
        return iter(self.model_dump())


class OpticalAdapterProtocol(Protocol):
    """Common callable surface for optical simulation adapters."""

    def validate_environment(self) -> dict[str, Any]:
        ...

    def simulate_psf_cube(self, spec: Any, sweep: Any, output_dir: Path) -> AdapterRunResult:
        ...
