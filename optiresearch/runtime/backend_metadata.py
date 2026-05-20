"""Backend evidence metadata helpers."""

from __future__ import annotations

import sys
from importlib import metadata as importlib_metadata
from typing import Any


DEEPLENS_SMOKE_CAVEAT = (
    "DeepLens backend currently validates integration-level behavior only; "
    "encoder-specific optical behavior is not yet fully realized."
)

DEEPLENS_PROXY_CAVEAT = "adapter-proxy DeepLens evidence; not native physical validation"


def backend_metadata(backend: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backend": backend,
        "backend_capability_level": "mock" if backend == "mock_deeplens" else "smoke",
        "encoder_behavior_realized": backend == "mock_deeplens",
        "python_executable": sys.executable,
    }
    if backend == "deeplens":
        payload["deeplens_version"] = _deeplens_version()
    else:
        payload["deeplens_version"] = None
    if extra:
        payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


def enrich_backend_metrics(metrics: dict[str, Any], backend: str) -> dict[str, Any]:
    enriched = dict(metrics)
    meta = backend_metadata(backend)
    enriched.setdefault("backend_capability_level", meta["backend_capability_level"])
    enriched.setdefault("encoder_behavior_realized", meta["encoder_behavior_realized"])
    if backend == "deeplens":
        if "mock_mtf_mean" in enriched:
            enriched.setdefault("deeplens_mtf_mean", enriched["mock_mtf_mean"])
        if "mock_energy_efficiency" in enriched:
            enriched.setdefault("deeplens_energy_efficiency", enriched["mock_energy_efficiency"])
    return enriched


def _deeplens_version() -> str | None:
    try:
        return importlib_metadata.version("deeplens-core")
    except importlib_metadata.PackageNotFoundError:
        return None
