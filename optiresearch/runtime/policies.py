"""Runtime policy helpers."""

from __future__ import annotations


def allow_mock_backend(backend: str) -> bool:
    """Return whether the backend is allowed in the MVP."""

    return backend == "mock_deeplens"
