"""MinIO artifact backend placeholder."""

from __future__ import annotations


class MinIOArtifactStore:
    def __init__(self, *_: object, **__: object) -> None:
        raise NotImplementedError("MinIO is reserved for v1; use FileArtifactStore for the MVP.")
