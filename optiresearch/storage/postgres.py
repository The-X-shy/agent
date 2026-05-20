"""PostgreSQL backend placeholder for future production deployments."""

from __future__ import annotations


class PostgresStore:
    def __init__(self, *_: object, **__: object) -> None:
        raise NotImplementedError("PostgreSQL is reserved for v1; use SQLiteStore for the MVP.")
