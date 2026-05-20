"""SQLite JSON store used by the MVP memory layer."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel


TABLES = (
    "meta_traces",
    "artifacts",
    "run_memories",
    "design_rules",
    "claims",
    "plan_templates",
    "skill_memories",
)


class SQLiteStore:
    """Small SQLite store where typed objects are persisted as JSON payloads."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or os.getenv("OPTIRESEARCH_DB_PATH", "./workspace/optiresearch.sqlite"))

    def init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            for table in TABLES:
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id TEXT PRIMARY KEY,
                        workspace_id TEXT,
                        run_id TEXT,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT
                    )
                    """
                )
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_workspace ON {table}(workspace_id)")
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_run ON {table}(run_id)")

    def upsert(
        self,
        table: str,
        row_id: str,
        payload: dict[str, Any] | BaseModel,
        workspace_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self._check_table(table)
        self.init_db()
        now = datetime.now(timezone.utc).isoformat()
        payload_json = self._payload_json(payload)
        with self._connect() as conn:
            existing = conn.execute(f"SELECT created_at FROM {table} WHERE id = ?", (row_id,)).fetchone()
            if existing:
                conn.execute(
                    f"""
                    UPDATE {table}
                    SET workspace_id = ?, run_id = ?, payload_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (workspace_id, run_id, payload_json, now, row_id),
                )
            else:
                conn.execute(
                    f"""
                    INSERT INTO {table} (id, workspace_id, run_id, payload_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (row_id, workspace_id, run_id, payload_json, now, now),
                )

    def insert_once(
        self,
        table: str,
        row_id: str,
        payload: dict[str, Any] | BaseModel,
        workspace_id: str | None = None,
        run_id: str | None = None,
    ) -> bool:
        self._check_table(table)
        self.init_db()
        now = datetime.now(timezone.utc).isoformat()
        payload_json = self._payload_json(payload)
        with self._connect() as conn:
            try:
                conn.execute(
                    f"""
                    INSERT INTO {table} (id, workspace_id, run_id, payload_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (row_id, workspace_id, run_id, payload_json, now, now),
                )
            except sqlite3.IntegrityError:
                return False
        return True

    def get(self, table: str, row_id: str) -> dict[str, Any] | None:
        self._check_table(table)
        self.init_db()
        with self._connect() as conn:
            row = conn.execute(f"SELECT payload_json FROM {table} WHERE id = ?", (row_id,)).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def list(
        self,
        table: str,
        workspace_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self._check_table(table)
        self.init_db()
        clauses: list[str] = []
        params: list[str] = []
        if workspace_id is not None:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT payload_json FROM {table} {where} ORDER BY created_at ASC",
                tuple(params),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def search_by_text(self, table: str, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        self._check_table(table)
        self.init_db()
        needle = query.lower()
        matches: list[dict[str, Any]] = []
        for payload in self.list(table):
            if needle in json.dumps(payload, ensure_ascii=False).lower():
                matches.append(payload)
            if len(matches) >= top_k:
                break
        return matches

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _check_table(self, table: str) -> None:
        if table not in TABLES:
            raise ValueError(f"Unknown table: {table}")

    def _payload_json(self, payload: dict[str, Any] | BaseModel) -> str:
        if isinstance(payload, BaseModel):
            payload = payload.model_dump(mode="json")
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
