"""Remote worker registry backed by workspace/remote_workers/workers.json."""

from __future__ import annotations

import json
import os
from pathlib import Path

from optiresearch.schemas.remote import RemoteWorkerSpec


class RemoteWorkerRegistry:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or os.getenv("OPTIRESEARCH_REMOTE_WORKER_ROOT", "workspace/remote_workers"))
        self.config_path = self.root / "workers.json"

    def list_workers(self) -> list[RemoteWorkerSpec]:
        payload = self._read()
        return [RemoteWorkerSpec(**item) for item in payload.get("workers", [])]

    def get_worker(self, worker_id: str) -> RemoteWorkerSpec:
        for worker in self.list_workers():
            if worker.worker_id == worker_id:
                return worker
        raise KeyError(f"Unknown remote worker: {worker_id}")

    def add_worker(self, worker: RemoteWorkerSpec) -> RemoteWorkerSpec:
        workers = [item for item in self.list_workers() if item.worker_id != worker.worker_id]
        workers.append(worker)
        self._write({"workers": [item.model_dump(mode="json") for item in workers]})
        return worker

    def _read(self) -> dict:
        if not self.config_path.exists():
            return {"workers": []}
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def _write(self, payload: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
