import json

from optiresearch.memory.schemas import compute_file_sha256
from optiresearch.storage.file_artifact_store import FileArtifactStore
from optiresearch.storage.sqlite_store import SQLiteStore


def test_register_json_and_file_artifacts(tmp_path):
    sqlite = SQLiteStore(tmp_path / "memory.sqlite")
    sqlite.init_db()
    artifacts = FileArtifactStore(root=tmp_path / "artifacts", store=sqlite)

    json_ref = artifacts.register_json(
        {"metric": 0.9},
        workspace_id="ws",
        run_id="run-1",
        trace_id="trace-1",
        producer="unit-test",
        metadata={"kind": "metrics"},
        metrics={"metric": 0.9},
    )
    source = tmp_path / "source.txt"
    source.write_text("artifact body", encoding="utf-8")
    file_ref = artifacts.register_file(
        source,
        workspace_id="ws",
        run_id="run-1",
        trace_id="trace-1",
        producer="unit-test",
        metadata={"kind": "text"},
        metrics={},
    )

    assert json.loads((tmp_path / json_ref.uri).read_text(encoding="utf-8")) == {"metric": 0.9}
    assert file_ref.content_hash == compute_file_sha256(tmp_path / file_ref.uri)
    assert artifacts.get_artifact(json_ref.artifact_id) == json_ref
    assert {item.artifact_id for item in artifacts.list_artifacts(run_id="run-1")} == {
        json_ref.artifact_id,
        file_ref.artifact_id,
    }
