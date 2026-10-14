"""Published snapshots and active cache should only write changed resources."""

import copy
import json

import pytest

from backend.app.skills.files import FileStore


def state():
    return {
        "skills": {
            "skill": {
                "id": "skill",
                "name": "repair",
                "description": "Repair errors",
                "body": "# Repair",
                "status": "active",
                "file_path": "versions/one/SKILL.md",
            }
        },
        "references": {
            "ref": {
                "id": "ref",
                "skill_id": "skill",
                "name": "retry",
                "description": "Retry errors",
                "body": "Retry once.",
                "status": "active",
                "file_path": "versions/one/retry.md",
            }
        },
    }


def test_preparing_unchanged_snapshots_creates_no_temporary_files(tmp_path, monkeypatch):
    import tempfile

    files = FileStore(tmp_path)
    data = state()
    files.prepare(data)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", lambda **kw: pytest.fail("unchanged write"))
    files.prepare(data)


def test_materializing_identical_cache_creates_no_writes(tmp_path, monkeypatch):
    files = FileStore(tmp_path)
    data = state()
    files.prepare(data)
    files.materialize(data)
    monkeypatch.setattr(files, "_atomic_write", lambda *a: pytest.fail("unchanged cache write"))
    files.materialize(data)


def test_reference_update_writes_only_its_changed_cache_file(tmp_path, monkeypatch):
    files = FileStore(tmp_path)
    data = state()
    files.prepare(data)
    files.materialize(data)
    updated = copy.deepcopy(data)
    updated["references"]["ref"].update(body="Retry twice.", file_path="versions/two/retry.md")
    files.prepare(updated)
    writes = []
    original = files._atomic_write

    def write(target, content):
        writes.append(target.relative_to(tmp_path).as_posix())
        original(target, content)

    monkeypatch.setattr(files, "_atomic_write", write)
    files.materialize(updated)
    assert writes == ["skills/repair/references/retry.md"]
    assert (tmp_path / "skills/repair/references/retry.md").read_text() == "Retry twice.\n"


@pytest.mark.parametrize(
    "invalid",
    [
        ["skills/repair/SKILL.md", "../outside.md"],
        ["skills/repair/SKILL.md", 5],
        {"skills/repair/SKILL.md": True},
        ["skills//repair/SKILL.md"],
    ],
)
def test_invalid_manifest_is_rejected_before_any_cache_mutation(tmp_path, monkeypatch, invalid):
    files = FileStore(tmp_path)
    data = state()
    files.prepare(data)
    files.materialize(data)
    master = tmp_path / "skills/repair/SKILL.md"
    master.write_text("must survive", encoding="utf-8")
    manifest = tmp_path / ".managed-files.json"
    manifest.write_text(json.dumps(invalid), encoding="utf-8")
    monkeypatch.setattr(
        files, "_atomic_write", lambda *a: pytest.fail("mutation before validation")
    )
    with pytest.raises(ValueError, match="manifest"):
        files.materialize(data)
    assert master.read_text() == "must survive"


def test_cache_retry_repairs_only_the_missing_resource(tmp_path, monkeypatch):
    files = FileStore(tmp_path)
    data = state()
    files.prepare(data)
    files.materialize(data)
    target = tmp_path / "skills/repair/references/retry.md"
    target.unlink()
    writes = []
    original = files._atomic_write

    def write(path, content):
        writes.append(path)
        original(path, content)

    monkeypatch.setattr(files, "_atomic_write", write)
    files.materialize(data)
    assert writes == [target]
    assert target.read_text() == "Retry once.\n"


def test_cache_replay_repairs_invalid_utf8_from_immutable_snapshot(tmp_path):
    files = FileStore(tmp_path)
    data = state()
    files.prepare(data)
    files.materialize(data)
    target = tmp_path / "skills/repair/references/retry.md"
    target.write_bytes(b"\xff")
    files.materialize(data)
    assert target.read_text(encoding="utf-8") == "Retry once.\n"
