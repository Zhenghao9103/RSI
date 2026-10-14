import copy
import math

import pytest


def empty_state():
    return {
        "skills": {},
        "clusters": {},
        "references": {},
        "members": {},
        "candidates": {},
        "tasks": {},
        "meta": {"pool": {"updates": 0, "dirty_clusters": []}},
    }


def sample_state():
    state = empty_state()
    state["skills"]["s1"] = {
        "id": "s1",
        "name": "repair",
        "description": "repair failed tasks",
        "body": "# Repair\n\nFollow the steps.",
        "version": 1,
        "status": "active",
        "file_path": "snapshots/s1/1/SKILL.md",
        "embedding": [1.0, 0.0, 0.0],
    }
    state["references"]["r1"] = {
        "id": "r1",
        "skill_id": "s1",
        "name": "retry",
        "description": "retry errors",
        "body": "Retry once.",
        "version": 1,
        "status": "active",
        "file_path": "snapshots/r1/1/retry.md",
        "embedding": [1.0, 0.0, 0.0],
    }
    return state


def test_repository_pending_publish_restart_and_revision_guard(tmp_path):
    from backend.app.skills.repository import Repository

    repo = Repository(tmp_path)
    revision, before = repo.load()
    assert revision == 0 and before == empty_state()
    after = sample_state()
    operation = repo.stage("op1", "IMPORT", revision, after)
    assert operation["status"] == "pending"
    assert repo.load() == (revision, before)
    second = Repository(tmp_path)
    with pytest.raises(RuntimeError, match="pending"):
        second.stage("op2", "IMPORT", revision, after)
    repo.record_failure("op1", ValueError("sensitive data"))
    assert repo.operation("op1")["last_error"] == "ValueError"
    assert repo.operation("op1")["attempts"] == 1
    repo.dispose()
    assert second.pending()[0]["id"] == "op1"
    second.publish("op1")
    assert second.load() == (1, after)
    second.publish("op1")  # Idempotent recovery.
    assert second.load() == (1, after)
    with pytest.raises(RuntimeError, match="revision"):
        second.stage("op2", "IMPORT", 0, after)
    assert second.history()[0]["before"] == before
    assert second.history()[0]["after"] == after
    second.dispose()


def test_repository_publish_applies_add_update_delete_and_rollback(tmp_path):
    from backend.app.skills.repository import Repository

    repo = Repository(tmp_path)
    first = sample_state()
    repo.stage("one", "IMPORT", 0, first)
    repo.publish("one")
    second = copy.deepcopy(first)
    second["skills"]["s1"]["description"] = "updated"
    second["references"].clear()
    second["tasks"]["task"] = {"id": "task", "done": True}
    repo.stage("two", "UPDATE", 1, second)
    repo.publish("two")
    assert repo.load() == (2, second)
    repo.stage("three", "ROLLBACK", 2, first)
    repo.publish("three")
    assert repo.load() == (3, first)
    repo.dispose()


def test_files_immutable_yaml_and_managed_cache_cleanup(tmp_path):
    from backend.app.skills.files import FileStore

    store = FileStore(tmp_path)
    state = sample_state()
    store.prepare(state)
    text = store.load(state["skills"]["s1"]["file_path"])
    assert text.startswith("---\n") and "name: repair" in text
    assert "# Repair" in text
    store.prepare(state)
    altered = copy.deepcopy(state)
    altered["skills"]["s1"]["body"] = "different"
    with pytest.raises(ValueError, match="immutable"):
        store.prepare(altered)
    store.materialize(state)
    cache = tmp_path / "skills" / "repair"
    assert (cache / "SKILL.md").read_text(encoding="utf-8") == text
    assert (cache / "references" / "retry.md").read_text(encoding="utf-8") == "Retry once.\n"
    untouched = cache / "notes.txt"
    untouched.write_text("keep", encoding="utf-8")
    state["references"]["r1"]["status"] = "archived"
    store.materialize(state)
    assert not (cache / "references" / "retry.md").exists()
    assert untouched.read_text() == "keep"


def test_files_reject_traversal_and_external_paths(tmp_path):
    from backend.app.skills.files import FileStore

    store = FileStore(tmp_path)
    with pytest.raises(ValueError):
        store.load("../outside.md")
    state = sample_state()
    state["skills"]["s1"]["file_path"] = "../outside.md"
    with pytest.raises(ValueError):
        store.prepare(state)
    state = sample_state()
    state["skills"]["s1"]["name"] = "../bad"
    with pytest.raises(ValueError):
        store.materialize(state)


def test_files_keep_promoted_redirect_reference(tmp_path):
    from backend.app.skills.files import FileStore

    store = FileStore(tmp_path)
    state = sample_state()
    state["references"]["r1"]["status"] = "promoted"
    store.prepare(state)
    store.materialize(state)
    assert (tmp_path / "skills/repair/references/retry.md").exists()


def test_snapshot_creation_failure_is_recoverable(tmp_path, monkeypatch):
    import os
    from backend.app.skills.files import FileStore

    store = FileStore(tmp_path)
    state = sample_state()
    original = os.link

    def fail_link(*args, **kwargs):
        raise OSError("simulated interrupted snapshot publication")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError):
        store.prepare(state)
    assert not (tmp_path / state["skills"]["s1"]["file_path"]).exists()
    monkeypatch.setattr(os, "link", original)
    store.prepare(state)
    assert "# Repair" in store.load(state["skills"]["s1"]["file_path"])


def test_index_existing_dimension_is_checked_before_new_candidate(tmp_path):
    from backend.app.skills.index import SkillIndex

    first = SkillIndex(tmp_path, embed=lambda _: [1.0, 0.0, 0.0])
    first.sync(empty_state(), sample_state())
    reopened = SkillIndex(tmp_path, embed=lambda _: [1.0, 0.0])
    with pytest.raises(ValueError, match="dimension"):
        reopened.encode("candidate")


def test_index_normalizes_and_rejects_invalid_embeddings(tmp_path):
    from backend.app.skills.index import SkillIndex

    index = SkillIndex(tmp_path, embed=lambda _: [3.0, 4.0, 0.0])
    assert index.encode("test") == pytest.approx([0.6, 0.8, 0.0])
    index.embed = lambda _: [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        index.encode("zero")
    index.embed = lambda _: [math.nan, 0.0, 1.0]
    with pytest.raises(ValueError):
        index.encode("nan")
    index.embed = lambda _: [1.0, 0.0]
    with pytest.raises(ValueError, match="dimension"):
        index.encode("dimension")


def test_index_real_chroma_excludes_unpublished_and_stale_versions(tmp_path):
    from backend.app.skills.index import SkillIndex

    index = SkillIndex(tmp_path, embed=lambda _: [1.0, 0.0, 0.0])
    first = sample_state()
    index.sync(empty_state(), first)
    assert index.discover(empty_state(), "repair") == []
    assert index.discover(first, "repair")[0]["version"] == 1
    updated = copy.deepcopy(first)
    updated["skills"]["s1"]["version"] = 2
    updated["skills"]["s1"]["embedding"] = [0.0, 1.0, 0.0]
    index.sync(first, updated)
    assert index.discover(updated, "repair") == []
    assert index.discover(first, "repair")[0]["version"] == 1
    refs = index.discover(first, "retry", skill_id="s1", kind="reference")
    assert refs[0]["id"] == "r1"
    archived = copy.deepcopy(first)
    archived["skills"]["s1"]["status"] = "archived"
    assert index.discover(archived, "repair") == []
