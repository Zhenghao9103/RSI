"""End-to-end structure and recovery checks with real temporary pool storage."""

from copy import deepcopy

import pytest

from backend.app.skills import SkillManager
from backend.app.skills.config import SkillPoolSettings
from backend.app.skills.locking import writer_lock


def draft(name="data-analysis", description="Analyze supplied data", body=None):
    return {
        "name": name,
        "description": description,
        "body": body
        or "## When to use\nSupplied data\n## Steps\nInspect input\n"
        "## Constraints\nPreserve observations",
    }


def embedding(text):
    if "statistics" in text.lower():
        return [0.8, 0.6, 0.0]
    return [1.0, 0.0, 0.0]


class Decisions:
    def __init__(self):
        self.relations = []
        self.evolution = {"action": "KEEP", "reason": "Stable"}
        self.generated = {**draft(), "confidence": 0.95}

    def consolidate(self, candidate, master, references):
        if self.relations:
            return deepcopy(self.relations.pop(0))
        if master is None:
            return {
                "action": "NEW_CLUSTER",
                "reason": "New responsibility",
                "has_information": True,
                "master": draft(candidate["name"], candidate["description"]),
                "reference": draft("workflow", candidate["description"]),
            }
        return {
            "action": "ADD_REFERENCE",
            "reason": "Complementary procedure",
            "has_information": True,
            "reference": draft(candidate["name"], candidate["description"]),
        }

    def maintenance(self, *args, **kwargs):
        return deepcopy(self.evolution)

    def generate(self, trace):
        return deepcopy(self.generated)


@pytest.fixture
def pool(tmp_path):
    decisions = Decisions()
    manager = SkillManager(
        tmp_path / "pool",
        embed=embedding,
        policy=decisions,
        settings=SkillPoolSettings(cooldown_hours=0),
    )
    yield manager, decisions
    manager.close()


def accept(manager, name="data-analysis", description="Analyze supplied data"):
    candidate = manager.import_candidate(draft(name, description))
    return manager.integrate(candidate["id"])


def refactor(decisions, reference_id, body):
    decisions.evolution = {
        "action": "REFACTOR",
        "reason": "Remove repetition",
        "reference_id": reference_id,
        "reference": draft("workflow", body=body),
    }


def test_refactor_preserves_reference_and_historical_member_ids(pool):
    manager, decisions = pool
    first = accept(manager)
    before = manager.state()
    refactor(
        decisions,
        first["reference_id"],
        "## Workflow\nInspect then report without duplication.",
    )
    manager.maintain(first["cluster_id"], action="REFACTOR")
    after = manager.state()
    assert set(after["references"]) == set(before["references"])
    assert after["members"] == before["members"]
    assert (
        after["clusters"][first["cluster_id"]]["centroid"]
        == before["clusters"][first["cluster_id"]]["centroid"]
    )
    assert "without duplication" in manager.load_reference(first["reference_id"])


def test_reference_merge_remaps_members_without_duplicate_weighting(pool):
    manager, decisions = pool
    first = accept(manager)
    second = accept(manager, "statistics-check", "statistics data")
    before = manager.state()
    decisions.evolution = {
        "action": "MERGE",
        "reason": "Overlapping procedures",
        "source_reference_ids": [second["reference_id"]],
        "target_reference_id": first["reference_id"],
        "reference": draft("workflow", "Analyze data and statistics"),
    }
    manager.maintain(first["cluster_id"], action="MERGE")
    after = manager.state()
    assert set(after["members"]) == set(before["members"])
    assert {member["reference_id"] for member in after["members"].values()} == {
        first["reference_id"]
    }
    assert after["references"][second["reference_id"]]["status"] == "archived"
    assert after["references"][second["reference_id"]]["merged_into"] == first["reference_id"]
    assert after["clusters"][first["cluster_id"]]["member_count"] == 2
    assert after["clusters"][first["cluster_id"]]["centroid"] == pytest.approx(
        before["clusters"][first["cluster_id"]]["centroid"]
    )
    results = manager.discover_references(first["skill_id"], "Analyze data")
    assert [item["id"] for item in results] == [first["reference_id"]]


def test_cluster_merge_requires_per_reference_relation_then_recomputes_target(pool):
    manager, decisions = pool
    first = accept(manager)
    decisions.relations = [
        {"action": "NEW_CLUSTER", "reason": "Different scope", "has_information": True},
        {
            "action": "NEW_CLUSTER",
            "reason": "Independent",
            "has_information": True,
            "master": draft("statistics", "statistics data"),
            "reference": draft("workflow", "statistics data"),
        },
    ]
    second = accept(manager, "statistics", "statistics data")
    before = manager.state()
    decisions.evolution = {
        "action": "MERGE",
        "reason": "Related workflows",
        "target_cluster_id": first["cluster_id"],
    }
    decisions.relations = [
        {"action": "NEW_CLUSTER", "reason": "Unsafe boundary", "has_information": True}
    ]
    with pytest.raises(ValueError, match="boundary"):
        manager.maintain(second["cluster_id"], action="MERGE")
    assert manager.state() == before
    decisions.relations = [
        {
            "action": "UPDATE_REFERENCE",
            "reason": "Shared workflow with extension",
            "has_information": True,
            "reference_id": first["reference_id"],
            "reference": draft("workflow", "Analyze data including statistics"),
        }
    ]
    manager.maintain(second["cluster_id"], action="MERGE")
    after = manager.state()
    assert set(after["members"]) == set(before["members"])
    assert {member["cluster_id"] for member in after["members"].values()} == {first["cluster_id"]}
    assert after["clusters"][second["cluster_id"]]["status"] == "archived"
    assert after["clusters"][second["cluster_id"]]["centroid"] == []
    assert after["clusters"][first["cluster_id"]]["member_count"] == 2
    assert after["clusters"][first["cluster_id"]]["centroid"] == pytest.approx(
        [0.948683298, 0.316227766, 0]
    )
    assert [item["id"] for item in manager.discover("Analyze data")] == [first["skill_id"]]


def test_promoted_pointer_follows_target_cluster_merge(pool):
    manager, decisions = pool
    parent = accept(manager)
    child = accept(manager, "statistics-check", "statistics data")
    decisions.evolution = {
        "action": "PROMOTE",
        "reason": "Independent workflow",
        "reference_id": child["reference_id"],
        "independent": True,
        "complete_workflow": True,
        "draft": draft("statistical-testing", "statistics tests"),
    }
    promoted = manager.maintain(parent["cluster_id"], action="PROMOTE")
    decisions.relations = [
        {"action": "NEW_CLUSTER", "reason": "Separate scope", "has_information": True},
        {"action": "NEW_CLUSTER", "reason": "Separate scope", "has_information": True},
        {
            "action": "NEW_CLUSTER",
            "reason": "Independent hub",
            "has_information": True,
            "master": draft("analysis-hub", "Analyze supplied data"),
            "reference": draft("workflow", "Analyze supplied data"),
        },
    ]
    hub = accept(manager, "analysis-hub", "Analyze supplied data")
    decisions.evolution = {
        "action": "MERGE",
        "reason": "Shared responsibility",
        "target_cluster_id": hub["cluster_id"],
    }
    decisions.relations = [
        {
            "action": "UPDATE_REFERENCE",
            "reason": "Add statistical workflow",
            "has_information": True,
            "reference_id": hub["reference_id"],
            "reference": draft("workflow", "Analyze supplied data including statistics"),
        }
    ]
    manager.maintain(promoted["target_cluster_id"], action="MERGE")
    pointer = manager.state()["references"][child["reference_id"]]
    assert pointer["status"] == "promoted"
    assert pointer["promoted_to"] == hub["skill_id"]
    assert pointer["promoted_reference_id"] == hub["reference_id"]
    assert "../../analysis-hub/SKILL.md" in manager.load_reference(child["reference_id"])


def test_reference_archive_exits_discovery_and_removes_its_center_weight(pool):
    manager, decisions = pool
    first = accept(manager)
    second = accept(manager, "statistics-check", "statistics data")
    decisions.evolution = {
        "action": "ARCHIVE",
        "reason": "Obsolete procedure",
        "reference_id": second["reference_id"],
    }
    manager.maintain(first["cluster_id"], action="ARCHIVE")
    state = manager.state()
    assert state["references"][second["reference_id"]]["status"] == "archived"
    assert list(state["members"]) == [first["candidate_id"]]
    assert state["clusters"][first["cluster_id"]]["centroid"] == pytest.approx([1, 0, 0])
    assert second["reference_id"] not in [
        item["id"] for item in manager.discover_references(first["skill_id"], "statistics")
    ]
    with pytest.raises(ValueError, match="loadable"):
        manager.load_reference(second["reference_id"])


def test_cluster_archive_exits_both_indexes_and_zeroes_center(pool):
    manager, decisions = pool
    first = accept(manager)
    decisions.evolution = {
        "action": "ARCHIVE",
        "reason": "Obsolete capability",
        "cluster_id": first["cluster_id"],
    }
    manager.maintain(first["cluster_id"], action="ARCHIVE")
    state = manager.state()
    assert state["clusters"][first["cluster_id"]]["centroid"] == []
    assert state["clusters"][first["cluster_id"]]["member_count"] == 0
    assert state["skills"][first["skill_id"]]["status"] == "archived"
    assert state["references"][first["reference_id"]]["status"] == "archived"
    assert manager.discover("Analyze data") == []
    assert manager.discover_references(first["skill_id"], "Analyze data") == []


def test_rollback_requires_newest_order_and_restores_two_operations(pool):
    manager, decisions = pool
    first = accept(manager)
    original = manager.state()
    refactor(decisions, first["reference_id"], "Revision one: concise steps.")
    one = manager.maintain(first["cluster_id"], action="REFACTOR")
    after_one = manager.state()
    refactor(decisions, first["reference_id"], "Revision two: revised constraints.")
    two = manager.maintain(first["cluster_id"], action="REFACTOR")
    with pytest.raises(ValueError, match="reverse order"):
        manager.rollback(one["operation_id"])
    manager.rollback(two["operation_id"])
    assert manager.state() == after_one
    assert "Revision one" in manager.load_reference(first["reference_id"])
    manager.rollback(one["operation_id"])
    assert manager.state() == original
    assert "Revision one" not in manager.load_reference(first["reference_id"])
    assert [item["id"] for item in manager.discover("Analyze data")] == [first["skill_id"]]


def test_manual_generate_rejects_low_confidence_without_writing_candidate(pool):
    manager, decisions = pool
    decisions.generated["confidence"] = 0.4
    trace = {
        "task_id": "t1",
        "user_query": "Save method",
        "completion_status": "success",
        "trace_events": [{"tool": "repair", "status": "ok"}],
        "final_answer": "Repaired",
    }
    before = manager.state()
    with pytest.raises(ValueError, match="confidence"):
        manager.generate_candidate(trace)
    assert manager.state() == before


def test_duplicate_import_and_integration_are_idempotent(pool):
    manager, _ = pool
    first = accept(manager)
    revision = manager.status()["revision"]
    candidate = manager.import_candidate(draft())
    assert candidate["id"] == first["candidate_id"]
    manager.integrate(candidate["id"])
    assert manager.status()["revision"] == revision
    assert len(manager.state()["members"]) == 1


@pytest.mark.parametrize("component, method", [("files", "prepare"), ("repository", "publish")])
def test_failed_resource_or_publication_recovers_without_partial_state(
    pool, monkeypatch, component, method
):
    manager, _ = pool
    candidate = manager.import_candidate(draft())
    before = manager.state()
    target = getattr(manager, component)
    original = getattr(target, method)

    def fail(*args, **kwargs):
        raise OSError("Injected failure")

    monkeypatch.setattr(target, method, fail)
    with pytest.raises(OSError, match="Injected"):
        manager.integrate(candidate["id"])
    assert manager.state() == before
    assert len(manager.status()["pending_operations"]) == 1
    assert manager.discover("Analyze data") == []
    monkeypatch.setattr(target, method, original)
    recovered = manager.recover()
    assert len(recovered) == 1 and recovered[0]["status"] == "committed"
    assert manager.status()["pending_operations"] == []
    assert manager.status()["active_skills"] == 1
    assert len(manager.discover("Analyze data")) == 1
    assert manager.recover() == []


def test_status_uses_no_model_or_embedding(tmp_path):
    def fail(*args, **kwargs):
        raise AssertionError("status attempted model access")

    manager = SkillManager(tmp_path / "status", llm=fail, embed=fail)
    try:
        assert manager.status()["active_skills"] == 0
        assert manager.index._collection_value is None
    finally:
        manager.close()


def test_second_manager_cannot_write_same_root_while_writer_is_active(pool):
    manager, decisions = pool
    other = SkillManager(manager.root, embed=embedding, policy=decisions)
    try:
        with writer_lock(manager.root):
            with pytest.raises(RuntimeError, match="busy"):
                other.import_candidate(draft())
        assert other.import_candidate(draft())["status"] == "candidate"
    finally:
        other.close()


def test_usage_metadata_does_not_upsert_unchanged_vectors(pool):
    manager, _ = pool
    first = accept(manager)
    collection = manager.index.collection

    class TrackingCollection:
        def __init__(self):
            self.upserts = 0

        def __getattr__(self, name):
            return getattr(collection, name)

        def upsert(self, **kwargs):
            self.upserts += 1
            return collection.upsert(**kwargs)

    tracking = TrackingCollection()
    manager.index._collection_value = tracking
    before = manager.state()
    manager.record_usage(first["reference_id"], "independent-task", independent=True)
    assert manager.state()["references"][first["reference_id"]]["independent_task_ids"] == [
        "independent-task"
    ]
    assert (
        manager.state()["references"][first["reference_id"]]["version"]
        == before["references"][first["reference_id"]]["version"]
    )
    assert tracking.upserts == 0
