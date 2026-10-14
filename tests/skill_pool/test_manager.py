from copy import deepcopy

import pytest


def embedding(text):
    text = text.lower()
    if "statistics" in text:
        return [0.8, 0.6, 0.0]
    if "writing" in text:
        return [0.0, 0.0, 1.0]
    return [1.0, 0.0, 0.0]


def draft(name="data-analysis", description="Analyze data"):
    return {
        "name": name,
        "description": description,
        "body": "## When to use\nAnalyze supplied data.\n## Steps\n1. Inspect input.\n"
        "2. Report results.\n## Constraints\nPreserve supplied values.",
    }


class Decisions:
    def __init__(self):
        self.next = []
        self.evolution = None

    def consolidate(self, candidate, master, references):
        if self.next:
            return self.next.pop(0)
        if master is None:
            return {
                "action": "NEW_CLUSTER",
                "reason": "new capability",
                "has_information": True,
                "master": draft(candidate["name"], candidate["description"]),
                "reference": draft("workflow", candidate["description"]),
            }
        return {
            "action": "ADD_REFERENCE",
            "reason": "complementary workflow",
            "has_information": True,
            "reference": draft(candidate["name"], candidate["description"]),
        }

    def maintenance(self, *args, **kwargs):
        return deepcopy(self.evolution or {"action": "KEEP", "reason": "stable"})

    def generate(self, trace):
        return draft()


def pool(tmp_path):
    from backend.app.skills import SkillManager

    policy = Decisions()
    return SkillManager(tmp_path / "pool", embed=embedding, policy=policy), policy


def accept(manager, name="data-analysis", description="Analyze data"):
    candidate = manager.import_candidate(draft(name, description))
    return manager.integrate(candidate["id"])


def test_similar_skills_update_reference_and_duplicate_does_not_reweight(tmp_path):
    manager, policy = pool(tmp_path)
    first = accept(manager)
    reference_id = first["reference_id"]
    policy.next = [
        {
            "action": "UPDATE_REFERENCE",
            "reason": "new missing-value rules",
            "has_information": True,
            "reference_id": reference_id,
            "reference": {
                **draft("workflow", "Analyze data including missing values"),
                "body": draft()["body"] + "\nCheck inputs including missing values.",
            },
        }
    ]
    second = accept(manager, "data-cleaning", "Clean data")
    assert first["cluster_id"] == second["cluster_id"]
    state = manager.state()
    cluster = state["clusters"][first["cluster_id"]]
    assert cluster["member_count"] == 2
    assert len(state["references"]) == 1
    assert state["references"][reference_id]["update_count"] == 1
    policy.next = [{"action": "IGNORE", "reason": "covered", "has_information": False}]
    ignored = accept(manager, "data-inspection", "Inspect data")
    assert ignored["action"] == "IGNORE"
    assert manager.state()["clusters"][first["cluster_id"]]["member_count"] == 2
    assert manager.discover("Analyze data")[0]["id"] == first["skill_id"]
    master = manager.load_master(first["skill_id"])
    assert "references/workflow.md" in master
    assert "including missing values" in manager.load_reference(reference_id)


def test_boundary_rejection_tries_next_cluster(tmp_path):
    manager, policy = pool(tmp_path)
    one = accept(manager)
    # Force a second, semantically close but distinct cluster.
    policy.next = [
        {"action": "NEW_CLUSTER", "reason": "independent boundary", "has_information": True},
        {
            "action": "NEW_CLUSTER",
            "reason": "new",
            "has_information": True,
            "master": draft("statistics", "statistics data"),
            "reference": draft("workflow", "statistics data"),
        },
    ]
    two = accept(manager, "statistics", "statistics data")
    policy.next = [
        {"action": "NEW_CLUSTER", "reason": "wrong boundary", "has_information": True},
        {
            "action": "ADD_REFERENCE",
            "reason": "belongs to statistics",
            "has_information": True,
            "reference": draft("data-tests", "data tests"),
        },
    ]
    three = accept(manager, "data-tests", "data tests")
    assert one["cluster_id"] != two["cluster_id"] == three["cluster_id"]


def test_promote_migrates_members_and_cooldown_blocks_reverse_merge(tmp_path):
    manager, policy = pool(tmp_path)
    first = accept(manager)
    second = accept(manager, "statistics-check", "statistics data")
    policy.evolution = {
        "action": "PROMOTE",
        "reason": "independent request workflow",
        "reference_id": second["reference_id"],
        "independent": True,
        "complete_workflow": True,
        "draft": draft("statistical-testing", "statistics tests"),
    }
    result = manager.maintain(first["cluster_id"], action="PROMOTE")
    state = manager.state()
    assert result["target_cluster_id"] != first["cluster_id"]
    assert state["clusters"][first["cluster_id"]]["member_count"] == 1
    assert state["clusters"][result["target_cluster_id"]]["member_count"] == 1
    assert len(state["members"]) == 2
    assert state["references"][second["reference_id"]]["status"] == "promoted"
    policy.evolution = {
        "action": "MERGE",
        "reason": "reverse merge",
        "target_cluster_id": first["cluster_id"],
    }
    with pytest.raises(ValueError, match="cooldown"):
        manager.maintain(result["target_cluster_id"], action="MERGE")


def test_split_recomputes_centroids_without_reembedding_history(tmp_path):
    manager, policy = pool(tmp_path)
    one = accept(manager)
    two = accept(manager, "statistics-check", "statistics data")
    state = manager.state()
    members = list(state["members"].values())
    original_vectors = {m["id"]: m["embedding"] for m in members}
    policy.evolution = {
        "action": "SPLIT",
        "reason": "independent workflows",
        "independent": True,
        "groups": [
            {
                "master": draft("data-inspection", "Inspect data"),
                "reference_ids": [one["reference_id"]],
                "member_ids": [members[0]["id"]],
            },
            {
                "master": draft("statistical-testing", "statistics data"),
                "reference_ids": [two["reference_id"]],
                "member_ids": [members[1]["id"]],
            },
        ],
    }
    result = manager.maintain(one["cluster_id"], action="SPLIT")
    after = manager.state()
    assert after["clusters"][one["cluster_id"]]["status"] == "archived"
    assert len(result["cluster_ids"]) == 2
    for cid in result["cluster_ids"]:
        cluster = after["clusters"][cid]
        member = next(m for m in after["members"].values() if m["cluster_id"] == cid)
        assert cluster["centroid"] == pytest.approx(member["embedding"])
        assert cluster["member_count"] == 1
    assert {m["id"]: m["embedding"] for m in after["members"].values()} == original_vectors
    assert all(hit["id"] != one["skill_id"] for hit in manager.discover("data"))


def test_failed_index_publish_recovers_and_rolls_back(tmp_path, monkeypatch):
    manager, _ = pool(tmp_path)
    candidate = manager.import_candidate(draft())
    before = manager.state()
    sync = manager.index.sync
    monkeypatch.setattr(manager.index, "sync", lambda *a: (_ for _ in ()).throw(OSError("down")))
    with pytest.raises(OSError):
        manager.integrate(candidate["id"])
    assert manager.state() == before
    assert not manager.discover("data")
    pending = manager.repository.pending()
    assert len(pending) == 1
    assert pending[0]["last_error"] == "OSError"
    monkeypatch.setattr(manager.index, "sync", sync)
    manager.recover()
    assert len(manager.discover("data")) == 1
    committed_id = pending[0]["id"]
    manager.rollback(committed_id)
    assert manager.state() == before
    assert manager.discover("data") == []


def test_generation_usage_and_manual_only_maintenance(tmp_path):
    manager, _ = pool(tmp_path)
    trace = {
        "task_id": "t1",
        "user_query": "Please save this method",
        "trace_events": [{"type": "tool_result", "status": "ok", "tool": "inspect_document"}],
        "final_answer": "Method completed",
        "completion_status": "completed",
    }
    observed = manager.observe(trace)
    assert observed["triggered"] is True
    assert not manager.state()["candidates"]
    candidate = manager.generate_candidate(trace)
    result = manager.integrate(candidate["id"])
    for task_id in ["a", "a", "b", "c"]:
        manager.record_usage(result["reference_id"], task_id, independent=True)
    ref = manager.state()["references"][result["reference_id"]]
    assert ref["independent_task_ids"] == ["a", "b", "c"]
    assert manager.status()["maintenance_due"]
    assert ref["status"] == "active"


def test_split_rejects_omitted_members_and_keeps_state(tmp_path):
    manager, policy = pool(tmp_path)
    one = accept(manager)
    two = accept(manager, "statistics-check", "statistics data")
    before = manager.state()
    policy.evolution = {
        "action": "SPLIT",
        "reason": "invalid partition",
        "independent": True,
        "groups": [
            {"master": draft("data-one"), "reference_ids": [one["reference_id"]], "member_ids": []},
            {"master": draft("data-two"), "reference_ids": [two["reference_id"]], "member_ids": []},
        ],
    }
    with pytest.raises(ValueError, match="partition"):
        manager.maintain(one["cluster_id"], action="SPLIT")
    assert manager.state() == before


def test_keep_acknowledges_usage_signals_without_repeated_reviews(tmp_path):
    manager, _ = pool(tmp_path)
    result = accept(manager)
    for task_id in ["a", "b", "c"]:
        manager.record_usage(result["reference_id"], task_id, independent=True)
    assert manager.status()["maintenance_due"]
    result = manager.maintain(result["cluster_id"])
    assert result["action"] == "KEEP"
    assert manager.status()["maintenance_due"] is False


def test_promoted_pointer_tracks_target_split(tmp_path):
    from backend.app.skills import SkillPoolSettings

    manager, policy = pool(tmp_path)
    manager.settings = SkillPoolSettings(cooldown_hours=0)
    parent = accept(manager)
    second = accept(manager, "statistics-check", "statistics data")
    policy.evolution = {
        "action": "PROMOTE",
        "reason": "independent",
        "reference_id": second["reference_id"],
        "independent": True,
        "complete_workflow": True,
        "draft": draft("statistical-testing", "statistics tests"),
    }
    promoted = manager.maintain(parent["cluster_id"], action="PROMOTE")
    # Add another capability to the promoted cluster, rather than the original parent.
    policy.next = [
        {
            "action": "ADD_REFERENCE",
            "reason": "same promoted scope",
            "has_information": True,
            "reference": draft("statistics-output", "statistics report"),
        }
    ]
    extra = accept(manager, "statistics-output", "statistics report")
    cid = promoted["target_cluster_id"]
    # Centroid prefers promoted cluster: default fallback also suffices if queue order differs.
    assert extra["cluster_id"] == cid
    state = manager.state()
    refs = [
        r
        for r in state["references"].values()
        if r["skill_id"] == promoted["target_skill_id"] and r["status"] == "active"
    ]
    groups = [
        {
            "master": draft(f"statistics-group-{i}", "statistics independent"),
            "reference_ids": [ref["id"]],
            "member_ids": [
                m["id"] for m in state["members"].values() if m["reference_id"] == ref["id"]
            ],
        }
        for i, ref in enumerate(refs)
    ]
    policy.evolution = {
        "action": "SPLIT",
        "reason": "independent groups",
        "independent": True,
        "groups": groups,
    }
    manager.maintain(cid, action="SPLIT")
    pointer = manager.state()["references"][second["reference_id"]]
    assert manager.state()["skills"][pointer["promoted_to"]]["status"] == "active"
    assert "statistics-group-" in manager.load_reference(second["reference_id"])


def test_candidate_raw_file_is_retained_before_integration(tmp_path):
    manager, _ = pool(tmp_path)
    candidate = manager.import_candidate(draft())
    raw = manager.files.load(candidate["file_path"])
    assert "name: data-analysis" in raw
    assert "Preserve supplied values." in raw
    assert candidate["status"] == "candidate"
    assert manager.discover("data") == []


def test_promoted_pointer_follows_repromotion(tmp_path):
    from backend.app.skills import SkillPoolSettings

    manager, policy = pool(tmp_path)
    manager.settings = SkillPoolSettings(cooldown_hours=0)
    parent = accept(manager)
    ref = accept(manager, "statistics-check", "statistics data")
    policy.evolution = {
        "action": "PROMOTE",
        "reason": "independent",
        "reference_id": ref["reference_id"],
        "independent": True,
        "complete_workflow": True,
        "draft": draft("statistical-testing", "statistics tests"),
    }
    one = manager.maintain(parent["cluster_id"], action="PROMOTE")
    state = manager.state()
    target_ref = state["references"][ref["reference_id"]]["promoted_reference_id"]
    policy.evolution = {
        "action": "PROMOTE",
        "reason": "refined independent task",
        "reference_id": target_ref,
        "independent": True,
        "complete_workflow": True,
        "draft": draft("statistics-research", "statistics research workflow"),
    }
    policy.next = [
        {
            "action": "NEW_CLUSTER",
            "reason": "original parent scope differs",
            "has_information": True,
        }
    ]
    two = manager.maintain(one["target_cluster_id"], action="PROMOTE")
    pointer = manager.state()["references"][ref["reference_id"]]
    assert pointer["status"] == "promoted"
    assert pointer["promoted_to"] == two["target_skill_id"]


def test_due_maintenance_consumes_event_and_periodic_signals(tmp_path):
    from backend.app.skills import SkillPoolSettings

    manager, _ = pool(tmp_path)
    manager.settings = SkillPoolSettings(maintenance_interval=1)
    result = accept(manager)
    for task_id in ["a", "b", "c"]:
        manager.record_usage(result["reference_id"], task_id, independent=True)
    signals = manager.status()["signals"]
    assert signals["event_clusters"] == [result["cluster_id"]]
    assert signals["periodic_due"]
    reviews = manager.maintain_due()
    assert len(reviews) == 1
    assert reviews[0]["action"] == "KEEP"
    assert reviews[0]["operation_id"]
    assert manager.maintain_due() == []
    assert not manager.status()["maintenance_due"]


def test_divergence_review_repeats_only_after_new_membership(tmp_path):
    from backend.app.skills import SkillPoolSettings

    manager, _ = pool(tmp_path)
    manager.settings = SkillPoolSettings(divergence_similarity=0.99)
    first = accept(manager)
    accept(manager, "statistics-check", "statistics data")
    assert manager.status()["signals"]["divergence_clusters"] == [first["cluster_id"]]
    assert len(manager.maintain_due()) == 1
    assert manager.status()["signals"]["divergence_clusters"] == []
    accept(manager, "statistics-output", "statistics reports")
    assert manager.status()["signals"]["divergence_clusters"] == [first["cluster_id"]]
