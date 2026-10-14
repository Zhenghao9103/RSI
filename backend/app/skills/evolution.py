"""Validated local structure transformations; publication belongs to Manager."""

from copy import deepcopy
from datetime import datetime, timedelta

from .manager import new_id, now, parse_draft


def _cooldown(cluster):
    value = cluster.get("cooldown_until")
    if value and datetime.fromisoformat(value) > now():
        raise ValueError("structure is in cooldown")


def _reference(state, cid, rid):
    ref = state["references"].get(rid)
    if (
        not ref
        or ref["status"] != "active"
        or ref["skill_id"] != state["clusters"][cid]["master_id"]
    ):
        raise ValueError("active reference does not belong to maintenance cluster")
    return ref


def _archive_cluster(state, cid):
    cluster = state["clusters"][cid]
    cluster.update(status="archived", centroid=[], sum_vector=[], member_count=0)
    state["skills"][cluster["master_id"]]["status"] = "archived"
    for ref in state["references"].values():
        if ref["skill_id"] == cluster["master_id"]:
            ref["status"] = "archived"


def _promote(manager, state, cid, decision, until):
    if decision.get("independent") is not True or decision.get("complete_workflow") is not True:
        raise ValueError("promotion requires independent and complete workflow")
    ref = _reference(state, cid, decision.get("reference_id"))
    candidate = {
        **parse_draft(decision["draft"]),
        "id": new_id(),
        "status": "accepted",
        "source": {"kind": "promotion", "reference_id": ref["id"]},
        "created_at": now().isoformat(),
    }
    candidate["digest"] = "promotion:" + candidate["id"]
    candidate["file_path"] = f"candidates/{candidate['id']}/SKILL.md"
    candidate["embedding"] = manager.index.encode(
        candidate["name"] + "\n" + candidate["description"]
    )
    target_cid, target_sid, consolidation = manager._select_destination(
        state, candidate, exclude=[cid]
    )
    _cooldown(state["clusters"][target_cid])
    if consolidation["action"] == "IGNORE":
        refs = [
            r
            for r in state["references"].values()
            if r["skill_id"] == target_sid and r["status"] == "active"
        ]
        target_rid = consolidation.get("reference_id")
        if target_rid is None and len(refs) == 1:
            target_rid = refs[0]["id"]
        _reference(state, target_cid, target_rid)
    else:
        target_rid = manager._accept_content(
            state, candidate, target_cid, target_sid, consolidation
        )
    moved = []
    for member in state["members"].values():
        if member["cluster_id"] == cid and member["reference_id"] == ref["id"]:
            member.update(cluster_id=target_cid, reference_id=target_rid)
            moved.append(member["id"])
    if not moved:
        raise ValueError("promotion reference has no historical members")
    target_skill = state["skills"][target_sid]
    ref.update(
        status="promoted",
        promoted_to=target_sid,
        promoted_reference_id=target_rid,
        body=f"## Promoted capability\nThis workflow is maintained by "
        f"[{target_skill['name']}](../../{target_skill['name']}/SKILL.md).\n"
        "Load that Skill for its complete current workflow.\n",
    )
    candidate["result"] = {
        "action": "PROMOTE",
        "cluster_id": target_cid,
        "skill_id": target_sid,
        "reference_id": target_rid,
    }
    state["candidates"][candidate["id"]] = candidate
    if decision.get("master"):
        state["skills"][state["clusters"][cid]["master_id"]].update(parse_draft(decision["master"]))
    for affected in [cid, target_cid]:
        state["clusters"][affected]["cooldown_until"] = until
        state["skills"][state["clusters"][affected]["master_id"]]["cooldown_until"] = until
    return {
        "target_cluster_id": target_cid,
        "target_skill_id": target_sid,
        "reference_id": ref["id"],
        "moved_member_ids": moved,
        "cluster_ids": [cid, target_cid],
    }


def _split(manager, state, cid, decision, until):
    groups = decision.get("groups")
    if decision.get("independent") is not True or not isinstance(groups, list) or len(groups) < 2:
        raise ValueError("split requires independent groups")
    master_id = state["clusters"][cid]["master_id"]
    refs = {
        r["id"]
        for r in state["references"].values()
        if r["skill_id"] == master_id and r["status"] == "active"
    }
    members = {m["id"] for m in state["members"].values() if m["cluster_id"] == cid}
    listed_refs = [rid for group in groups for rid in group.get("reference_ids", [])]
    listed_members = [mid for group in groups for mid in group.get("member_ids", [])]
    if (
        set(listed_refs) != refs
        or len(listed_refs) != len(refs)
        or set(listed_members) != members
        or len(listed_members) != len(members)
    ):
        raise ValueError("split partition must contain every reference and member exactly once")
    for group in groups:
        if not group["reference_ids"] or not group["member_ids"]:
            raise ValueError("split partition groups cannot be empty")
        if any(
            state["members"][mid]["reference_id"] not in group["reference_ids"]
            for mid in group["member_ids"]
        ):
            raise ValueError("split partition must preserve reference/member relationships")
    cluster_ids = []
    for group in groups:
        new_cid, sid = manager._new_cluster(state, group["master"], until)
        cluster_ids.append(new_cid)
        for rid in group["reference_ids"]:
            state["references"][rid]["skill_id"] = sid
        for mid in group["member_ids"]:
            state["members"][mid]["cluster_id"] = new_cid
    # Previously promoted pointers stay as historical resources of the old master.
    _archive_cluster(state, cid)
    return {"cluster_ids": cluster_ids}


def _merge_clusters(manager, state, cid, decision, until):
    target_cid = decision["target_cluster_id"]
    if target_cid == cid or target_cid not in state["clusters"]:
        raise ValueError("invalid merge target")
    target = state["clusters"][target_cid]
    if target["status"] != "active":
        raise ValueError("merge target is not active")
    _cooldown(target)
    source_sid = state["clusters"][cid]["master_id"]
    target_sid = target["master_id"]
    # Each source Reference receives its own content relation decision. Moving
    # whole files without this check would merely relocate Reference bloat.
    refs = [
        r
        for r in state["references"].values()
        if r["skill_id"] == source_sid and r["status"] == "active"
    ]
    mapping = {}
    for ref in refs:
        candidate = deepcopy(ref)
        master = state["skills"][target_sid]
        related = manager._related_references(state, target_sid, candidate["embedding"])
        relationship = manager.policy.consolidate(candidate, master, related)
        if relationship["action"] == "NEW_CLUSTER":
            raise ValueError("merge content boundary rejected")
        if relationship["action"] == "IGNORE":
            rid = relationship.get("reference_id")
            if rid is None and len(related) == 1:
                rid = related[0]["id"]
            _reference(state, target_cid, rid)
        else:
            rid = manager._accept_content(state, candidate, target_cid, target_sid, relationship)
        mapping[ref["id"]] = rid
    for member in state["members"].values():
        if member["cluster_id"] == cid:
            member.update(cluster_id=target_cid, reference_id=mapping[member["reference_id"]])
    if decision.get("master"):
        state["skills"][target_sid].update(parse_draft(decision["master"]))
    _archive_cluster(state, cid)
    state["skills"][source_sid]["merged_into"] = target_sid
    for source_rid, target_rid in mapping.items():
        state["references"][source_rid]["merged_into"] = target_rid
    target["cooldown_until"] = until
    state["skills"][target_sid]["cooldown_until"] = until
    return {
        "target_cluster_id": target_cid,
        "cluster_ids": [target_cid],
        "reference_mapping": mapping,
    }


def _merge_references(state, cid, decision):
    target = _reference(state, cid, decision.get("target_reference_id"))
    source_ids = decision.get("source_reference_ids")
    if (
        not isinstance(source_ids, list)
        or not source_ids
        or len(set(source_ids)) != len(source_ids)
    ):
        raise ValueError("invalid reference merge sources")
    if target["id"] in source_ids:
        raise ValueError("reference merge target cannot be a source")
    for rid in source_ids:
        _reference(state, cid, rid)
    target.update(parse_draft(decision["reference"]))
    target["update_count"] += 1
    for rid in source_ids:
        ref = state["references"][rid]
        ref.update(status="archived", merged_into=target["id"])
        for member in state["members"].values():
            if member["reference_id"] == rid:
                member["reference_id"] = target["id"]
        target["independent_task_ids"] = list(
            dict.fromkeys(target["independent_task_ids"] + ref["independent_task_ids"])
        )
    return {"reference_id": target["id"], "cluster_ids": [cid]}


def evolve(manager, state, cid, decision):
    action = decision.get("action")
    if action == "KEEP":
        return {"action": "KEEP", "reason": decision.get("reason", "")}
    if action not in {"PROMOTE", "SPLIT", "MERGE", "REFACTOR", "ARCHIVE"}:
        raise ValueError("invalid evolution action")
    if action in {"PROMOTE", "SPLIT", "MERGE"}:
        _cooldown(state["clusters"][cid])
    until = (now() + timedelta(hours=manager.settings.cooldown_hours)).isoformat()
    if action == "PROMOTE":
        result = _promote(manager, state, cid, decision, until)
    elif action == "SPLIT":
        result = _split(manager, state, cid, decision, until)
    elif action == "MERGE":
        result = (
            _merge_clusters(manager, state, cid, decision, until)
            if decision.get("target_cluster_id")
            else _merge_references(state, cid, decision)
        )
    elif action == "REFACTOR":
        ref = _reference(state, cid, decision.get("reference_id"))
        ref.update(parse_draft(decision["reference"]))
        ref["update_count"] += 1
        result = {"reference_id": ref["id"], "cluster_ids": [cid]}
    else:
        rid = decision.get("reference_id")
        if rid:
            ref = _reference(state, cid, rid)
            ref["status"] = "archived"
            # Remove from active clustering, retaining immutable original vectors
            # in the candidate and operation snapshots for rollback.
            for mid in [m["id"] for m in state["members"].values() if m["reference_id"] == rid]:
                del state["members"][mid]
        else:
            if decision.get("cluster_id") != cid:
                raise ValueError("archive cluster differs from maintenance target")
            _archive_cluster(state, cid)
        result = {"cluster_ids": [cid]}
    return {"action": action, "reason": decision.get("reason", ""), **result}
