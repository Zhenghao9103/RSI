"""Explicitly invoked orchestration; no application lifecycle or Agent hooks."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml

from .clustering import cosine, ranked_clusters, recompute_centers
from .config import SkillPoolSettings
from .files import FileStore
from .index import SkillIndex
from .locking import writer_lock
from .policy import Policy, SkillTrigger, validate_draft
from .repository import Repository


def new_id():
    return uuid4().hex


def now():
    return datetime.now(UTC)


def parse_draft(value):
    if isinstance(value, str):
        match = re.match(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", value, re.S)
        if not match:
            raise ValueError("SKILL.md requires YAML frontmatter and a Markdown body")
        metadata = yaml.safe_load(match[1])
        if not isinstance(metadata, dict):
            raise ValueError("invalid skill metadata")
        value = {**metadata, "body": match[2]}
    if not isinstance(value, dict):
        raise ValueError("skill draft must be an object")
    return validate_draft(value)


class SkillManager:
    def __init__(self, root=None, *, settings=None, embed=None, llm=None, policy=None, index=None):
        self.settings = settings or SkillPoolSettings()
        self.root = Path(root or self.settings.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.repository = Repository(self.root)
        self.files = FileStore(self.root)
        self.index = index or SkillIndex(self.root, embed=embed)
        self.policy = policy or Policy(llm=llm, max_tokens=self.settings.llm_max_tokens)
        self.trigger = SkillTrigger(repeat_threshold=self.settings.repeat_threshold)

    def close(self):
        self.repository.dispose()

    def state(self):
        return self.repository.load()[1]

    def _snapshot(self):
        if self.repository.pending():
            raise RuntimeError("pending skill operation; run recover before writing")
        revision, state = self.repository.load()
        return revision, state

    def _touch(self, state, cluster_ids):
        meta = state["meta"]["pool"]
        meta["updates"] += 1
        meta["dirty_clusters"] = sorted(set(meta["dirty_clusters"]) | set(cluster_ids))

    def _refresh_master_indexes(self, state):
        for skill in state["skills"].values():
            if skill["status"] != "active":
                continue
            refs = [
                r
                for r in state["references"].values()
                if r["skill_id"] == skill["id"] and r["status"] in {"active", "promoted"}
            ]
            base = re.sub(
                r"\n<!-- skill-pool-references:start -->.*?"
                r"<!-- skill-pool-references:end -->\n?",
                "",
                skill["body"],
                flags=re.S,
            )
            lines = ["\n<!-- skill-pool-references:start -->", "## Reference loading conditions"]
            for ref in sorted(refs, key=lambda item: item["name"]):
                description = " ".join(ref["description"].split())
                lines.append(
                    f"- [{ref['name']}](references/{ref['name']}.md): "
                    f"Load only when the task requires: {description}"
                )
            lines.append("<!-- skill-pool-references:end -->\n")
            skill["body"] = base.rstrip() + "\n".join(lines)

    def _refresh_promoted_links(self, state):
        for ref in state["references"].values():
            if ref["status"] != "promoted":
                continue
            target_ref = state["references"].get(ref.get("promoted_reference_id"))
            seen = set()
            while target_ref and (
                (target_ref["status"] == "archived" and target_ref.get("merged_into"))
                or (target_ref["status"] == "promoted" and target_ref.get("promoted_reference_id"))
            ):
                if target_ref["id"] in seen:
                    raise ValueError("cyclic reference redirect")
                seen.add(target_ref["id"])
                redirect = (
                    target_ref.get("promoted_reference_id")
                    if target_ref["status"] == "promoted"
                    else target_ref["merged_into"]
                )
                target_ref = state["references"].get(redirect)
            target_id = target_ref["skill_id"] if target_ref else ref.get("promoted_to")
            target = state["skills"].get(target_id)
            if (
                not target
                or target["status"] != "active"
                or (target_ref and target_ref["status"] != "active")
            ):
                ref["status"] = "archived"
                continue
            ref["promoted_to"] = target_id
            if target_ref:
                ref["promoted_reference_id"] = target_ref["id"]
            ref["body"] = (
                f"## Promoted capability\nThis workflow is maintained by "
                f"[{target['name']}](../../{target['name']}/SKILL.md).\n"
                "Load that Skill for its complete current workflow.\n"
            )

    def _stamp(self, before, after, operation_id):
        for bucket, kind in [("skills", "skill"), ("references", "reference")]:
            for entity in after[bucket].values():
                old = before[bucket].get(entity["id"])
                fields = ["name", "description", "body", "status"]
                if kind == "reference":
                    fields.append("skill_id")
                changed = old is None or any(old.get(k) != entity.get(k) for k in fields)
                if not changed:
                    continue
                entity["version"] = (old["version"] if old else 0) + 1
                if old is None or any(old.get(k) != entity.get(k) for k in ["name", "description"]):
                    entity["embedding"] = self.index.encode(
                        entity["name"] + "\n" + entity["description"]
                    )
                else:
                    entity["embedding"] = old["embedding"]
                skill = entity if kind == "skill" else after["skills"][entity["skill_id"]]
                filename = "SKILL.md" if kind == "skill" else f"references/{entity['name']}.md"
                entity["file_path"] = (
                    f"versions/{operation_id}/{kind}-{entity['id']}/"
                    f"skills/{skill['name']}/{filename}"
                )

    def _validate(self, state):
        names = []
        for skill in state["skills"].values():
            parse_draft(skill)
            if skill["status"] == "active":
                names.append(skill["name"])
                cluster = state["clusters"].get(skill["cluster_id"])
                if (
                    not cluster
                    or cluster["status"] != "active"
                    or cluster["master_id"] != skill["id"]
                ):
                    raise ValueError("invalid active master/cluster relationship")
        if len(names) != len(set(names)):
            raise ValueError("duplicate active skill name")
        ref_names = set()
        for ref in state["references"].values():
            parse_draft(ref)
            if ref["skill_id"] not in state["skills"]:
                raise ValueError("reference has no parent")
            if ref["status"] in {"active", "promoted"}:
                pair = (ref["skill_id"], ref["name"])
                if pair in ref_names:
                    raise ValueError("duplicate reference name")
                ref_names.add(pair)
        for member in state["members"].values():
            cluster = state["clusters"].get(member["cluster_id"])
            ref = state["references"].get(member["reference_id"])
            if not cluster or not ref:
                raise ValueError("invalid member relationship")
            if cluster["status"] == "active" and (
                ref["status"] != "active" or ref["skill_id"] != cluster["master_id"]
            ):
                raise ValueError("member reference does not belong to its cluster")

    def _apply(self, revision, before, after, kind, reason=""):
        operation_id = new_id()
        self._refresh_promoted_links(after)
        self._refresh_master_indexes(after)
        changed_clusters = set()
        for mid in set(before["members"]) | set(after["members"]):
            old, new = before["members"].get(mid), after["members"].get(mid)
            if old != new:
                changed_clusters.update(m["cluster_id"] for m in [old, new] if m)
        recompute_centers(after, changed_clusters)
        self._stamp(before, after, operation_id)
        self._validate(after)
        self.repository.stage(operation_id, kind, revision, after, reason)
        self._finish(operation_id)
        return operation_id

    def _finish(self, operation_id):
        operation = self.repository.operation(operation_id)
        if operation["status"] == "committed":
            return operation
        try:
            self.files.prepare(operation["after"])
            self.files.materialize(operation["after"])
            self.index.sync(operation["before"], operation["after"])
            self.repository.publish(operation_id)
        except Exception as exc:
            self.repository.record_failure(operation_id, exc)
            raise
        return self.repository.operation(operation_id)

    def import_candidate(self, value, source=None):
        draft = parse_draft(value)
        digest = hashlib.sha256(json.dumps(draft, sort_keys=True).encode()).hexdigest()
        with writer_lock(self.root):
            revision, before = self._snapshot()
            for candidate in before["candidates"].values():
                if candidate["digest"] == digest:
                    return candidate
            state = deepcopy(before)
            candidate = {
                **draft,
                "id": new_id(),
                "status": "candidate",
                "embedding": None,
                "source": source or {},
                "digest": digest,
                "created_at": now().isoformat(),
            }
            candidate["file_path"] = f"candidates/{candidate['id']}/SKILL.md"
            state["candidates"][candidate["id"]] = candidate
            self._apply(revision, before, state, "IMPORT", "candidate stored")
            return candidate

    def generate_candidate(self, trace):
        self._validate_trace(trace)
        generated = self.policy.generate(trace)
        if generated.get("confidence", 1.0) < self.settings.generation_min_confidence:
            raise ValueError("candidate generation confidence below threshold")
        return self.import_candidate(generated, source={"kind": "trace", **deepcopy(trace)})

    @staticmethod
    def _validate_trace(trace):
        if (
            not isinstance(trace, dict)
            or not all(
                isinstance(trace.get(k), str) and trace[k].strip()
                for k in ["task_id", "user_query", "completion_status"]
            )
            or not isinstance(trace.get("trace_events"), list)
        ):
            raise ValueError(
                "Trace requires task_id, user_query, completion_status and trace_events"
            )
        if not isinstance(trace.get("final_answer"), str):
            raise ValueError("Trace requires final_answer")

    def observe(self, trace):
        self._validate_trace(trace)
        with writer_lock(self.root):
            revision, before = self._snapshot()
            if trace["task_id"] in before["tasks"]:
                old = before["tasks"][trace["task_id"]]
                return old["inspection"]
            # Discovery is read-only; absent indexed masters means uncovered.
            covered = bool(self.discover(trace["user_query"])) if before["skills"] else False
            inspection = self.trigger.inspect(
                trace, list(before["tasks"].values()), covered=covered
            )
            state = deepcopy(before)
            state["tasks"][trace["task_id"]] = {
                **deepcopy(trace),
                "id": trace["task_id"],
                "inspection": inspection,
                "fingerprint": inspection["fingerprint"],
            }
            self._apply(revision, before, state, "OBSERVE", "trace signals recorded")
        if inspection["triggered"] and self.settings.auto_accumulation:
            candidate = self.generate_candidate(trace)
            inspection = {**inspection, "candidate_id": candidate["id"]}
        return inspection

    def _new_cluster(self, state, draft, cooldown=None):
        draft = parse_draft(draft)
        cluster_id, skill_id = new_id(), new_id()
        state["skills"][skill_id] = {
            **draft,
            "id": skill_id,
            "cluster_id": cluster_id,
            "status": "active",
            "cooldown_until": cooldown,
        }
        state["clusters"][cluster_id] = {
            "id": cluster_id,
            "master_id": skill_id,
            "status": "active",
            "centroid": [],
            "sum_vector": [],
            "member_count": 0,
            "cooldown_until": cooldown,
        }
        return cluster_id, skill_id

    def _create_reference(self, state, skill_id, draft):
        draft = parse_draft(draft)
        ref_id = new_id()
        state["references"][ref_id] = {
            **draft,
            "id": ref_id,
            "skill_id": skill_id,
            "status": "active",
            "update_count": 0,
            "independent_task_ids": [],
        }
        return ref_id

    def _related_references(self, state, skill_id, vector):
        refs = [
            r
            for r in state["references"].values()
            if r["skill_id"] == skill_id and r["status"] == "active"
        ]
        return sorted(refs, key=lambda r: (-cosine(vector, r["embedding"]), r["id"]))[
            : self.settings.reference_limit
        ]

    def _select_destination(self, state, candidate, exclude=()):
        for _, cid in ranked_clusters(
            state,
            candidate["embedding"],
            self.settings.cluster_threshold,
            self.settings.cluster_limit,
            exclude,
        ):
            master = state["skills"][state["clusters"][cid]["master_id"]]
            refs = self._related_references(state, master["id"], candidate["embedding"])
            decision = self.policy.consolidate(candidate, master, refs)
            if decision["action"] != "NEW_CLUSTER":
                return cid, master["id"], decision
        decision = self.policy.consolidate(candidate, None, [])
        if decision["action"] != "NEW_CLUSTER" or not decision.get("master"):
            raise ValueError("new cluster decision requires a master draft")
        cid, sid = self._new_cluster(state, decision["master"])
        return cid, sid, decision

    def _accept_content(self, state, candidate, cid, sid, decision):
        action = decision["action"]
        if action == "IGNORE":
            return None
        if not decision.get("has_information", True):
            raise ValueError("accepted candidate must contain new information")
        if action == "UPDATE_REFERENCE":
            rid = decision.get("reference_id")
            ref = state["references"].get(rid)
            if not ref or ref["skill_id"] != sid or ref["status"] != "active":
                raise ValueError("invalid update reference target")
            ref.update(parse_draft(decision["reference"]))
            ref["update_count"] += 1
        elif action in {"ADD_REFERENCE", "NEW_CLUSTER"}:
            ref_draft = decision.get("reference") or {
                k: candidate[k] for k in ["name", "description", "body"]
            }
            rid = self._create_reference(state, sid, ref_draft)
        else:
            raise ValueError("invalid consolidation action")
        if decision.get("master") and action != "NEW_CLUSTER":
            state["skills"][sid].update(parse_draft(decision["master"]))
        return rid

    def integrate(self, candidate_id):
        with writer_lock(self.root):
            revision, before = self._snapshot()
            state = deepcopy(before)
            candidate = state["candidates"].get(candidate_id)
            if not candidate:
                raise ValueError("candidate not found")
            if candidate["status"] != "candidate":
                return candidate["result"]
            candidate["embedding"] = candidate["embedding"] or self.index.encode(
                candidate["name"] + "\n" + candidate["description"]
            )
            cid, sid, decision = self._select_destination(state, candidate)
            rid = self._accept_content(state, candidate, cid, sid, decision)
            action = decision["action"]
            candidate["status"] = "ignored" if action == "IGNORE" else "accepted"
            result = {
                "action": action,
                "candidate_id": candidate_id,
                "cluster_id": cid,
                "skill_id": sid,
                "reference_id": rid,
                "reason": decision.get("reason", ""),
            }
            candidate["result"] = deepcopy(result)
            if action != "IGNORE":
                state["members"][candidate_id] = {
                    "id": candidate_id,
                    "candidate_id": candidate_id,
                    "cluster_id": cid,
                    "reference_id": rid,
                    "embedding": candidate["embedding"],
                }
                self._touch(state, [cid])
            result["operation_id"] = self._apply(
                revision, before, state, action, decision.get("reason", "")
            )
            return result

    def record_usage(self, reference_id, task_id, *, independent=False):
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be nonempty")
        with writer_lock(self.root):
            revision, before = self._snapshot()
            state = deepcopy(before)
            ref = state["references"].get(reference_id)
            if not ref or ref["status"] != "active":
                raise ValueError("active reference not found")
            if independent and task_id not in ref["independent_task_ids"]:
                ref["independent_task_ids"].append(task_id)
                cid = state["skills"][ref["skill_id"]]["cluster_id"]
                dirty = state["meta"]["pool"]["dirty_clusters"]
                if cid not in dirty:
                    dirty.append(cid)
                self._apply(revision, before, state, "USAGE", "independent task demand")
            return ref

    def maintain(self, cluster_id, *, action=None):
        from .evolution import evolve

        with writer_lock(self.root):
            revision, before = self._snapshot()
            state = deepcopy(before)
            cluster = state["clusters"].get(cluster_id)
            if not cluster or cluster["status"] != "active":
                raise ValueError("active cluster not found")
            master = state["skills"][cluster["master_id"]]
            refs = [
                r
                for r in state["references"].values()
                if r["skill_id"] == master["id"] and r["status"] == "active"
            ]
            members = [m for m in state["members"].values() if m["cluster_id"] == cluster_id]
            signals = self._maintenance_signals(state)
            neighbors = ranked_clusters(
                state,
                cluster["centroid"],
                self.settings.cluster_threshold,
                self.settings.cluster_limit,
                exclude=[cluster_id],
            )
            signals["neighbor_clusters"] = []
            for _, cid in neighbors:
                neighbor = state["clusters"][cid]
                neighbor_master = state["skills"][neighbor["master_id"]]
                signals["neighbor_clusters"].append(
                    {
                        "cluster": neighbor,
                        "master": neighbor_master,
                        "references": self._related_references(
                            state, neighbor_master["id"], cluster["centroid"]
                        ),
                    }
                )
            decision = self.policy.maintenance(
                cluster, master, refs, members, signals, requested_action=action
            )
            if action and decision["action"] != action:
                if decision["action"] == "KEEP":
                    pass
                else:
                    raise ValueError("maintenance decision differs from requested action")
            result = evolve(self, state, cluster_id, decision)
            if result["action"] != "KEEP":
                self._touch(state, result.get("cluster_ids", [cluster_id]))
            for ref in refs:
                ref["reviewed_update_count"] = ref["update_count"]
                ref["reviewed_independent_hits"] = len(ref["independent_task_ids"])
            state["meta"]["pool"]["last_maintenance_update"] = state["meta"]["pool"]["updates"]
            state["meta"]["pool"].setdefault("reviewed_member_signatures", {})[cluster_id] = (
                self._member_signature(state, cluster_id)
            )
            dirty = state["meta"]["pool"]["dirty_clusters"]
            if cluster_id in dirty:
                dirty.remove(cluster_id)
            result["operation_id"] = self._apply(
                revision, before, state, result["action"], decision.get("reason", "")
            )
            return result

    def _maintenance_signals(self, state):
        settings = self.settings
        event_clusters = set()
        for ref in state["references"].values():
            if ref["status"] == "active" and (
                len(ref["independent_task_ids"]) - ref.get("reviewed_independent_hits", 0)
                >= settings.independent_hit_threshold
                or ref["update_count"] - ref.get("reviewed_update_count", 0)
                >= settings.reference_update_threshold
            ):
                event_clusters.add(state["skills"][ref["skill_id"]]["cluster_id"])
        meta = state["meta"]["pool"]
        divergence_clusters = []
        reviewed = meta.get("reviewed_member_signatures", {})
        for cluster in state["clusters"].values():
            if cluster["status"] != "active" or cluster["member_count"] < 2:
                continue
            signature = self._member_signature(state, cluster["id"])
            if reviewed.get(cluster["id"]) == signature:
                continue
            similarities = [
                cosine(m["embedding"], cluster["centroid"])
                for m in state["members"].values()
                if m["cluster_id"] == cluster["id"]
            ]
            if min(similarities) < settings.divergence_similarity:
                divergence_clusters.append(cluster["id"])
                event_clusters.add(cluster["id"])
        last = meta.get("last_maintenance_update", 0)
        periodic = meta["updates"] - last >= settings.maintenance_interval
        return {
            "event_clusters": sorted(event_clusters),
            "periodic_due": periodic,
            "divergence_clusters": divergence_clusters,
            "dirty_clusters": meta["dirty_clusters"],
            "updates": meta["updates"],
        }

    @staticmethod
    def _member_signature(state, cluster_id):
        ids = sorted(m["id"] for m in state["members"].values() if m["cluster_id"] == cluster_id)
        return hashlib.sha256(json.dumps(ids).encode()).hexdigest()

    def maintain_due(self):
        state = self.state()
        signals = self._maintenance_signals(state)
        ids = set(signals["event_clusters"])
        if signals["periodic_due"]:
            ids.update(signals["dirty_clusters"])
        results = []
        for cid in sorted(ids):
            cluster = self.state()["clusters"].get(cid)
            if not cluster or cluster["status"] != "active":
                continue
            if (
                cluster.get("cooldown_until")
                and datetime.fromisoformat(cluster["cooldown_until"]) > now()
            ):
                results.append({"cluster_id": cid, "action": "KEEP", "reason": "cooldown"})
                continue
            results.append({"cluster_id": cid, **self.maintain(cid)})
        return results

    def status(self):
        revision, state = self.repository.load()
        signals = self._maintenance_signals(state)
        return {
            "root": str(self.root),
            "revision": revision,
            "active_skills": sum(s["status"] == "active" for s in state["skills"].values()),
            "active_references": sum(r["status"] == "active" for r in state["references"].values()),
            "candidates": sum(c["status"] == "candidate" for c in state["candidates"].values()),
            "clusters": list(state["clusters"].values()),
            "candidate_ids": list(state["candidates"]),
            "skills": [
                {k: s[k] for k in ["id", "name", "description", "status", "cluster_id"]}
                for s in state["skills"].values()
            ],
            "references": [
                {k: r[k] for k in ["id", "name", "skill_id", "status"]}
                for r in state["references"].values()
            ],
            "pending_operations": [op["id"] for op in self.repository.pending()],
            "maintenance_due": bool(signals["event_clusters"] or signals["periodic_due"]),
            "signals": signals,
            "auto_accumulation": self.settings.auto_accumulation,
        }

    def discover(self, query, limit=None):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonempty")
        return self.index.discover(
            self.state(),
            query,
            limit=limit or self.settings.discovery_limit,
            min_similarity=self.settings.discovery_threshold,
        )

    def discover_references(self, skill_id, query, limit=None):
        return self.index.discover(
            self.state(),
            query,
            limit=limit or self.settings.reference_limit,
            min_similarity=self.settings.discovery_threshold,
            skill_id=skill_id,
            kind="reference",
        )

    def load_master(self, skill_id):
        return self._load("skills", skill_id, {"active"})

    def load_reference(self, reference_id):
        return self._load("references", reference_id, {"active", "promoted"})

    def _load(self, bucket, entity_id, statuses):
        entity = self.state()[bucket].get(entity_id)
        if not entity or entity["status"] not in statuses:
            raise ValueError("loadable skill resource not found")
        return self.files.load(entity["file_path"])

    def history(self):
        return self.repository.history()

    def recover(self):
        with writer_lock(self.root):
            return [self._finish(op["id"]) for op in self.repository.pending()]

    def rollback(self, operation_id):
        with writer_lock(self.root):
            revision, before = self._snapshot()
            operation = self.repository.operation(operation_id)
            if operation["status"] != "committed":
                raise ValueError("only committed operations can be rolled back")
            # Content equality permits stepping backwards after a previous rollback;
            # newer unrelated work must be rolled back first as well.
            if before != operation["after"]:
                raise ValueError("later dependent operations exist; rollback in reverse order")
            target = deepcopy(operation["before"])
            # Restore historical file/vector revisions exactly, without re-embedding.
            rollback_id = new_id()
            self.repository.stage(
                rollback_id, "ROLLBACK", revision, target, f"rollback {operation_id}"
            )
            self._finish(rollback_id)
            return {
                "action": "ROLLBACK",
                "operation_id": rollback_id,
                "restored_from": operation_id,
            }
