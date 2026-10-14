"""Opt-in offline demonstration and separately invoked real dependency smoke."""

import json
import math
import subprocess
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory


class SmokeFailure(RuntimeError):
    """Only bounded diagnostic identifiers may cross the isolated worker boundary."""

    def __init__(self, diagnostic):
        stages = {
            "demo",
            "smoke",
            "embedding",
            "index",
            "discovery",
            "llm_generate",
            "llm_consolidation",
            "loading",
            "worker",
        }
        categories = {
            "connection_error",
            "provider_error",
            "configuration_error",
            "timeout",
            "timeout_error",
            "rate_limit",
            "rate_limit_error",
            "authentication_error",
            "invalid_json",
            "empty_response",
            "context_window_exceeded",
            "validation_error",
            "unknown",
        }
        types = {
            "ModelClientError",
            "ModelConfigurationError",
            "ValueError",
            "RuntimeError",
            "AssertionError",
            "OSError",
            "TimeoutExpired",
            "PermissionError",
            "UnknownError",
        }
        self.diagnostic = {
            "stage": diagnostic.get("stage") if diagnostic.get("stage") in stages else "worker",
            "error_type": diagnostic.get("error_type")
            if diagnostic.get("error_type") in types
            else "UnknownError",
            "category": diagnostic.get("category")
            if diagnostic.get("category") in categories
            else "unknown",
        }
        super().__init__(json.dumps(self.diagnostic, ensure_ascii=False))


def _at_stage(stage, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        category = getattr(
            exc,
            "category",
            "configuration_error"
            if type(exc).__name__ == "ModelConfigurationError"
            else "validation_error"
            if isinstance(exc, ValueError)
            else "unknown",
        )
        raise SmokeFailure(
            {"stage": stage, "error_type": type(exc).__name__, "category": category}
        ) from exc


def _worker_entry(root, mode, with_llm=False):
    try:
        result = _run_demo(root) if mode == "demo" else _run_smoke(root, with_llm=with_llm)
        print("\nSKILL_POOL_WORKER_RESULT=" + json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        diagnostic = (
            exc.diagnostic
            if isinstance(exc, SmokeFailure)
            else SmokeFailure(
                {
                    "stage": mode,
                    "error_type": type(exc).__name__,
                    "category": getattr(exc, "category", "unknown"),
                }
            ).diagnostic
        )
        print("\nSKILL_POOL_WORKER_ERROR=" + json.dumps(diagnostic, ensure_ascii=False))
        return 1


def _draft(name, description, step="Inspect the supplied data and record observations."):
    return {
        "name": name,
        "description": description,
        "body": "## Applicability\nUse for the named workflow.\n\n## Steps\n1. "
        + step
        + "\n2. Verify the result.\n\n## Constraints\nPreserve supplied evidence.",
    }


def _demo_embedding(text):
    """Only demo uses these vectors; production imports the real BGE-M3 encoder."""
    if "writing" in text.lower():
        return [0.0, 0.0, 1.0]
    if "statistics" in text.lower():
        return [0.8, 0.6, 0.0]
    return [1.0, 0.0, 0.0]


class _DemoPolicy:
    def __init__(self):
        self.update_reference = None
        self.evolution = None

    def consolidate(self, candidate, master, references):
        if master is None:
            return {
                "action": "NEW_CLUSTER",
                "reason": "Demonstration: new workflow",
                "has_information": True,
                "master": _draft(candidate["name"], candidate["description"]),
                "reference": _draft("workflow", candidate["description"]),
            }
        if candidate["name"] == "statistical-workflow":
            return {
                "action": "NEW_CLUSTER",
                "reason": "Demonstration: independent boundary",
                "has_information": True,
            }
        if self.update_reference:
            reference_id, self.update_reference = self.update_reference, None
            return {
                "action": "UPDATE_REFERENCE",
                "reason": "Demonstration: adds quality checks",
                "has_information": True,
                "reference_id": reference_id,
                "reference": _draft("workflow", "Analyze data with quality checks"),
            }
        return {
            "action": "ADD_REFERENCE",
            "reason": "Demonstration: complementary method",
            "has_information": True,
            "reference": _draft(candidate["name"], candidate["description"]),
        }

    def maintenance(self, *args, **kwargs):
        return deepcopy(self.evolution or {"action": "KEEP", "reason": "Demonstration: stable"})


def _check_centroids(state, cluster_ids):
    for cluster_id in cluster_ids:
        members = [m for m in state["members"].values() if m["cluster_id"] == cluster_id]
        if len(members) != state["clusters"][cluster_id]["member_count"] or not members:
            return False
        total = [
            sum(m["embedding"][i] for m in members) for i in range(len(members[0]["embedding"]))
        ]
        norm = math.sqrt(sum(v * v for v in total))
        if any(
            abs(a - b / norm) > 1e-9
            for a, b in zip(state["clusters"][cluster_id]["centroid"], total, strict=True)
        ):
            return False
    return True


def _run_demo(root):
    from . import SkillManager

    policy = _DemoPolicy()
    manager = SkillManager(root=root, embed=_demo_embedding, policy=policy)
    checks = {}
    try:

        def accept(name, description):
            candidate = manager.import_candidate(
                _draft(name, description), source={"type": "offline-demo"}
            )
            return manager.integrate(candidate["id"])

        first = accept("data-analysis", "Analyze data")
        second = accept("statistics-checks", "statistics data checks")
        checks["similar_candidates_same_cluster"] = first["cluster_id"] == second["cluster_id"]
        policy.update_reference = first["reference_id"]
        third = accept("data-quality", "Analyze data quality")
        checks["third_candidate_updates_reference"] = (
            third["action"] == "UPDATE_REFERENCE"
            and third["reference_id"] == first["reference_id"]
            and manager.state()["references"][first["reference_id"]]["update_count"] == 1
        )
        prior = manager.state()
        policy.evolution = {
            "action": "PROMOTE",
            "reason": "Demonstration: standalone statistics workflow",
            "independent": True,
            "complete_workflow": True,
            "reference_id": second["reference_id"],
            "draft": _draft("statistical-workflow", "statistics independent tests"),
        }
        promoted = manager.maintain(first["cluster_id"], action="PROMOTE")
        after_promotion = manager.state()
        checks["promotion_migrates_members"] = (
            len(prior["members"]) == len(after_promotion["members"])
            and after_promotion["clusters"][first["cluster_id"]]["member_count"] == 2
            and after_promotion["clusters"][promoted["target_cluster_id"]]["member_count"] == 1
            and after_promotion["references"][second["reference_id"]]["status"] == "promoted"
        )
        pointer = manager.load_reference(second["reference_id"])
        target_name = after_promotion["skills"][promoted["target_skill_id"]]["name"]
        checks["promoted_reference_is_collaboration_pointer"] = (
            f"../../{target_name}/SKILL.md" in pointer and "## Steps" not in pointer
        )

        # An unrelated writing cluster can split without bypassing the promotion cooldown.
        writing = accept("writing-review", "writing manuscript review")
        editing = accept("writing-editing", "writing manuscript editing")
        before_split = manager.state()
        members = [
            m for m in before_split["members"].values() if m["cluster_id"] == writing["cluster_id"]
        ]
        groups = []
        for result, name in [(writing, "writing-review"), (editing, "writing-editing")]:
            groups.append(
                {
                    "master": _draft(name, name.replace("-", " ")),
                    "reference_ids": [result["reference_id"]],
                    "member_ids": [
                        m["id"] for m in members if m["reference_id"] == result["reference_id"]
                    ],
                }
            )
        policy.evolution = {
            "action": "SPLIT",
            "reason": "Demonstration: separate writing workflows",
            "independent": True,
            "groups": groups,
        }
        split = manager.maintain(writing["cluster_id"], action="SPLIT")
        after_split = manager.state()
        checks["split_preserves_vectors_and_centroids"] = (
            {m["id"]: m["embedding"] for m in before_split["members"].values()}
            == {m["id"]: m["embedding"] for m in after_split["members"].values()}
            and _check_centroids(after_split, split["cluster_ids"])
            and after_split["clusters"][writing["cluster_id"]]["status"] == "archived"
        )
        before_hits = sorted(hit["id"] for hit in manager.discover("writing review"))
        checks["archived_master_excluded"] = writing["skill_id"] not in before_hits
        checks["real_storage_used"] = bool(manager.history()) and any(Path(root).rglob("SKILL.md"))
    finally:
        manager.close()
    reopened = SkillManager(root=root, embed=_demo_embedding, policy=policy)
    try:
        checks["reopen_discovery_consistent"] = before_hits == sorted(
            hit["id"] for hit in reopened.discover("writing review")
        )
        status = reopened.status()
    finally:
        reopened.close()
    if not all(checks.values()):
        raise AssertionError(
            "Offline demonstration failed: " + ", ".join(k for k, v in checks.items() if not v)
        )
    return {
        "mode": "offline-demo",
        "root": str(root),
        "checks": checks,
        "status": status,
        "note": (
            "Deterministic demonstration embeddings and decisions; real SQLite/files/Chroma. "
            "Not model-quality evidence."
        ),
    }


def run_demo(root=None):
    """Run on an empty explicit directory, or clean up an automatically isolated temporary root."""
    if root is not None:
        root = Path(root)
        if root.exists() and any(root.iterdir()):
            raise ValueError("Demo requires an empty independent directory")
        return _run_demo(root)
    return _run_isolated("demo")


def run_smoke(with_llm=False):
    """Use real BGE-M3 and Chroma; optionally generate with the configured Agent model."""
    return _run_isolated("smoke", with_llm=with_llm)


def _run_isolated(mode, with_llm=False):
    # Chroma keeps Windows file handles in a process-wide client cache. A short-lived
    # worker releases only its own handles before the parent removes its temporary root.
    script = (
        "import json, sys; from pathlib import Path; "
        "from backend.app.skills.demo import _worker_entry; "
        "sys.stdout.reconfigure(encoding='utf-8'); "
        "sys.stderr.reconfigure(encoding='utf-8'); "
        "sys.exit(_worker_entry(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'true'))"
    )
    with TemporaryDirectory(prefix=f"paper-agent-skill-{mode}-") as directory:
        worker = subprocess.run(
            [sys.executable, "-c", script, directory, mode, str(with_llm).lower()],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
        if worker.returncode:
            marker = "SKILL_POOL_WORKER_ERROR="
            diagnostic = (
                json.loads(worker.stdout.rsplit(marker, 1)[1].strip())
                if marker in worker.stdout
                else {}
            )
            raise SmokeFailure(diagnostic)
        marker = "SKILL_POOL_WORKER_RESULT="
        if marker not in worker.stdout:
            raise RuntimeError(f"Isolated {mode} worker returned no result")
        result = json.loads(worker.stdout.rsplit(marker, 1)[1].strip())
    result["temporary_storage_removed"] = True
    return result


def _run_smoke(root, with_llm=False):
    from backend.app.rag.vector_store import embed_text

    from . import SkillManager
    from .index import SkillIndex
    from .repository import empty_state

    with nullcontext(root) as directory:
        index = SkillIndex(Path(directory), embed=embed_text)
        vector = _at_stage(
            "embedding",
            index.encode,
            "document-review Review an academic document with supplied evidence",
        )
        state = empty_state()
        state["skills"]["smoke-master"] = {
            "id": "smoke-master",
            "name": "document-review",
            "description": "Review an academic document with supplied evidence",
            "body": "Inspect evidence.",
            "version": 1,
            "status": "active",
            "embedding": vector,
        }
        _at_stage("index", index.sync, empty_state(), state)
        hits = _at_stage(
            "discovery",
            index.discover,
            state,
            "document-review Review an academic document with supplied evidence",
        )
        if not hits or hits[0]["id"] != "smoke-master":
            raise AssertionError("Real embedding/index round trip failed")
        result = {
            "mode": "real-dependency-smoke",
            "embedding_model": "BAAI/bge-m3",
            "embedding_dimensions": len(vector),
            "embedding_norm": math.sqrt(sum(v * v for v in vector)),
            "chroma_round_trip": True,
            "llm": {"executed": False},
            "temporary_storage_removed": True,
        }
        if with_llm:
            manager = SkillManager(root=Path(directory) / "llm")
            try:
                candidate = _at_stage(
                    "llm_generate",
                    manager.generate_candidate,
                    {
                        "task_id": "explicit-smoke",
                        "user_query": "Save the repaired indexing workflow as a reusable skill.",
                        "trace_events": [
                            {
                                "tool": "inspect_index",
                                "status": "error",
                                "error": "Stale document index",
                            },
                            {
                                "tool": "rebuild_index",
                                "args": {"preserve_sources": True},
                                "status": "success",
                            },
                            {
                                "tool": "verify_index",
                                "status": "success",
                                "result": "Document IDs and versions match",
                            },
                        ],
                        "final_answer": (
                            "Rebuilt the stale index and verified document IDs and versions "
                            "while preserving source files."
                        ),
                        "completion_status": "success",
                    },
                )
                integrated = _at_stage("llm_consolidation", manager.integrate, candidate["id"])
                master_id = integrated["skill_id"]
                query = candidate["name"] + "\n" + candidate["description"]
                hits = _at_stage("discovery", manager.discover, query)
                master = _at_stage("loading", manager.load_master, master_id)
                reference = _at_stage("loading", manager.load_reference, integrated["reference_id"])
                if master_id not in {hit["id"] for hit in hits} or not master or not reference:
                    raise SmokeFailure(
                        {
                            "stage": "discovery",
                            "error_type": "AssertionError",
                            "category": "validation_error",
                        }
                    )
                result["llm"] = {
                    "executed": True,
                    "candidate_id": candidate["id"],
                    "name": candidate["name"],
                    "integration_action": integrated["action"],
                    "cluster_id": integrated["cluster_id"],
                    "discover_load_round_trip": True,
                }
            finally:
                manager.close()
        return result
