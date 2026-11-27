"""Paired ordinary-dialogue experiment. Run from the repo root as a Python module."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from statistics import mean
from time import perf_counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .memory import ask, open_database, record_feedback
from .models import ProceduralFeedback, ProceduralSkill
from .procedural_writes import process_event, process_sync


def mark(answer, case):
    normalized = answer.casefold()
    first, second = (normalized.find(label.casefold()) for label in case["markers"])
    return 0 <= first < second and any(term in normalized for term in case["topic"])


def run(cases, db, output):
    settings = get_settings()
    result = {
        "config": {
            "agent": settings.agent_model,
            "embedding": settings.bge_model,
            "cluster_threshold": settings.procedural_cluster_threshold,
            "recall_threshold": settings.procedural_recall_threshold,
            "context_tokens": settings.procedural_memory_context_tokens,
            "model_description": os.environ.get(
                "EVAL_MODEL_DESCRIPTION", settings.agent_model
            ),
            "seed": os.environ.get("EVAL_SERVER_SEED", "provider default"),
            "routing": "standalone ordinary dialogue",
            "scoring": "both requested headings in order and at least one topic term",
        },
        "learning": [],
        "snapshot": [],
        "evaluation": [],
        "summary": {},
    }

    def save():
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
        )

    expected = {}
    for case in cases["learning"]:
        message_id = record_feedback(db, case["feedback"], case["previous_answer"])
        started = perf_counter()
        status = process_event(db, message_id)
        entry = db.get(ProceduralFeedback, message_id)
        skill_id = entry.skill_id if entry else None
        expected[case["key"]] = (
            {"id": skill_id, "version": entry.version} if skill_id else None
        )
        result["learning"].append(
            {
                "key": case["key"],
                "status": status,
                "skill_id": skill_id,
                "action": entry.action if entry else None,
                "reason": entry.reason if entry else None,
                "latency_ms": round((perf_counter() - started) * 1000, 2),
            }
        )
        save()
    for skill in db.scalars(
        select(ProceduralSkill).order_by(ProceduralSkill.source_event_id)
    ):
        result["snapshot"].append(
            {
                "id": skill.id,
                "version": skill.version,
                "condition": skill.condition,
                "steps": skill.steps,
                "constraints": skill.constraints,
                "active": skill.active,
            }
        )
    started = perf_counter()
    process_sync(db)
    result["index_build_ms"] = round((perf_counter() - started) * 1000, 2)
    save()
    # Freeze the training history. V2 and write-only never read the frozen skills.
    for mode in ("off", "write", "flat", "rsi"):
        for case in cases["evaluation"]:
            started = perf_counter()
            answer = ask(db, case["question"], mode)
            used = answer.used_skills
            row = {
                "mode": mode,
                "id": case["id"],
                "question": case["question"],
                "answer": answer.answer,
                "expected": expected[case["key"]],
                "used": used,
                "hit": expected[case["key"]] in used,
                "success": mark(answer.answer, case),
                "tokens": answer.token_count,
                "degraded": answer.degraded,
                "query_ms": answer.query_ms,
                "failure": answer.failure,
                "elapsed_ms": round((perf_counter() - started) * 1000, 2),
            }
            result["evaluation"].append(row)
            save()
        rows = [r for r in result["evaluation"] if r["mode"] == mode]
        result["summary"][mode] = {
            k: round(mean(r[k] for r in rows), 3)
            for k in ("hit", "success", "tokens", "degraded", "query_ms", "elapsed_ms")
        }
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=Path(__file__).with_name("general_dialogue.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not get_settings().agent_model
        or not get_settings().agent_base_url
        or not get_settings().agent_api_key
    ):
        parser.error(
            "Configure AGENT_MODEL, AGENT_BASE_URL and AGENT_API_KEY before running the evaluation."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        engine = open_database(Path(scratch) / "experiment.db")
        try:
            with Session(engine) as db:
                run(json.loads(args.cases.read_text(encoding="utf-8")), db, args.output)
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()
