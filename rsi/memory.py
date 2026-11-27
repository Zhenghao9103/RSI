from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from sqlalchemy import URL, create_engine, select
from sqlalchemy.orm import Session

from . import answering, procedural_retrieval, procedural_writes
from .config import get_settings
from .models import Base, ChatMessage, ChatSession, ProceduralSkill


@dataclass
class Answer:
    answer: str
    used_skills: list[dict] = field(default_factory=list)
    token_count: int = 0
    degraded: bool = False
    query_ms: float = 0
    failure: str | None = None


def open_database(database: str | Path):
    path = str(database)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(URL.create("sqlite", database=path))
    try:
        Base.metadata.create_all(engine)
    except Exception:
        engine.dispose()
        raise
    return engine


def record_feedback(db: Session, feedback: str, previous_answer: str = "") -> int:
    session = ChatSession()
    db.add(session)
    db.flush()
    if previous_answer:
        db.add(
            ChatMessage(
                role="assistant", session_id=session.id, content=previous_answer
            )
        )
        db.flush()
    message = ChatMessage(role="user", session_id=session.id, content=feedback)
    db.add(message)
    db.flush()
    if procedural_writes.eligible(feedback):
        procedural_writes.enqueue(db, message.id)
    message_id = message.id
    db.commit()
    return message_id


def ask(db: Session, question: str, mode: str) -> Answer:
    settings = get_settings()
    result = Answer("")
    section = ""
    if mode in {"flat", "rsi"}:
        started = perf_counter()
        try:
            skills, result.degraded = procedural_retrieval.retrieve_skills(
                db, question, mode
            )
            context = answering.prepare_procedural_context(
                skills,
                settings.procedural_memory_context_tokens,
                settings.agent_model or "",
            )
            section = context.section
            result.used_skills = context.used_skills
            result.token_count = context.token_count
            # Keep an index rebuilt on demand across invocations.
            db.commit()
        except Exception as exc:
            db.rollback()
            result.degraded = True
            result.failure = type(exc).__name__
        result.query_ms = round((perf_counter() - started) * 1000, 3)
    result.answer = answering.answer_general(question, section)
    if result.answer == "当前无法生成回答。":
        result.failure = "generation_failed"
    return result


class Memory:
    """Own a local database; each public operation uses its own SQL session."""

    def __init__(self, database: str | Path = "data/memory.db"):
        self.engine = open_database(database)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self) -> None:
        self.engine.dispose()

    def learn(self, feedback: str, previous_answer: str = "") -> str:
        mode = get_settings().procedural_memory_mode
        if mode == "off":
            return "disabled"
        with Session(self.engine) as db:
            message_id = record_feedback(db, feedback, previous_answer)
            status = procedural_writes.process_event(db, message_id)
            if mode in {"flat", "rsi"}:
                procedural_writes.process_sync(db)
            return status

    def ask(self, question: str) -> Answer:
        with Session(self.engine) as db:
            return ask(db, question, get_settings().procedural_memory_mode)

    def skills(self) -> list[dict]:
        with Session(self.engine) as db:
            return [
                {
                    "id": row.id,
                    "version": row.version,
                    "active": row.active,
                    "condition": row.condition,
                    "steps": row.steps,
                    "constraints": row.constraints,
                }
                for row in db.scalars(
                    select(ProceduralSkill).order_by(ProceduralSkill.id)
                )
            ]

    def retry(self) -> None:
        with Session(self.engine) as db:
            procedural_writes.recover(db, get_settings().procedural_memory_mode)

    def rebuild(self) -> None:
        with Session(self.engine) as db:
            procedural_retrieval.rebuild_index(db)
            db.commit()
