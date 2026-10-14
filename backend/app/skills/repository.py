"""Standalone SQLite state and publication journal for the skill pool."""

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Column, Integer, Table, Text, create_engine, delete, insert, select, update
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Independent metadata; never participates in application migrations."""


BUCKETS = ("skills", "clusters", "references", "members", "candidates", "tasks", "meta")
TABLES = {
    name: Table(
        f"pool_{name}",
        Base.metadata,
        Column("id", Text, primary_key=True),
        Column("payload", Text, nullable=False),
    )
    for name in BUCKETS
}
REVISION = Table(
    "pool_revision",
    Base.metadata,
    Column("id", Integer, primary_key=True),
    Column("value", Integer, nullable=False),
)
OPERATIONS = Table(
    "pool_operations",
    Base.metadata,
    Column("id", Text, primary_key=True),
    Column("status", Text, nullable=False),
    Column("payload", Text, nullable=False),
)


def empty_state() -> dict:
    state = {name: {} for name in BUCKETS}
    state["meta"] = {"pool": {"updates": 0, "dirty_clusters": []}}
    return state


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class Repository:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            "sqlite:///" + (self.root / "pool.db").as_posix(),
            connect_args={"timeout": 30},
        )
        Base.metadata.create_all(self.engine)
        with self.engine.connect() as conn:
            conn.exec_driver_sql("BEGIN IMMEDIATE")
            if conn.execute(select(REVISION.c.value)).first() is None:
                conn.execute(insert(REVISION).values(id=1, value=0))
                for key, value in empty_state()["meta"].items():
                    conn.execute(insert(TABLES["meta"]).values(id=key, payload=_dump(value)))
            conn.commit()

    def _load(self, conn) -> tuple[int, dict]:
        revision = conn.execute(select(REVISION.c.value).where(REVISION.c.id == 1)).scalar_one()
        state = {
            name: {row.id: json.loads(row.payload) for row in conn.execute(select(table))}
            for name, table in TABLES.items()
        }
        return revision, state

    def load(self) -> tuple[int, dict]:
        with self.engine.connect() as conn:
            conn.exec_driver_sql("BEGIN")
            return self._load(conn)

    def stage(
        self, operation_id: str, kind: str, expected_revision: int, after: dict, reason: str = ""
    ) -> dict:
        if set(after) != set(BUCKETS) or any(not isinstance(after[k], dict) for k in BUCKETS):
            raise ValueError("state must contain all skill-pool buckets")
        # Round trip prevents callers mutating staged data and rejects non-JSON values.
        after = json.loads(_dump(after))
        with self.engine.connect() as conn:
            conn.exec_driver_sql("BEGIN IMMEDIATE")
            if conn.execute(
                select(OPERATIONS.c.id).where(OPERATIONS.c.status == "pending")
            ).first():
                raise RuntimeError("recover pending operation before modifying the pool")
            revision, before = self._load(conn)
            if revision != expected_revision:
                raise RuntimeError("pool revision changed")
            if conn.execute(select(OPERATIONS.c.id).where(OPERATIONS.c.id == operation_id)).first():
                raise ValueError("operation id already exists")
            operation = {
                "id": operation_id,
                "kind": kind,
                "before_revision": revision,
                "before": before,
                "after": after,
                "reason": reason,
                "status": "pending",
                "attempts": 0,
                "last_error": None,
                "created_at": datetime.now(UTC).isoformat(),
            }
            conn.execute(
                insert(OPERATIONS).values(
                    id=operation_id, status="pending", payload=_dump(operation)
                )
            )
            conn.commit()
            return copy.deepcopy(operation)

    def operation(self, operation_id: str) -> dict:
        with self.engine.connect() as conn:
            value = conn.execute(
                select(OPERATIONS.c.payload).where(OPERATIONS.c.id == operation_id)
            ).scalar_one_or_none()
            if value is None:
                raise KeyError(operation_id)
            return json.loads(value)

    def history(self) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(OPERATIONS.c.payload))
            return sorted((json.loads(row[0]) for row in rows), key=lambda op: op["created_at"])

    def pending(self) -> list[dict]:
        return [op for op in self.history() if op["status"] == "pending"]

    def publish(self, operation_id: str) -> dict:
        with self.engine.connect() as conn:
            conn.exec_driver_sql("BEGIN IMMEDIATE")
            payload = conn.execute(
                select(OPERATIONS.c.payload).where(OPERATIONS.c.id == operation_id)
            ).scalar_one_or_none()
            if payload is None:
                raise KeyError(operation_id)
            operation = json.loads(payload)
            if operation["status"] == "committed":
                return operation
            revision, before = self._load(conn)
            if revision != operation["before_revision"] or before != operation["before"]:
                raise RuntimeError("pool revision changed before publication")
            after = operation["after"]
            for name, table in TABLES.items():
                for entity_id in before[name].keys() - after[name].keys():
                    conn.execute(delete(table).where(table.c.id == entity_id))
                for entity_id, entity in after[name].items():
                    if entity_id not in before[name]:
                        conn.execute(insert(table).values(id=entity_id, payload=_dump(entity)))
                    elif entity != before[name][entity_id]:
                        conn.execute(
                            update(table)
                            .where(table.c.id == entity_id)
                            .values(payload=_dump(entity))
                        )
            operation.update(
                status="committed",
                after_revision=revision + 1,
                committed_at=datetime.now(UTC).isoformat(),
                last_error=None,
            )
            conn.execute(update(REVISION).where(REVISION.c.id == 1).values(value=revision + 1))
            conn.execute(
                update(OPERATIONS)
                .where(OPERATIONS.c.id == operation_id)
                .values(status="committed", payload=_dump(operation))
            )
            conn.commit()
            return operation

    def record_failure(self, operation_id: str, error) -> None:
        with self.engine.connect() as conn:
            conn.exec_driver_sql("BEGIN IMMEDIATE")
            payload = conn.execute(
                select(OPERATIONS.c.payload).where(OPERATIONS.c.id == operation_id)
            ).scalar_one_or_none()
            if payload is None:
                raise KeyError(operation_id)
            operation = json.loads(payload)
            if operation["status"] == "pending":
                operation["attempts"] += 1
                operation["last_error"] = type(error).__name__
                conn.execute(
                    update(OPERATIONS)
                    .where(OPERATIONS.c.id == operation_id)
                    .values(payload=_dump(operation))
                )
            conn.commit()

    def dispose(self) -> None:
        self.engine.dispose()
