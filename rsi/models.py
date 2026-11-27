from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="Research Chat")
    session_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(Integer, index=True)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TranscriptEvent(Base):
    __tablename__ = "transcripts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ProceduralSkill(Base):
    __tablename__ = "procedural_skills"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[int] = mapped_column(default=1)
    active: Mapped[bool] = mapped_column(default=True)
    condition: Mapped[str] = mapped_column(Text)
    steps: Mapped[list[str]] = mapped_column(JSON)
    constraints: Mapped[list[str]] = mapped_column(JSON)
    source_event_id: Mapped[int]
    source_session_id: Mapped[int]


class ProceduralFeedback(Base):
    __tablename__ = "procedural_feedback"

    message_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    status: Mapped[str] = mapped_column(String, default="pending", index=True)
    trace_id: Mapped[int | None]
    strategy: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    attempts: Mapped[int] = mapped_column(default=0)
    reason: Mapped[str | None]
    skill_id: Mapped[str | None]
    version: Mapped[int | None]
    action: Mapped[str | None]


class ProceduralSync(Base):
    __tablename__ = "procedural_sync"

    id: Mapped[int] = mapped_column(primary_key=True)
    skill_id: Mapped[str]
    version: Mapped[int]
    attempts: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(default="pending", index=True)
    last_error: Mapped[str | None]


class SkillCluster(Base):
    __tablename__ = "procedural_skill_clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    # Sum of normalized member embeddings. Its direction is the cosine centroid.
    vector_sum: Mapped[list[float]] = mapped_column(JSON, nullable=False)


class SkillMember(Base):
    __tablename__ = "procedural_skill_members"

    skill_id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    cluster_id: Mapped[int] = mapped_column(
        ForeignKey("procedural_skill_clusters.id"),
        nullable=False,
        index=True,
    )
    vector: Mapped[list[float]] = mapped_column(JSON, nullable=False)
