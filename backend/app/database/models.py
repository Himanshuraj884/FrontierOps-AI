"""
SQLAlchemy models. Works against either SQLite (local dev default, see
database/db.py) or PostgreSQL (production, see docker-compose.yml) — same
models, different connection string.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    role = Column(String, default="member")
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)  # document_id
    title = Column(String, nullable=False)
    source_path = Column(String)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    chunks = relationship("DocumentChunk", back_populates="document")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(String, primary_key=True)  # chunk_id
    document_id = Column(String, ForeignKey("documents.id"))
    section = Column(String, nullable=True)
    chunk_index = Column(Integer)
    text = Column(Text, nullable=False)
    document = relationship("Document", back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)  # user | assistant
    content = Column(Text)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    conversation = relationship("Conversation", back_populates="messages")


class AgentRun(Base):
    __tablename__ = "agent_runs"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    agent_name = Column(String)
    status = Column(String)
    latency_ms = Column(Float)
    token_usage = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class ValidationResult(Base):
    __tablename__ = "validation_results"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    passed = Column(Boolean)
    details = Column(JSON)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class Evaluation(Base):
    __tablename__ = "evaluations"
    id = Column(Integer, primary_key=True)
    run_label = Column(String)
    metrics = Column(JSON)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class HumanReview(Base):
    __tablename__ = "human_reviews"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    decision = Column(String, nullable=True)  # approve | reject | regenerate
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    request_id = Column(String)
    agent = Column(String)
    action = Column(String)
    result = Column(JSON)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
