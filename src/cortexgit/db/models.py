import uuid
import os
import json
from enum import Enum as PyEnum
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    Enum,
    Float,
    String,
    UUID,
    Integer,
    UniqueConstraint,
    Index,
)
from sqlalchemy.types import TypeDecorator
from sqlalchemy.orm import declarative_base, relationship

def check_vector_support():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return False
    if "sqlite" in db_url:
        return False
    # Parse asyncpg URL to standard psycopg2 URL if needed
    if "postgresql+asyncpg" in db_url:
        db_url = db_url.replace("postgresql+asyncpg", "postgresql")
    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector';")
        has_vector = cursor.fetchone() is not None
        cursor.close()
        conn.close()
        return has_vector
    except Exception:
        return False

HAS_PGVECTOR = check_vector_support()

# Define Portable Custom Column Types using SQLAlchemy TypeDecorators

class PortableJSON(TypeDecorator):
    """
    Portable JSON type. Uses PostgreSQL JSONB dialect type if available,
    otherwise falls back to standard Text type with JSON serialization.
    """
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import JSONB
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return value


class PortableInt4Range(TypeDecorator):
    """
    Portable Integer Range type. Uses PostgreSQL INT4RANGE if available,
    otherwise falls back to String, storing ranges as "start,end".
    """
    impl = String
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import INT4RANGE
            return dialect.type_descriptor(INT4RANGE())
        return dialect.type_descriptor(String(255))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        if isinstance(value, (tuple, list)):
            return f"{value[0]},{value[1]}"
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        if isinstance(value, (tuple, list)):
            return value
        try:
            parts = value.split(",")
            return (int(parts[0]), int(parts[1]))
        except Exception:
            return value


class PortableArray(TypeDecorator):
    """
    Portable Array type. Uses PostgreSQL ARRAY(Text) if available,
    otherwise falls back to Text storing a JSON serialized list.
    """
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import ARRAY
            return dialect.type_descriptor(ARRAY(Text()))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        if isinstance(value, list):
            return value
        try:
            return json.loads(value)
        except Exception:
            return value


class PortableEmbedding(TypeDecorator):
    """
    Portable Embedding type. Uses pgvector Vector(1536) if available,
    otherwise standard PostgreSQL ARRAY(Float), and Text (JSON list of floats) on SQLite.
    """
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql" and HAS_PGVECTOR:
            try:
                from pgvector.sqlalchemy import Vector
                return dialect.type_descriptor(Vector(1536))
            except ImportError:
                pass
        if dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import ARRAY
            return dialect.type_descriptor(ARRAY(Float()))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        if isinstance(value, str):
            return value
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        if isinstance(value, (list, tuple)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return value


Base = declarative_base()

class EventType(str, PyEnum):
    SYSTEM = "system"
    USER = "user"
    AGENT = "agent"
    ACTION = "action"
    OBSERVATION = "observation"
    THOUGHT = "thought"
    ERROR = "error"

class EventLog(Base):
    __tablename__ = "event_log"

    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(Text, nullable=False, index=True)
    agent_id = Column(Text, nullable=False, index=True)
    event_type = Column(Enum(EventType, native_enum=True), nullable=False)
    payload = Column(PortableJSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

class EntityRegistry(Base):
    __tablename__ = "entity_registry"

    key = Column(Text, primary_key=True)
    value = Column(PortableJSON, nullable=False)
    agent_id = Column(Text, nullable=False)
    event_id = Column(UUID(as_uuid=True), ForeignKey("event_log.event_id"), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), 
        nullable=False, 
        default=lambda: datetime.now(timezone.utc), 
        onupdate=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    event = relationship("EventLog", backref="entities")

class SnapshotStore(Base):
    __tablename__ = "snapshot_store"

    snapshot_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(Text, nullable=True, index=True)
    event_range = Column(PortableInt4Range, nullable=False)
    summary = Column(Text, nullable=False)
    entities_mentioned = Column(PortableArray, nullable=False)
    embedding = Column(PortableEmbedding, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

class ConflictLog(Base):
    __tablename__ = "conflict_log"

    conflict_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(Text, nullable=False, index=True)
    existing_value = Column(PortableJSON, nullable=False)
    proposed_value = Column(PortableJSON, nullable=False)
    existing_event_id = Column(UUID(as_uuid=True), ForeignKey("event_log.event_id"), nullable=False)
    proposed_event_id = Column(UUID(as_uuid=True), ForeignKey("event_log.event_id"), nullable=False)
    resolved = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    # Relationships
    existing_event = relationship("EventLog", foreign_keys=[existing_event_id])
    proposed_event = relationship("EventLog", foreign_keys=[proposed_event_id])


class EntityNodeType(str, PyEnum):
    PROJECT = "project"
    CONCEPT = "concept"
    PERSON = "person"


class HitType(str, PyEnum):
    QUERY = "query"
    UPDATE = "update"
    INFERENCE = "inference"


class EntityNode(Base):
    __tablename__ = "entity_nodes"

    node_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_name = Column(Text, index=True, nullable=False)   # uniqueness scoped to agent; see __table_args__
    entity_type = Column(Enum(EntityNodeType, native_enum=True), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(Text, nullable=True)
    degree_centrality = Column(Float, nullable=False, default=0.0)
    hit_frequency = Column(Integer, nullable=False, default=0)
    last_hit = Column(DateTime(timezone=True), nullable=True)
    ttl_expiry = Column(DateTime(timezone=True), index=True, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    agent_id = Column(Text, index=True, nullable=False)

    __table_args__ = (
        # Per-agent uniqueness: same entity_name is allowed for different agents
        UniqueConstraint("entity_name", "agent_id", name="uq_entity_nodes_name_agent"),
        Index("ix_entity_nodes_name_agent", "entity_name", "agent_id"),
    )


class EntityEdge(Base):
    __tablename__ = "entity_edges"

    edge_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_node_id = Column(UUID(as_uuid=True), ForeignKey("entity_nodes.node_id"), nullable=False)
    target_node_id = Column(UUID(as_uuid=True), ForeignKey("entity_nodes.node_id"), nullable=False)
    relation_type = Column(Text, nullable=False)
    weight = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("source_node_id", "target_node_id", "relation_type", name="uq_entity_edges_source_target_relation"),
        Index("ix_entity_edges_source_target", "source_node_id", "target_node_id"),
    )


class NodeHit(Base):
    __tablename__ = "node_hits"

    hit_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_id = Column(UUID(as_uuid=True), ForeignKey("entity_nodes.node_id"), nullable=False)
    hit_type = Column(Enum(HitType, native_enum=True), nullable=False)
    hit_timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    session_id = Column(Text, nullable=False)

    __table_args__ = (
        Index("ix_node_hits_node_id_timestamp", "node_id", "hit_timestamp"),
    )

