from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Generator, Optional

from neo4j import GraphDatabase
from neo4j import Driver, Session

@contextmanager
def get_session(driver: Driver) -> Generator[Session, None, None]:
    """Yield a Neo4j Session (neo4j v5+: no default_access kwarg)."""
    session = driver.session()
    try:
        yield session
    finally:
        session.close()


def get_driver(uri: str, user: str, password: str) -> Driver:
    # Create neo4j driver
    return GraphDatabase.driver(uri, auth=(user, password))

def ensure_constraints(driver: Driver) -> None:
    """Create idempotent constraints required by the pipeline.

    - Uniqueness on CTIDocument.id ensures we never duplicate the same document node.
    - Uniqueness on CTIEntity (name, type, doc_id) to ensure per-document identity.
    """
    with get_session(driver) as session:
        session.run(
            "CREATE CONSTRAINT cti_document_id IF NOT EXISTS FOR (d:CTIDocument) REQUIRE d.id IS UNIQUE"
        )
        # Enforce per-document uniqueness for entities by (name, type, doc_id)
        session.run(
            "CREATE CONSTRAINT cti_entity_name_type_doc IF NOT EXISTS FOR (e:CTIEntity) REQUIRE (e.name, e.type, e.doc_id) IS UNIQUE"
        )

def clear_cti_entities(driver: Driver) -> None:
    """Delete all CTIEntity nodes and their relationships."""
    with get_session(driver) as session:
        session.run("MATCH (n:CTIEntity) DETACH DELETE n")


def clear_cti_graph(driver: Driver) -> None:
    """Delete all CTIEntity and CTIDocument nodes (and their relationships)."""
    with get_session(driver) as session:
        session.run("MATCH (n:CTIEntity) DETACH DELETE n")
        session.run("MATCH (d:CTIDocument) DETACH DELETE d")

def document_exists(driver: Driver, document_id: str) -> bool:
    """Return True if a CTIDocument node with the given id exists."""
    with get_session(driver) as session:
        rec = session.run(
            "MATCH (d:CTIDocument {id: $id}) RETURN d LIMIT 1", {"id": document_id}
        ).single()
        return rec is not None

def upsert_document(driver: Driver, document_id: str, props: Optional[Dict] = None) -> None:
    """Ensure a CTIDocument node exists with the given id and properties.

    Safe to call repeatedly; uses MERGE and SET += for idempotence.
    """
    with get_session(driver) as session:
        session.run(
            "MERGE (d:CTIDocument {id: $id}) SET d += $props",
            {"id": document_id, "props": props or {"id": document_id}},
        )

__all__ = [
    "get_driver",
    "get_session",
    "ensure_constraints",
    "clear_cti_entities",
    "clear_cti_graph",
    "document_exists",
    "upsert_document",
]