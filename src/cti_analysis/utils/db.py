from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Generator, Optional

try:
    from neo4j import GraphDatabase
    from neo4j import Driver, Session
except ImportError:
    GraphDatabase = None
    Driver = None
    Session = None


@contextmanager
def get_session(driver: Driver) -> Generator[Session, None, None]:
    """Yield a Neo4j Session."""
    session = driver.session()
    try:
        yield session
    finally:
        session.close()


def get_driver(uri: str, user: str, password: str) -> Driver:
    return GraphDatabase.driver(uri, auth=(user, password))


def ensure_constraints(driver: Driver) -> None:
    """Create idempotent constraints required by the pipeline."""
    with get_session(driver) as session:
        session.run(
            "CREATE CONSTRAINT cti_document_id IF NOT EXISTS FOR (d:CTIDocument) REQUIRE d.id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT cti_entity_name_type_doc IF NOT EXISTS FOR (e:CTIEntity) REQUIRE (e.name, e.type, e.doc_id) IS UNIQUE"
        )


def clear_cti_entities(driver: Driver) -> None:
    with get_session(driver) as session:
        session.run("MATCH (n:CTIEntity) DETACH DELETE n")


def clear_cti_graph(driver: Driver) -> None:
    with get_session(driver) as session:
        session.run("MATCH (n:CTIEntity) DETACH DELETE n")
        session.run("MATCH (d:CTIDocument) DETACH DELETE d")


def document_exists(driver: Driver, document_id: str) -> bool:
    with get_session(driver) as session:
        rec = session.run(
            "MATCH (d:CTIDocument {id: $id}) RETURN d LIMIT 1", {"id": document_id}
        ).single()
        return rec is not None


def upsert_document(driver: Driver, document_id: str, props: Optional[Dict] = None) -> None:
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

