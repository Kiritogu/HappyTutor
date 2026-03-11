# -*- coding: utf-8 -*-
from src.services.graph_store.neo4j_client import Neo4jClient


SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Doc) REQUIRE d.doc_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE INDEX doc_kb_idx IF NOT EXISTS FOR (d:Doc) ON (d.kb_name)",
    "CREATE INDEX chunk_kb_idx IF NOT EXISTS FOR (c:Chunk) ON (c.kb_name)",
    "CREATE INDEX entity_kb_idx IF NOT EXISTS FOR (e:Entity) ON (e.kb_name)",
    (
        "CREATE VECTOR INDEX chunk_embedding_idx IF NOT EXISTS "
        "FOR (c:Chunk) ON (c.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}"
    ),
]


def ensure_schema(client: Neo4jClient) -> None:
    for stmt in SCHEMA_STATEMENTS:
        client.execute_write(stmt)


def verify_vector_index(client: Neo4jClient) -> None:
    rows = client.execute_read("SHOW VECTOR INDEXES YIELD name RETURN name")
    names = {r.get("name") for r in rows}
    if "chunk_embedding_idx" not in names:
        raise RuntimeError("Neo4j vector index chunk_embedding_idx is missing")

