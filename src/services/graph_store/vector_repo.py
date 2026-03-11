# -*- coding: utf-8 -*-
from typing import Any

from src.services.graph_store.neo4j_client import Neo4jClient
from src.services.graph_store.types import RetrievedChunk


class Neo4jVectorRepository:
    def __init__(self, client: Neo4jClient):
        self.client = client

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        cypher = """
        UNWIND $rows AS row
        MERGE (c:Chunk {chunk_id: row.chunk_id})
        SET c.kb_name = row.kb_name,
            c.content = row.content,
            c.embedding = row.embedding,
            c.source_doc = row.source_doc,
            c.provider = row.provider,
            c.hash = row.hash,
            c.created_at = row.created_at
        """
        self.client.execute_write(cypher, {"rows": chunks})

    def query_similar_chunks(
        self,
        *,
        kb_name: str,
        query_embedding: list[float],
        top_k: int = 5,
        provider: str = "",
    ) -> list[RetrievedChunk]:
        cypher = """
        CALL db.index.vector.queryNodes('chunk_embedding_idx', $top_k, $query_embedding)
        YIELD node, score
        WHERE node.kb_name = $kb_name
          AND ($provider = '' OR node.provider = $provider)
        RETURN node.chunk_id AS chunk_id,
               node.content AS text,
               score AS score,
               coalesce(node.source_doc, '') AS source_doc,
               coalesce(node.provider, '') AS provider
        ORDER BY score DESC
        """
        rows = self.client.execute_read(
            cypher,
            {
                "kb_name": kb_name,
                "query_embedding": query_embedding,
                "top_k": top_k,
                "provider": provider,
            },
        )
        return [
            RetrievedChunk(
                chunk_id=str(r.get("chunk_id") or ""),
                text=str(r.get("text") or ""),
                score=float(r.get("score") or 0.0),
                source_doc=str(r.get("source_doc") or ""),
                provider=str(r.get("provider") or ""),
            )
            for r in rows
        ]

