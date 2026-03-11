# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any

from src.services.embedding import get_embedding_client
from src.services.graph_store.graph_repo import Neo4jGraphRepository
from src.services.graph_store.neo4j_client import Neo4jClient, load_graph_store_settings
from src.services.graph_store.vector_repo import Neo4jVectorRepository


class GraphQueryOrchestrator:
    def __init__(self, project_root: Path | None = None):
        settings = load_graph_store_settings(project_root=project_root)
        self.client = Neo4jClient(settings)
        self.vector_repo = Neo4jVectorRepository(self.client)
        self.graph_repo = Neo4jGraphRepository(self.client)
        self.embedding_client = get_embedding_client()

    async def query(
        self,
        *,
        query: str,
        kb_name: str,
        provider: str,
        mode: str,
        top_k: int = 5,
        with_graph: bool = True,
    ) -> dict[str, Any]:
        query_embedding = (await self.embedding_client.embed([query]))[0]
        chunks = self.vector_repo.query_similar_chunks(
            kb_name=kb_name,
            query_embedding=query_embedding,
            top_k=top_k,
            provider=provider,
        )

        retrieved_chunks = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "score": c.score,
                "source_doc": c.source_doc,
                "provider": c.provider,
            }
            for c in chunks
        ]

        context = "\n\n".join(c["text"] for c in retrieved_chunks if c.get("text"))
        response: dict[str, Any] = {
            "query": query,
            "provider": provider,
            "mode": mode,
            "answer": context,
            "content": context,
            "retrieved_chunks": retrieved_chunks,
        }

        if with_graph and provider in {"lightrag", "raganything"}:
            nodes, edges, truncated = self.graph_repo.fetch_subgraph(
                kb_name=kb_name,
                query=query,
                hops=2,
                limit=120,
            )
            response["graph_context"] = {
                "nodes": [n.__dict__ for n in nodes],
                "edges": [e.__dict__ for e in edges],
                "truncated": truncated,
            }

        return response
