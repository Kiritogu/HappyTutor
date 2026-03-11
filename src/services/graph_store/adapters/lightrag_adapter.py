# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any

from src.services.embedding import get_embedding_client
from src.services.graph_store.graph_repo import Neo4jGraphRepository
from src.services.graph_store.neo4j_client import Neo4jClient, load_graph_store_settings
from src.services.graph_store.vector_repo import Neo4jVectorRepository

from ._utils import build_chunk_id, chunk_text, extract_entities, pairwise_relations, utc_now_iso


class LightRAGNeo4jAdapter:
    provider = "lightrag"

    @classmethod
    async def ingest_documents(cls, *, kb_name: str, documents: list[dict[str, Any]]) -> None:
        settings = load_graph_store_settings(project_root=Path(__file__).resolve().parent.parent.parent.parent.parent)
        client = Neo4jClient(settings)
        client.connect()
        try:
            vector_repo = Neo4jVectorRepository(client)
            graph_repo = Neo4jGraphRepository(client)
            embed_client = get_embedding_client()

            doc_rows: list[dict[str, Any]] = []
            link_rows: list[dict[str, Any]] = []
            chunk_rows: list[dict[str, Any]] = []
            mention_rows: list[dict[str, Any]] = []
            relation_rows: list[dict[str, Any]] = []

            for doc in documents:
                doc_id = str(doc.get("doc_id") or doc.get("path") or doc.get("title") or "doc")
                doc_rows.append(
                    {
                        "doc_id": doc_id,
                        "kb_name": kb_name,
                        "path": str(doc.get("path") or ""),
                        "title": str(doc.get("title") or ""),
                        "provider": cls.provider,
                    }
                )
                chunks = chunk_text(str(doc.get("text") or ""))
                if not chunks:
                    continue

                vectors = await embed_client.embed(chunks)
                for idx, text in enumerate(chunks):
                    chunk_id = build_chunk_id(kb_name, cls.provider, doc_id, text, idx)
                    chunk_rows.append(
                        {
                            "chunk_id": chunk_id,
                            "kb_name": kb_name,
                            "content": text,
                            "embedding": vectors[idx],
                            "source_doc": doc_id,
                            "provider": cls.provider,
                            "hash": chunk_id,
                            "created_at": utc_now_iso(),
                        }
                    )
                    link_rows.append({"doc_id": doc_id, "chunk_id": chunk_id})

                    entities = extract_entities(text)
                    for ent in entities:
                        mention_rows.append(
                            {
                                "kb_name": kb_name,
                                "chunk_id": chunk_id,
                                "entity": {
                                    "entity_id": f"{kb_name}:{ent}",
                                    "name": ent,
                                    "type": "keyword",
                                    "aliases": [],
                                },
                                "weight": 1.0,
                                "evidence": text[:200],
                            }
                        )
                    for src, dst in pairwise_relations(entities):
                        relation_rows.append(
                            {
                                "kb_name": kb_name,
                                "source": {
                                    "entity_id": f"{kb_name}:{src}",
                                    "name": src,
                                    "type": "keyword",
                                    "aliases": [],
                                },
                                "target": {
                                    "entity_id": f"{kb_name}:{dst}",
                                    "name": dst,
                                    "type": "keyword",
                                    "aliases": [],
                                },
                                "relation_type": "CO_OCCURS_WITH",
                                "weight": 1.0,
                                "evidence": text[:200],
                                "provider": cls.provider,
                            }
                        )

            if doc_rows:
                graph_repo.upsert_docs(doc_rows)
            if chunk_rows:
                vector_repo.upsert_chunks(chunk_rows)
            if link_rows:
                graph_repo.link_doc_chunks(link_rows)
            if mention_rows:
                graph_repo.upsert_chunk_mentions(mention_rows)
            if relation_rows:
                graph_repo.upsert_entities_relations(relation_rows)
        finally:
            client.close()

