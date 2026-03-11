import pytest

from src.services.graph_store.adapters import (
    LlamaIndexNeo4jAdapter,
    LightRAGNeo4jAdapter,
    RAGAnythingNeo4jAdapter,
)


class FakeEmbedClient:
    async def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeClient:
    def connect(self):
        return None

    def close(self):
        return None


@pytest.mark.asyncio
async def test_llamaindex_adapter_vector_only(monkeypatch: pytest.MonkeyPatch):
    calls = {"vector": 0, "entity_rel": 0, "mentions": 0}

    class FakeVectorRepo:
        def __init__(self, _client):
            pass

        def upsert_chunks(self, chunks):
            calls["vector"] += len(chunks)

    class FakeGraphRepo:
        def __init__(self, _client):
            pass

        def upsert_docs(self, docs):
            return None

        def link_doc_chunks(self, links):
            return None

        def upsert_entities_relations(self, rows):
            calls["entity_rel"] += len(rows)

        def upsert_chunk_mentions(self, rows):
            calls["mentions"] += len(rows)

    monkeypatch.setattr("src.services.graph_store.adapters.llamaindex_adapter.load_graph_store_settings", lambda project_root=None: object())
    monkeypatch.setattr("src.services.graph_store.adapters.llamaindex_adapter.Neo4jClient", lambda settings: FakeClient())
    monkeypatch.setattr("src.services.graph_store.adapters.llamaindex_adapter.Neo4jVectorRepository", FakeVectorRepo)
    monkeypatch.setattr("src.services.graph_store.adapters.llamaindex_adapter.Neo4jGraphRepository", FakeGraphRepo)
    monkeypatch.setattr("src.services.graph_store.adapters.llamaindex_adapter.get_embedding_client", lambda: FakeEmbedClient())

    await LlamaIndexNeo4jAdapter.ingest_documents(
        kb_name="kb",
        documents=[{"doc_id": "d1", "path": "/a", "title": "t", "text": "Alpha beta gamma delta epsilon"}],
    )
    assert calls["vector"] > 0
    assert calls["entity_rel"] == 0
    assert calls["mentions"] == 0


@pytest.mark.asyncio
async def test_lightrag_adapter_writes_graph(monkeypatch: pytest.MonkeyPatch):
    calls = {"vector": 0, "entity_rel": 0, "mentions": 0}

    class FakeVectorRepo:
        def __init__(self, _client):
            pass

        def upsert_chunks(self, chunks):
            calls["vector"] += len(chunks)

    class FakeGraphRepo:
        def __init__(self, _client):
            pass

        def upsert_docs(self, docs):
            return None

        def link_doc_chunks(self, links):
            return None

        def upsert_entities_relations(self, rows):
            calls["entity_rel"] += len(rows)

        def upsert_chunk_mentions(self, rows):
            calls["mentions"] += len(rows)

    monkeypatch.setattr("src.services.graph_store.adapters.lightrag_adapter.load_graph_store_settings", lambda project_root=None: object())
    monkeypatch.setattr("src.services.graph_store.adapters.lightrag_adapter.Neo4jClient", lambda settings: FakeClient())
    monkeypatch.setattr("src.services.graph_store.adapters.lightrag_adapter.Neo4jVectorRepository", FakeVectorRepo)
    monkeypatch.setattr("src.services.graph_store.adapters.lightrag_adapter.Neo4jGraphRepository", FakeGraphRepo)
    monkeypatch.setattr("src.services.graph_store.adapters.lightrag_adapter.get_embedding_client", lambda: FakeEmbedClient())

    await LightRAGNeo4jAdapter.ingest_documents(
        kb_name="kb",
        documents=[{"doc_id": "d1", "path": "/a", "title": "t", "text": "Knowledge graph extraction test content"}],
    )
    assert calls["vector"] > 0
    assert calls["entity_rel"] > 0
    assert calls["mentions"] > 0


@pytest.mark.asyncio
async def test_raganything_adapter_inherits_graph(monkeypatch: pytest.MonkeyPatch):
    calls = {"entity_rel": 0}

    class FakeVectorRepo:
        def __init__(self, _client):
            pass

        def upsert_chunks(self, chunks):
            return None

    class FakeGraphRepo:
        def __init__(self, _client):
            pass

        def upsert_docs(self, docs):
            return None

        def link_doc_chunks(self, links):
            return None

        def upsert_entities_relations(self, rows):
            calls["entity_rel"] += len(rows)

        def upsert_chunk_mentions(self, rows):
            return None

    monkeypatch.setattr("src.services.graph_store.adapters.lightrag_adapter.load_graph_store_settings", lambda project_root=None: object())
    monkeypatch.setattr("src.services.graph_store.adapters.lightrag_adapter.Neo4jClient", lambda settings: FakeClient())
    monkeypatch.setattr("src.services.graph_store.adapters.lightrag_adapter.Neo4jVectorRepository", FakeVectorRepo)
    monkeypatch.setattr("src.services.graph_store.adapters.lightrag_adapter.Neo4jGraphRepository", FakeGraphRepo)
    monkeypatch.setattr("src.services.graph_store.adapters.lightrag_adapter.get_embedding_client", lambda: FakeEmbedClient())

    await RAGAnythingNeo4jAdapter.ingest_documents(
        kb_name="kb",
        documents=[{"doc_id": "d1", "path": "/a", "title": "t", "text": "Vision table equation extraction"}],
    )
    assert calls["entity_rel"] > 0

