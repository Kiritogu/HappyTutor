from src.services.graph_store.vector_repo import Neo4jVectorRepository


class StubClient:
    def __init__(self):
        self.last_write = None
        self.last_read = None

    def execute_write(self, cypher, parameters=None):
        self.last_write = (cypher, parameters)
        return []

    def execute_read(self, cypher, parameters=None):
        self.last_read = (cypher, parameters)
        return [
            {
                "chunk_id": "c1",
                "text": "chunk text",
                "score": 0.9,
                "source_doc": "doc1",
                "provider": "llamaindex",
            }
        ]


def test_upsert_chunks_writes_rows():
    repo = Neo4jVectorRepository(StubClient())  # type: ignore[arg-type]
    repo.upsert_chunks(
        [
            {
                "chunk_id": "c1",
                "kb_name": "kb",
                "content": "hello",
                "embedding": [0.1, 0.2],
                "source_doc": "doc1",
                "provider": "llamaindex",
                "hash": "h1",
                "created_at": "2026-03-11T00:00:00Z",
            }
        ]
    )
    assert repo.client.last_write is not None  # type: ignore[attr-defined]


def test_query_similar_chunks_returns_contract():
    repo = Neo4jVectorRepository(StubClient())  # type: ignore[arg-type]
    chunks = repo.query_similar_chunks(kb_name="kb", query_embedding=[0.3, 0.4], top_k=3)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c1"
    assert chunks[0].provider == "llamaindex"

