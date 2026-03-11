from src.services.graph_store.schema import ensure_schema, verify_vector_index


class StubClient:
    def __init__(self):
        self.writes = []
        self.read_rows = [{"name": "chunk_embedding_idx"}]

    def execute_write(self, cypher, params=None):
        self.writes.append((cypher, params))
        return []

    def execute_read(self, cypher, params=None):
        return self.read_rows


def test_ensure_schema_runs_statements():
    client = StubClient()
    ensure_schema(client)  # type: ignore[arg-type]
    assert len(client.writes) >= 6


def test_verify_vector_index_present():
    client = StubClient()
    verify_vector_index(client)  # type: ignore[arg-type]

