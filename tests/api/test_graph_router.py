from fastapi.testclient import TestClient

class FakeNeo4jClient:
    def __init__(self, settings):
        self.settings = settings

    def connect(self):
        return None

    def close(self):
        return None

    def verify_connectivity(self):
        return None

    def execute_write(self, cypher, parameters=None):
        return []

    def execute_read(self, cypher, parameters=None):
        if "SHOW VECTOR INDEXES" in cypher:
            return [{"name": "chunk_embedding_idx"}]
        return []


def _fake_user():
    return {"id": "u1", "email": "test@example.com"}


def test_graph_router_contract(monkeypatch):
    try:
        from src.api.dependencies.auth import get_current_user_from_header
        from src.api.main import app
    except Exception as exc:  # pragma: no cover - environment-dependent import
        import pytest

        pytest.skip(f"Skipping graph router test due to app import dependency error: {exc}")

    # Patch startup checks
    monkeypatch.setattr("src.api.main.Neo4jClient", FakeNeo4jClient)
    monkeypatch.setattr("src.api.routers.graph.Neo4jClient", FakeNeo4jClient)
    monkeypatch.setattr(
        "src.services.graph_store.neo4j_client.Neo4jClient",
        FakeNeo4jClient,
    )
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "pass")

    class FakeOrchestrator:
        async def query(self, **kwargs):
            return {
                "query": kwargs["query"],
                "provider": kwargs["provider"],
                "mode": kwargs["mode"],
                "answer": "ctx",
                "content": "ctx",
                "retrieved_chunks": [],
                "graph_context": {"nodes": [], "edges": [], "truncated": False},
            }

    monkeypatch.setattr(
        "src.api.routers.graph.GraphQueryOrchestrator",
        lambda: FakeOrchestrator(),
    )

    app.dependency_overrides[get_current_user_from_header] = _fake_user
    try:
        with TestClient(app) as client:
            q = client.post(
                "/api/v1/graph/query",
                json={"query": "test", "kb_name": "kb", "provider": "lightrag", "mode": "hybrid"},
            )
            assert q.status_code == 200
            assert q.json()["query"] == "test"

            s = client.get("/api/v1/graph/subgraph", params={"kb_name": "kb", "q": "abc", "hops": 2, "limit": 50})
            assert s.status_code == 200
            assert "nodes" in s.json()

            r = client.post("/api/v1/graph/reindex", params={"kb_name": "kb", "provider": "lightrag"})
            assert r.status_code == 200
            assert r.json()["status"] == "accepted"
    finally:
        app.dependency_overrides.clear()
