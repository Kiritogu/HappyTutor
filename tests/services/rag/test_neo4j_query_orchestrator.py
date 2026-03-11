import pytest

from src.services.rag.service import RAGService


@pytest.mark.asyncio
async def test_rag_service_search_uses_graph_orchestrator(monkeypatch: pytest.MonkeyPatch):
    class FakeOrchestrator:
        async def query(self, **kwargs):
            return {
                "query": kwargs["query"],
                "provider": kwargs["provider"],
                "mode": kwargs["mode"],
                "answer": "ctx",
                "content": "ctx",
                "retrieved_chunks": [
                    {
                        "chunk_id": "c1",
                        "text": "hello",
                        "score": 0.99,
                        "source_doc": "d1",
                        "provider": kwargs["provider"],
                    }
                ],
            }

    monkeypatch.setattr("src.services.rag.service.GraphQueryOrchestrator", lambda: FakeOrchestrator())

    service = RAGService(provider="llamaindex")
    result = await service.search("What is AI?", "kb_test", mode="hybrid", top_k=3)

    assert result["query"] == "What is AI?"
    assert result["provider"] == "llamaindex"
    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieved_chunks"][0]["chunk_id"] == "c1"

