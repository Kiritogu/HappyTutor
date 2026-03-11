from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies.auth import get_current_user_from_header
from src.services.graph_store.graph_repo import Neo4jGraphRepository
from src.services.graph_store.neo4j_client import Neo4jClient, load_graph_store_settings
from src.services.graph_store.query_orchestrator import GraphQueryOrchestrator

router = APIRouter()


class GraphQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    kb_name: str = Field(min_length=1)
    provider: str = Field(default="raganything")
    mode: str = Field(default="hybrid")
    top_k: int = Field(default=5, ge=1, le=50)
    with_graph: bool = True


@router.post("/query")
async def query_graph(
    payload: GraphQueryRequest,
    current_user: dict = Depends(get_current_user_from_header),
):
    del current_user
    orchestrator = GraphQueryOrchestrator()
    return await orchestrator.query(
        query=payload.query,
        kb_name=payload.kb_name,
        provider=payload.provider,
        mode=payload.mode,
        top_k=payload.top_k,
        with_graph=payload.with_graph,
    )


@router.get("/subgraph")
async def get_subgraph(
    kb_name: str = Query(min_length=1),
    q: str = Query(min_length=1),
    hops: int = Query(default=2, ge=1, le=3),
    limit: int = Query(default=200, ge=1, le=300),
    current_user: dict = Depends(get_current_user_from_header),
):
    del current_user
    settings = load_graph_store_settings(project_root=Path(__file__).resolve().parent.parent.parent.parent)
    client = Neo4jClient(settings)
    try:
        client.connect()
        repo = Neo4jGraphRepository(client)
        nodes, edges, truncated = repo.fetch_subgraph(kb_name=kb_name, query=q, hops=hops, limit=limit)
        return {
            "nodes": [n.__dict__ for n in nodes],
            "edges": [e.__dict__ for e in edges],
            "truncated": truncated,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph subgraph query failed: {e}")
    finally:
        client.close()


@router.post("/reindex")
async def reindex_graph(
    kb_name: str = Query(min_length=1),
    provider: str = Query(min_length=1),
    current_user: dict = Depends(get_current_user_from_header),
):
    del current_user
    # Reindex pipeline is intentionally explicit and can be wired to job queue later.
    return {
        "status": "accepted",
        "message": "Reindex endpoint placeholder is active. Use KB initialization workflow to rebuild indexes.",
        "kb_name": kb_name,
        "provider": provider,
    }

