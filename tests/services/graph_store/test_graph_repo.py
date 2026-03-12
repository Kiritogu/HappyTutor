from src.services.graph_store.graph_repo import Neo4jGraphRepository


class StubRelationship:
    def __init__(self):
        self.start_node = {"entity_id": "e1"}
        self.end_node = {"entity_id": "e2"}
        self._data = {"type": "RELATED_TO", "weight": 0.7}

    def get(self, key, default=None):
        return self._data.get(key, default)


class StubClient:
    def __init__(self):
        self.last_write = None

    def execute_write(self, cypher, parameters=None):
        self.last_write = (cypher, parameters)
        return []

    def execute_read(self, cypher, parameters=None):
        return [
            {
                "nodes": [
                    {"entity_id": "e1", "name": "NodeA", "type": "concept", "kb_name": "kb"},
                    {"entity_id": "e2", "name": "NodeB", "type": "concept", "kb_name": "kb"},
                ],
                "rels": [StubRelationship()],
                "truncated": False,
            }
        ]


def test_upsert_entities_relations_writes():
    repo = Neo4jGraphRepository(StubClient())  # type: ignore[arg-type]
    repo.upsert_entities_relations(
        [
            {
                "kb_name": "kb",
                "source": {"entity_id": "e1", "name": "A", "type": "concept"},
                "target": {"entity_id": "e2", "name": "B", "type": "concept"},
                "relation_type": "RELATED_TO",
                "weight": 0.7,
                "evidence": "line 1",
                "provider": "lightrag",
            }
        ]
    )
    assert repo.client.last_write is not None  # type: ignore[attr-defined]


def test_fetch_subgraph_returns_nodes_edges():
    repo = Neo4jGraphRepository(StubClient())  # type: ignore[arg-type]
    nodes, edges, truncated = repo.fetch_subgraph(kb_name="kb", query="Node", hops=2, limit=100)
    assert len(nodes) == 2
    assert len(edges) == 1
    assert truncated is False


def test_fetch_overview_subgraph_returns_nodes_edges():
    repo = Neo4jGraphRepository(StubClient())  # type: ignore[arg-type]
    nodes, edges, truncated = repo.fetch_overview_subgraph(kb_name="kb", limit=120)
    assert len(nodes) == 2
    assert len(edges) == 1
    assert truncated is False


class TupleRelClient(StubClient):
    def execute_read(self, cypher, parameters=None):
        return [
            {
                "nodes": [
                    {"entity_id": "e1", "name": "NodeA", "type": "concept", "kb_name": "kb"},
                    {"entity_id": "e2", "name": "NodeB", "type": "concept", "kb_name": "kb"},
                ],
                "rels": [("e1", "e2", "RELATED_TO", 0.9)],
                "truncated": False,
            }
        ]


def test_fetch_overview_subgraph_supports_tuple_relationships():
    repo = Neo4jGraphRepository(TupleRelClient())  # type: ignore[arg-type]
    nodes, edges, truncated = repo.fetch_overview_subgraph(kb_name="kb", limit=120)
    assert len(nodes) == 2
    assert len(edges) == 1
    assert edges[0].source == "e1"
    assert edges[0].target == "e2"
    assert edges[0].relation == "RELATED_TO"
    assert truncated is False
