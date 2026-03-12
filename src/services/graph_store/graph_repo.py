# -*- coding: utf-8 -*-
from typing import Any

from src.services.graph_store.neo4j_client import Neo4jClient
from src.services.graph_store.types import GraphEdge, GraphNode


class Neo4jGraphRepository:
    def __init__(self, client: Neo4jClient):
        self.client = client

    def upsert_docs(self, docs: list[dict[str, Any]]) -> None:
        cypher = """
        UNWIND $rows AS row
        MERGE (d:Doc {doc_id: row.doc_id})
        SET d.kb_name = row.kb_name,
            d.path = row.path,
            d.title = row.title,
            d.provider = row.provider
        """
        self.client.execute_write(cypher, {"rows": docs})

    def link_doc_chunks(self, links: list[dict[str, Any]]) -> None:
        cypher = """
        UNWIND $rows AS row
        MATCH (d:Doc {doc_id: row.doc_id})
        MATCH (c:Chunk {chunk_id: row.chunk_id})
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        self.client.execute_write(cypher, {"rows": links})

    def upsert_entities_relations(self, rows: list[dict[str, Any]]) -> None:
        cypher = """
        UNWIND $rows AS row
        MERGE (src:Entity {entity_id: row.source.entity_id})
        SET src.kb_name = row.kb_name,
            src.name = row.source.name,
            src.type = row.source.type,
            src.aliases = coalesce(row.source.aliases, [])
        MERGE (dst:Entity {entity_id: row.target.entity_id})
        SET dst.kb_name = row.kb_name,
            dst.name = row.target.name,
            dst.type = row.target.type,
            dst.aliases = coalesce(row.target.aliases, [])
        MERGE (src)-[r:REL {type: row.relation_type}]->(dst)
        SET r.weight = coalesce(row.weight, 1.0),
            r.evidence = coalesce(row.evidence, ''),
            r.provider = coalesce(row.provider, '')
        """
        self.client.execute_write(cypher, {"rows": rows})

    def upsert_chunk_mentions(self, rows: list[dict[str, Any]]) -> None:
        cypher = """
        UNWIND $rows AS row
        MATCH (c:Chunk {chunk_id: row.chunk_id})
        MERGE (e:Entity {entity_id: row.entity.entity_id})
        SET e.kb_name = row.kb_name,
            e.name = row.entity.name,
            e.type = row.entity.type,
            e.aliases = coalesce(row.entity.aliases, [])
        MERGE (c)-[m:MENTIONS]->(e)
        SET m.weight = coalesce(row.weight, 1.0),
            m.evidence = coalesce(row.evidence, '')
        """
        self.client.execute_write(cypher, {"rows": rows})

    def fetch_subgraph(
        self,
        *,
        kb_name: str,
        query: str,
        hops: int = 2,
        limit: int = 200,
    ) -> tuple[list[GraphNode], list[GraphEdge], bool]:
        normalized_limit = max(1, min(limit, 300))
        normalized_hops = max(1, min(hops, 3))
        cypher = """
        MATCH (anchor:Entity)
        WHERE anchor.kb_name = $kb_name
          AND toLower(anchor.name) CONTAINS toLower($query)
        WITH anchor LIMIT 10
        MATCH p=(anchor)-[rels:REL*1..3]-(nbr:Entity)
        WHERE length(rels) <= $hops AND nbr.kb_name = $kb_name
        WITH nodes(p) AS ns, relationships(p) AS rs
        UNWIND ns AS n
        WITH collect(DISTINCT n) AS all_nodes, collect(rs) AS rel_groups
        UNWIND rel_groups AS rg
        UNWIND rg AS r
        WITH all_nodes, collect(DISTINCT r) AS all_rels
        RETURN all_nodes[0..$limit] AS nodes,
               all_rels[0..$edge_limit] AS rels,
               size(all_nodes) > $limit OR size(all_rels) > $edge_limit AS truncated
        """
        rows = self.client.execute_read(
            cypher,
            {
                "kb_name": kb_name,
                "query": query,
                "hops": normalized_hops,
                "limit": normalized_limit,
                "edge_limit": min(800, normalized_limit * 3),
            },
        )
        if not rows:
            return [], [], False

        row = rows[0]
        raw_nodes = row.get("nodes") or []
        raw_rels = row.get("rels") or []
        nodes = [
            GraphNode(
                id=str(n.get("entity_id") or ""),
                label=str(n.get("name") or ""),
                type=str(n.get("type") or "entity"),
                metadata={"kb_name": n.get("kb_name", "")},
            )
            for n in raw_nodes
        ]
        edges = [
            GraphEdge(
                source=str(r.start_node.get("entity_id") or ""),
                target=str(r.end_node.get("entity_id") or ""),
                relation=str(r.get("type") or "REL"),
                weight=float(r.get("weight") or 1.0),
            )
            for r in raw_rels
        ]
        return nodes, edges, bool(row.get("truncated"))

    def fetch_overview_subgraph(
        self,
        *,
        kb_name: str,
        limit: int = 120,
    ) -> tuple[list[GraphNode], list[GraphEdge], bool]:
        normalized_limit = max(10, min(limit, 300))
        cypher = """
        MATCH (n:Entity {kb_name: $kb_name})
        OPTIONAL MATCH (n)-[r:REL]-(:Entity {kb_name: $kb_name})
        WITH n, count(r) AS degree
        ORDER BY degree DESC, n.name ASC
        LIMIT $limit
        WITH collect(n) AS top_nodes
        UNWIND top_nodes AS src
        OPTIONAL MATCH (src)-[r:REL]-(dst:Entity {kb_name: $kb_name})
        WHERE dst IN top_nodes
        WITH top_nodes, collect(DISTINCT {
            source: startNode(r).entity_id,
            target: endNode(r).entity_id,
            relation: coalesce(r.type, 'REL'),
            weight: coalesce(r.weight, 1.0)
        }) AS rels
        RETURN top_nodes AS nodes,
               rels[0..$edge_limit] AS rels,
               size(rels) > $edge_limit AS truncated
        """
        rows = self.client.execute_read(
            cypher,
            {
                "kb_name": kb_name,
                "limit": normalized_limit,
                "edge_limit": min(1000, normalized_limit * 5),
            },
        )
        if not rows:
            return [], [], False

        row = rows[0]
        raw_nodes = row.get("nodes") or []
        raw_rels = row.get("rels") or []
        nodes = [
            GraphNode(
                id=str(n.get("entity_id") or ""),
                label=str(n.get("name") or ""),
                type=str(n.get("type") or "entity"),
                metadata={"kb_name": n.get("kb_name", "")},
            )
            for n in raw_nodes
        ]
        edges: list[GraphEdge] = []
        for r in raw_rels:
            if not r:
                continue
            if isinstance(r, dict):
                source = str(r.get("source") or "")
                target = str(r.get("target") or "")
                relation = str(r.get("relation") or "REL")
                weight = float(r.get("weight") or 1.0)
            elif isinstance(r, (tuple, list)) and len(r) >= 2:
                source = str(r[0] or "")
                target = str(r[1] or "")
                relation = str(r[2] or "REL") if len(r) > 2 else "REL"
                weight = float(r[3] or 1.0) if len(r) > 3 else 1.0
            else:
                source = str(getattr(r, "start_node", {}).get("entity_id") or "")
                target = str(getattr(r, "end_node", {}).get("entity_id") or "")
                relation = str(r.get("type") or "REL")
                weight = float(r.get("weight") or 1.0)

            if not source or not target:
                continue

            edges.append(
                GraphEdge(
                    source=source,
                    target=target,
                    relation=relation,
                    weight=weight,
                )
            )
        return nodes, edges, bool(row.get("truncated"))
