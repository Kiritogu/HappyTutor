# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source_doc: str = ""
    provider: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphNode:
    id: str
    label: str
    type: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

