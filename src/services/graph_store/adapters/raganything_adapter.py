# -*- coding: utf-8 -*-
from .lightrag_adapter import LightRAGNeo4jAdapter


class RAGAnythingNeo4jAdapter(LightRAGNeo4jAdapter):
    provider = "raganything"

