# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from src.logging import get_logger
from src.services.config import load_config_with_main


@dataclass(frozen=True)
class GraphStoreSettings:
    uri: str
    user: str
    password: str
    database: str = "neo4j"
    max_connection_pool_size: int = 50


def load_graph_store_settings(project_root: Path | None = None) -> GraphStoreSettings:
    """
    Load Neo4j settings from env first, then config/main.yaml.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent.parent

    cfg = load_config_with_main("solve_config.yaml", project_root)
    neo_cfg = cfg.get("neo4j", {}) if isinstance(cfg, dict) else {}

    uri = os.getenv("NEO4J_URI") or str(neo_cfg.get("uri") or "").strip()
    user = os.getenv("NEO4J_USER") or str(neo_cfg.get("user") or "").strip()
    password = os.getenv("NEO4J_PASSWORD") or str(neo_cfg.get("password") or "").strip()
    database = os.getenv("NEO4J_DATABASE") or str(neo_cfg.get("database") or "neo4j")
    pool_size_raw: Any = neo_cfg.get("max_connection_pool_size", 50)
    try:
        pool_size = int(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE") or pool_size_raw)
    except (TypeError, ValueError):
        pool_size = 50

    missing = [k for k, v in {"NEO4J_URI": uri, "NEO4J_USER": user, "NEO4J_PASSWORD": password}.items() if not v]
    if missing:
        raise ValueError(
            "Neo4j configuration missing required settings: "
            + ", ".join(missing)
            + ". Set env vars or config/main.yaml neo4j section."
        )

    return GraphStoreSettings(
        uri=uri,
        user=user,
        password=password,
        database=database,
        max_connection_pool_size=pool_size,
    )


class Neo4jClient:
    def __init__(self, settings: GraphStoreSettings):
        self.settings = settings
        self.logger = get_logger("Neo4jClient")
        self._driver = None

    def connect(self) -> None:
        if self._driver is not None:
            return
        self._driver = GraphDatabase.driver(
            self.settings.uri,
            auth=(self.settings.user, self.settings.password),
            max_connection_pool_size=self.settings.max_connection_pool_size,
        )

    def close(self) -> None:
        if self._driver is None:
            return
        self._driver.close()
        self._driver = None

    def verify_connectivity(self) -> None:
        if self._driver is None:
            self.connect()
        try:
            self._driver.verify_connectivity()
        except Neo4jError:
            self.close()
            raise

    def execute_write(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if self._driver is None:
            self.connect()
        with self._driver.session(database=self.settings.database) as session:
            result = session.run(cypher, parameters or {})
            return [record.data() for record in result]

    def execute_read(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if self._driver is None:
            self.connect()
        with self._driver.session(database=self.settings.database) as session:
            result = session.run(cypher, parameters or {})
            return [record.data() for record in result]

