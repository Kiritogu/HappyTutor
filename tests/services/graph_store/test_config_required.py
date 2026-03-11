from pathlib import Path

import pytest

from src.services.graph_store.neo4j_client import load_graph_store_settings


def test_graph_store_settings_require_env_or_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    # Use a temporary root without config/main.yaml to force missing config path.
    fake_root = Path(__file__).resolve().parent / "missing_root"
    with pytest.raises(ValueError, match="Neo4j configuration missing required settings"):
        load_graph_store_settings(project_root=fake_root)

