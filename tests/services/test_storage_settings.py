from pathlib import Path

import pytest
import yaml

from src.services.storage.user_db import get_storage_settings, get_user_db


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_storage_settings_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_yaml(
        tmp_path / "config" / "main.yaml",
        {"storage": {"postgres_dsn": "", "auto_migrate": True}},
    )

    monkeypatch.setenv(
        "DEEPTUTOR_POSTGRES_DSN",
        "postgresql://user:pass@localhost:5432/deeptutor_test",
    )

    settings = get_storage_settings(project_root=tmp_path)
    assert settings.postgres_dsn == "postgresql://user:pass@localhost:5432/deeptutor_test"


def test_get_user_db_postgres_requires_dsn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_yaml(tmp_path / "config" / "main.yaml", {"storage": {}})
    monkeypatch.delenv("DEEPTUTOR_POSTGRES_DSN", raising=False)

    with pytest.raises(ValueError):
        get_user_db(project_root=tmp_path)
