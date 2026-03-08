from __future__ import annotations

"""
Database-backed storage for structured user data (PostgreSQL).

This module provides the storage configuration and factory for PostgresUserDB,
which stores:
- Activity history
- Notebooks
- Chat sessions
- UI settings

Large artifacts (markdown reports, images, etc.) remain file-based and are served from
`data/user/` via `/api/outputs`.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import threading
from typing import Any

from dotenv import load_dotenv

from src.services.config import load_config_with_main

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / "DeepTutor.env", override=False)
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class StorageSettings:
    postgres_dsn: str
    auto_migrate: bool


def get_storage_settings(project_root: Path | None = None) -> StorageSettings:
    """
    Get PostgreSQL storage settings.

    Priority:
    1) Environment variables
    2) config/main.yaml ("storage" section)
    3) Built-in defaults
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    cfg: dict[str, Any] = {}
    try:
        cfg = load_config_with_main("solve_config.yaml", project_root)
    except Exception:
        cfg = {}

    storage_cfg = cfg.get("storage", {}) if isinstance(cfg, dict) else {}

    default_postgres_dsn = str(storage_cfg.get("postgres_dsn", "") or "")
    default_auto_migrate = bool(storage_cfg.get("auto_migrate", True))

    postgres_dsn = os.getenv("DEEPTUTOR_POSTGRES_DSN") or default_postgres_dsn
    auto_migrate = _parse_bool(os.getenv("DEEPTUTOR_STORAGE_AUTO_MIGRATE"), default_auto_migrate)

    return StorageSettings(
        postgres_dsn=postgres_dsn,
        auto_migrate=auto_migrate,
    )


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


_user_db_lock = threading.Lock()
_user_db_instance: Any | None = None
_user_db_key: str | None = None


def get_user_db(project_root: Path | None = None) -> Any:
    """
    Get the PostgresUserDB singleton.

    Raises:
        ValueError: If postgres_dsn is not configured.
    """
    settings = get_storage_settings(project_root=project_root)

    if not settings.postgres_dsn:
        raise ValueError(
            "DEEPTUTOR_POSTGRES_DSN is required. "
            "Set it via environment variable or in config/main.yaml (storage.postgres_dsn)."
        )

    global _user_db_instance, _user_db_key
    with _user_db_lock:
        if _user_db_instance is None or _user_db_key != settings.postgres_dsn:
            if _user_db_instance is not None:
                try:
                    _user_db_instance.close()
                except Exception:
                    pass

            from .postgres_db import PostgresUserDB

            _user_db_instance = PostgresUserDB(
                settings.postgres_dsn,
                auto_migrate=settings.auto_migrate,
                project_root=project_root or PROJECT_ROOT,
            )
            _user_db_key = settings.postgres_dsn

    return _user_db_instance
