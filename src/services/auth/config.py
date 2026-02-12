from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from src.services.storage.user_db import get_storage_settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _as_int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    return int(value)


@dataclass(frozen=True)
class AuthSettings:
    jwt_secret: str
    jwt_issuer: str
    jwt_access_ttl_seconds: int
    jwt_refresh_ttl_seconds: int
    email_verification_ttl_seconds: int
    password_reset_ttl_seconds: int


def get_auth_settings(project_root: Path | None = None) -> AuthSettings:
    if project_root is None:
        project_root = PROJECT_ROOT

    storage_settings = get_storage_settings(project_root=project_root)
    if storage_settings.backend != "postgres":
        raise ValueError("Authentication currently requires postgres backend.")

    return AuthSettings(
        jwt_secret=os.getenv("AUTH_JWT_SECRET", ""),
        jwt_issuer=os.getenv("AUTH_JWT_ISSUER", "deeptutor"),
        jwt_access_ttl_seconds=_as_int(os.getenv("AUTH_ACCESS_TTL_SECONDS"), 900),
        jwt_refresh_ttl_seconds=_as_int(os.getenv("AUTH_REFRESH_TTL_SECONDS"), 1_209_600),
        email_verification_ttl_seconds=_as_int(
            os.getenv("AUTH_EMAIL_VERIFICATION_TTL_SECONDS"), 3_600
        ),
        password_reset_ttl_seconds=_as_int(os.getenv("AUTH_PASSWORD_RESET_TTL_SECONDS"), 3_600),
    )

