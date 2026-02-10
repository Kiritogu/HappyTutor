from __future__ import annotations

from functools import lru_cache

from fastapi import Header, HTTPException, status

from src.services.auth.auth_service import AuthService
from src.services.auth.config import get_auth_settings
from src.services.auth.password_service import PasswordService
from src.services.auth.repository import PostgresAuthRepository
from src.services.auth.token_service import TokenService
from src.services.storage import get_storage_settings, get_user_db


@lru_cache(maxsize=1)
def _build_auth_service() -> AuthService:
    settings = get_auth_settings()
    storage_settings = get_storage_settings()
    if storage_settings.backend != "postgres":
        raise ValueError("Authentication requires postgres backend")

    db = get_user_db()
    repository = PostgresAuthRepository(db)
    password_service = PasswordService()
    token_service = TokenService(settings)
    return AuthService(
        repository=repository,
        password_service=password_service,
        token_service=token_service,
        settings=settings,
    )


def get_auth_service() -> AuthService:
    try:
        return _build_auth_service()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication service unavailable: {exc}",
        ) from exc


def extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header",
        )
    return token.strip()


def get_current_user(
    *,
    auth_service: AuthService,
    authorization: str | None,
) -> dict:
    token = extract_bearer_token(authorization)
    try:
        return auth_service.get_current_user(token)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc


def get_current_user_from_header(
    authorization: str | None = Header(default=None),
) -> dict:
    service = get_auth_service()
    return get_current_user(auth_service=service, authorization=authorization)

