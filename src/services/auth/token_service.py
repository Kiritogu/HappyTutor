from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from src.services.auth.config import AuthSettings


class TokenService:
    def __init__(self, settings: AuthSettings) -> None:
        if not settings.jwt_secret:
            raise ValueError("AUTH_JWT_SECRET must be configured")
        self._settings = settings
        self._algorithm = "HS256"

    def _base_payload(self, *, user_id: str, token_type: str, ttl_seconds: int) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        exp = now + timedelta(seconds=ttl_seconds)
        return {
            "sub": user_id,
            "iss": self._settings.jwt_issuer,
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
            "token_type": token_type,
        }

    def create_access_token(self, *, user_id: str, email: str) -> str:
        payload = self._base_payload(
            user_id=user_id,
            token_type="access",
            ttl_seconds=self._settings.jwt_access_ttl_seconds,
        )
        payload["email"] = email
        return jwt.encode(payload, self._settings.jwt_secret, algorithm=self._algorithm)

    def create_refresh_token(self, *, user_id: str, token_id: str) -> str:
        payload = self._base_payload(
            user_id=user_id,
            token_type="refresh",
            ttl_seconds=self._settings.jwt_refresh_ttl_seconds,
        )
        payload["jti"] = token_id
        return jwt.encode(payload, self._settings.jwt_secret, algorithm=self._algorithm)

    def decode_token(self, token: str) -> dict[str, Any]:
        payload = jwt.decode(
            token,
            self._settings.jwt_secret,
            algorithms=[self._algorithm],
            issuer=self._settings.jwt_issuer,
            options={"require": ["exp", "iat", "sub", "iss", "token_type"]},
        )
        if not isinstance(payload, dict):
            raise ValueError("Decoded JWT payload must be a dict")
        return payload

