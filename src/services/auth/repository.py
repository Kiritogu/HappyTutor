from __future__ import annotations

from typing import Any


class PostgresAuthRepository:
    def __init__(self, db: Any) -> None:
        self._db = db

    def create_user(self, *, email: str, password_hash: str) -> dict[str, Any]:
        return self._db.auth_create_user(email=email, password_hash=password_hash)

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        return self._db.auth_get_user_by_email(email)

    def get_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        return self._db.auth_get_user_by_id(user_id)

    def update_password_hash(self, *, user_id: str, password_hash: str) -> bool:
        return self._db.auth_update_password_hash(user_id=user_id, password_hash=password_hash)

    def set_email_verified(self, *, user_id: str) -> bool:
        return self._db.auth_set_email_verified(user_id=user_id)

    def update_last_login_at(self, *, user_id: str) -> bool:
        return self._db.auth_update_last_login_at(user_id=user_id)

    def create_refresh_token(
        self,
        *,
        user_id: str,
        token_hash: str,
        expires_at: float,
        token_id: str | None = None,
        created_ip: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "user_id": user_id,
            "token_hash": token_hash,
            "expires_at": expires_at,
        }
        if token_id is not None:
            payload["token_id"] = token_id
        if created_ip is not None:
            payload["created_ip"] = created_ip
        if user_agent is not None:
            payload["user_agent"] = user_agent
        return self._db.auth_create_refresh_token(**payload)

    def get_refresh_token(self, token_id: str) -> dict[str, Any] | None:
        return self._db.auth_get_refresh_token(token_id)

    def get_refresh_token_by_hash(self, token_hash: str) -> dict[str, Any] | None:
        return self._db.auth_get_refresh_token_by_hash(token_hash)

    def is_refresh_token_active(self, token_id: str) -> bool:
        return bool(self._db.auth_is_refresh_token_active(token_id))

    def revoke_refresh_token(self, token_id: str) -> bool:
        return bool(self._db.auth_revoke_refresh_token(token_id))

    def revoke_all_refresh_tokens(self, *, user_id: str) -> int:
        return int(self._db.auth_revoke_all_refresh_tokens(user_id=user_id))

    def create_email_verification_token(
        self,
        *,
        user_id: str,
        token_hash: str,
        expires_at: float,
    ) -> dict[str, Any]:
        return self._db.auth_create_email_verification_token(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
        )

    def consume_email_verification_token(self, token_hash: str) -> dict[str, Any] | None:
        return self._db.auth_consume_email_verification_token(token_hash=token_hash)

    def create_password_reset_token(
        self,
        *,
        user_id: str,
        token_hash: str,
        expires_at: float,
    ) -> dict[str, Any]:
        return self._db.auth_create_password_reset_token(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
        )

    def consume_password_reset_token(self, token_hash: str) -> dict[str, Any] | None:
        return self._db.auth_consume_password_reset_token(token_hash=token_hash)
