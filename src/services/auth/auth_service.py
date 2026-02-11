from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import uuid
from typing import Any

from src.services.auth.config import AuthSettings
from src.services.auth.password_service import PasswordService
from src.services.auth.token_service import TokenService


class AuthService:
    def __init__(
        self,
        *,
        repository: Any,
        password_service: PasswordService,
        token_service: TokenService,
        settings: AuthSettings,
    ) -> None:
        self.repository = repository
        self.password_service = password_service
        self.token_service = token_service
        self.settings = settings

    @staticmethod
    def _hash_token(raw_token: str) -> str:
        return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    def register(self, *, email: str, password: str) -> dict[str, Any]:
        existing = self.repository.get_user_by_email(email)
        if existing is not None:
            raise ValueError("email already exists")

        password_hash = self.password_service.hash_password(password)
        user = self.repository.create_user(email=email, password_hash=password_hash)

        return {
            "user": {
                "id": user["id"],
                "email": user["email"],
                "is_email_verified": bool(user.get("is_email_verified", False)),
                "status": user.get("status", "active"),
            },
            "requires_email_verification": True,
        }

    def _issue_tokens_for_user(
        self,
        *,
        user: dict[str, Any],
        created_ip: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        refresh_token_id = str(uuid.uuid4())
        access_token = self.token_service.create_access_token(user_id=user["id"], email=user["email"])
        refresh_token = self.token_service.create_refresh_token(
            user_id=user["id"],
            token_id=refresh_token_id,
        )

        refresh_exp = datetime.now(timezone.utc) + timedelta(seconds=self.settings.jwt_refresh_ttl_seconds)
        self.repository.create_refresh_token(
            user_id=user["id"],
            token_hash=self._hash_token(refresh_token),
            expires_at=refresh_exp.timestamp(),
            token_id=refresh_token_id,
            created_ip=created_ip,
            user_agent=user_agent,
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.settings.jwt_access_ttl_seconds,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "is_email_verified": bool(user.get("is_email_verified", False)),
                "status": user.get("status", "active"),
            },
        }

    def login(
        self,
        *,
        email: str,
        password: str,
        created_ip: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        user = self.repository.get_user_by_email(email)
        if user is None:
            raise ValueError("invalid credentials")
        if user.get("status") != "active":
            raise ValueError("account disabled")
        if not self.password_service.verify_password(password, user.get("password_hash", "")):
            raise ValueError("invalid credentials")

        self.repository.update_last_login_at(user_id=user["id"])
        return self._issue_tokens_for_user(
            user=user,
            created_ip=created_ip,
            user_agent=user_agent,
        )

    def refresh_tokens(
        self,
        refresh_token: str,
        *,
        created_ip: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        payload = self.token_service.decode_token(refresh_token)
        if payload.get("token_type") != "refresh":
            raise ValueError("invalid token type")

        refresh_token_id = str(payload.get("jti") or "")
        if not refresh_token_id:
            raise ValueError("missing refresh token id")

        row = self.repository.get_refresh_token(refresh_token_id)
        if row is None:
            raise ValueError("refresh token not found")

        expected_hash = self._hash_token(refresh_token)
        if row.get("token_hash") != expected_hash:
            raise ValueError("refresh token hash mismatch")

        if not self.repository.is_refresh_token_active(refresh_token_id):
            raise ValueError("refresh token inactive")

        self.repository.revoke_refresh_token(refresh_token_id)

        user = self.repository.get_user_by_id(str(payload["sub"]))
        if user is None:
            raise ValueError("user not found")
        if user.get("status") != "active":
            raise ValueError("account disabled")

        return self._issue_tokens_for_user(
            user=user,
            created_ip=created_ip,
            user_agent=user_agent,
        )

    def logout(self, refresh_token: str) -> bool:
        payload = self.token_service.decode_token(refresh_token)
        if payload.get("token_type") != "refresh":
            raise ValueError("invalid token type")

        refresh_token_id = str(payload.get("jti") or "")
        if not refresh_token_id:
            return False
        return bool(self.repository.revoke_refresh_token(refresh_token_id))

    def get_current_user(self, access_token: str) -> dict[str, Any]:
        payload = self.token_service.decode_token(access_token)
        if payload.get("token_type") != "access":
            raise ValueError("invalid token type")

        user_id = str(payload.get("sub") or "")
        if not user_id:
            raise ValueError("missing user id")

        user = self.repository.get_user_by_id(user_id)
        if user is None:
            raise ValueError("user not found")
        if user.get("status") != "active":
            raise ValueError("account disabled")

        return {
            "id": user["id"],
            "email": user["email"],
            "is_email_verified": bool(user.get("is_email_verified", False)),
            "status": user.get("status", "active"),
        }

