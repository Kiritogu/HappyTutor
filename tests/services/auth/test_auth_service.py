from __future__ import annotations

import hashlib
import time
from typing import Any

import pytest

from src.services.auth.auth_service import AuthService
from src.services.auth.config import AuthSettings
from src.services.auth.password_service import PasswordService
from src.services.auth.token_service import TokenService


class FakeAuthRepository:
    def __init__(self) -> None:
        self.users: dict[str, dict[str, Any]] = {}
        self.users_by_email: dict[str, str] = {}
        self.refresh_tokens: dict[str, dict[str, Any]] = {}

    def create_user(self, *, email: str, password_hash: str) -> dict[str, Any]:
        if email in self.users_by_email:
            raise ValueError("email already exists")
        user_id = f"user-{len(self.users) + 1}"
        now = time.time()
        user = {
            "id": user_id,
            "email": email,
            "password_hash": password_hash,
            "is_email_verified": False,
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "last_login_at": None,
        }
        self.users[user_id] = user
        self.users_by_email[email] = user_id
        return user

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        user_id = self.users_by_email.get(email)
        if not user_id:
            return None
        return self.users.get(user_id)

    def get_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        return self.users.get(user_id)

    def update_last_login_at(self, *, user_id: str) -> bool:
        user = self.users.get(user_id)
        if not user:
            return False
        user["last_login_at"] = time.time()
        return True

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
        token_id = token_id or f"rt-{len(self.refresh_tokens) + 1}"
        token = {
            "id": token_id,
            "user_id": user_id,
            "token_hash": token_hash,
            "expires_at": expires_at,
            "revoked_at": None,
            "created_ip": created_ip,
            "user_agent": user_agent,
        }
        self.refresh_tokens[token_id] = token
        return token

    def get_refresh_token(self, token_id: str) -> dict[str, Any] | None:
        return self.refresh_tokens.get(token_id)

    def get_refresh_token_by_hash(self, token_hash: str) -> dict[str, Any] | None:
        for token in self.refresh_tokens.values():
            if token["token_hash"] == token_hash:
                return token
        return None

    def is_refresh_token_active(self, token_id: str) -> bool:
        token = self.refresh_tokens.get(token_id)
        if not token:
            return False
        return token["revoked_at"] is None and token["expires_at"] > time.time()

    def revoke_refresh_token(self, token_id: str) -> bool:
        token = self.refresh_tokens.get(token_id)
        if not token or token["revoked_at"] is not None:
            return False
        token["revoked_at"] = time.time()
        return True


@pytest.fixture
def auth_settings() -> AuthSettings:
    return AuthSettings(
        jwt_secret="service-test-secret",
        jwt_issuer="deeptutor-test",
        jwt_access_ttl_seconds=900,
        jwt_refresh_ttl_seconds=1209600,
        email_verification_ttl_seconds=3600,
        password_reset_ttl_seconds=3600,
    )


@pytest.fixture
def auth_service(auth_settings: AuthSettings) -> AuthService:
    repository = FakeAuthRepository()
    password_service = PasswordService()
    token_service = TokenService(auth_settings)
    return AuthService(
        repository=repository,
        password_service=password_service,
        token_service=token_service,
        settings=auth_settings,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_register_creates_user_and_verification_state(auth_service: AuthService):
    result = auth_service.register(email="a@example.com", password="Pass!23456")

    assert result["requires_email_verification"] is True
    assert result["user"]["email"] == "a@example.com"


def test_login_returns_access_and_refresh(auth_service: AuthService):
    auth_service.register(email="a@example.com", password="Pass!23456")

    tokens = auth_service.login(email="a@example.com", password="Pass!23456")

    assert "access_token" in tokens
    assert "refresh_token" in tokens
    assert tokens["user"]["email"] == "a@example.com"


def test_refresh_rotates_refresh_token(auth_service: AuthService):
    auth_service.register(email="a@example.com", password="Pass!23456")
    tokens = auth_service.login(email="a@example.com", password="Pass!23456")

    old_payload = auth_service.token_service.decode_token(tokens["refresh_token"])
    old_token_id = old_payload["jti"]
    old_hash = _sha256(tokens["refresh_token"])
    old_row = auth_service.repository.get_refresh_token(old_token_id)
    assert old_row is not None
    assert old_row["token_hash"] == old_hash

    refreshed = auth_service.refresh_tokens(tokens["refresh_token"])

    new_payload = auth_service.token_service.decode_token(refreshed["refresh_token"])
    new_token_id = new_payload["jti"]
    assert new_token_id != old_token_id
    assert not auth_service.repository.is_refresh_token_active(old_token_id)
    assert auth_service.repository.is_refresh_token_active(new_token_id)


def test_logout_revokes_refresh_token(auth_service: AuthService):
    auth_service.register(email="a@example.com", password="Pass!23456")
    tokens = auth_service.login(email="a@example.com", password="Pass!23456")
    payload = auth_service.token_service.decode_token(tokens["refresh_token"])
    token_id = payload["jti"]

    auth_service.logout(tokens["refresh_token"])

    assert not auth_service.repository.is_refresh_token_active(token_id)


def test_get_current_user_from_access_token(auth_service: AuthService):
    auth_service.register(email="a@example.com", password="Pass!23456")
    tokens = auth_service.login(email="a@example.com", password="Pass!23456")

    current_user = auth_service.get_current_user(tokens["access_token"])

    assert current_user["email"] == "a@example.com"

