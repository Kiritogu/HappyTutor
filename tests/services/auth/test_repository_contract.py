from __future__ import annotations

from src.services.auth.repository import PostgresAuthRepository
from src.services.storage.postgres_db import PostgresUserDB


class _FakeBackend:
    def __init__(self) -> None:
        self.users: dict[str, dict] = {}
        self.users_by_email: dict[str, str] = {}
        self.refresh_tokens: dict[str, dict] = {}

    def auth_create_user(self, *, email: str, password_hash: str) -> dict:
        user = {
            "id": f"user-{len(self.users) + 1}",
            "email": email,
            "password_hash": password_hash,
            "is_email_verified": False,
            "status": "active",
        }
        self.users[user["id"]] = user
        self.users_by_email[email] = user["id"]
        return user

    def auth_get_user_by_email(self, email: str) -> dict | None:
        user_id = self.users_by_email.get(email)
        if not user_id:
            return None
        return self.users.get(user_id)

    def auth_create_refresh_token(self, *, user_id: str, token_hash: str, expires_at: float) -> dict:
        token = {
            "id": f"rt-{len(self.refresh_tokens) + 1}",
            "user_id": user_id,
            "token_hash": token_hash,
            "expires_at": expires_at,
            "revoked_at": None,
        }
        self.refresh_tokens[token["id"]] = token
        return token

    def auth_is_refresh_token_active(self, token_id: str) -> bool:
        token = self.refresh_tokens.get(token_id)
        return bool(token and token.get("revoked_at") is None)

    def auth_revoke_refresh_token(self, token_id: str) -> bool:
        token = self.refresh_tokens.get(token_id)
        if not token:
            return False
        token["revoked_at"] = 1.0
        return True


def test_create_and_get_user():
    repository = PostgresAuthRepository(_FakeBackend())

    user = repository.create_user(email="u@example.com", password_hash="h")
    fetched = repository.get_user_by_email("u@example.com")

    assert fetched is not None
    assert fetched["id"] == user["id"]


def test_store_and_revoke_refresh_token():
    repository = PostgresAuthRepository(_FakeBackend())

    token = repository.create_refresh_token(user_id="u1", token_hash="h", expires_at=9_999_999_999.0)
    assert repository.is_refresh_token_active(token["id"])

    repository.revoke_refresh_token(token["id"])
    assert not repository.is_refresh_token_active(token["id"])


def test_postgres_db_exposes_auth_methods():
    required_methods = [
        "auth_create_user",
        "auth_get_user_by_email",
        "auth_get_user_by_id",
        "auth_update_password_hash",
        "auth_set_email_verified",
        "auth_update_last_login_at",
        "auth_create_refresh_token",
        "auth_get_refresh_token",
        "auth_get_refresh_token_by_hash",
        "auth_is_refresh_token_active",
        "auth_revoke_refresh_token",
        "auth_revoke_all_refresh_tokens",
        "auth_create_email_verification_token",
        "auth_consume_email_verification_token",
        "auth_create_password_reset_token",
        "auth_consume_password_reset_token",
    ]

    for method_name in required_methods:
        assert hasattr(PostgresUserDB, method_name), f"missing PostgresUserDB method: {method_name}"

