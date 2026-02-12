from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from src.api.dependencies.auth import get_auth_service
from src.api.routers import auth


class FakeAuthService:
    def __init__(self) -> None:
        self.users: dict[str, dict[str, Any]] = {}
        self.refresh_state: dict[str, bool] = {}

    def register(self, *, email: str, password: str) -> dict[str, Any]:
        if email in self.users:
            raise ValueError("email already exists")
        user = {
            "id": f"user-{len(self.users) + 1}",
            "email": email,
            "is_email_verified": False,
            "status": "active",
            "password": password,
        }
        self.users[email] = user
        return {"user": user, "requires_email_verification": True}

    def login(self, *, email: str, password: str, created_ip: str | None = None, user_agent: str | None = None):
        user = self.users.get(email)
        if not user or user["password"] != password:
            raise ValueError("invalid credentials")
        refresh = f"refresh-{user['id']}-1"
        self.refresh_state[refresh] = True
        return {
            "access_token": f"access-{user['id']}",
            "refresh_token": refresh,
            "token_type": "bearer",
            "expires_in": 900,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "is_email_verified": user["is_email_verified"],
                "status": user["status"],
            },
        }

    def refresh_tokens(
        self,
        refresh_token: str,
        *,
        created_ip: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        if not self.refresh_state.get(refresh_token):
            raise ValueError("refresh token inactive")
        self.refresh_state[refresh_token] = False

        user_id = refresh_token.split("-")[1]
        next_refresh = f"refresh-{user_id}-2"
        self.refresh_state[next_refresh] = True

        email = next(iter(self.users.keys()))
        user = self.users[email]
        return {
            "access_token": f"access-{user_id}",
            "refresh_token": next_refresh,
            "token_type": "bearer",
            "expires_in": 900,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "is_email_verified": user["is_email_verified"],
                "status": user["status"],
            },
        }

    def logout(self, refresh_token: str) -> bool:
        if refresh_token not in self.refresh_state:
            return False
        self.refresh_state[refresh_token] = False
        return True

    def get_current_user(self, access_token: str) -> dict[str, Any]:
        if not access_token.startswith("access-"):
            raise ValueError("invalid token")
        user_id = access_token.replace("access-", "", 1)
        for user in self.users.values():
            if user["id"] == user_id:
                return {
                    "id": user["id"],
                    "email": user["email"],
                    "is_email_verified": user["is_email_verified"],
                    "status": user["status"],
                }
        raise ValueError("user not found")


@pytest.fixture
def auth_client() -> tuple[TestClient, FakeAuthService]:
    app = FastAPI()
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])

    fake_service = FakeAuthService()
    app.dependency_overrides[get_auth_service] = lambda: fake_service

    return TestClient(app), fake_service


def test_register_login_me_refresh_logout(auth_client: tuple[TestClient, FakeAuthService]):
    client, _ = auth_client

    register = client.post(
        "/api/v1/auth/register",
        json={"email": "a@example.com", "password": "Pass!23456"},
    )
    assert register.status_code == 200

    login = client.post(
        "/api/v1/auth/login",
        json={"email": "a@example.com", "password": "Pass!23456"},
    )
    assert login.status_code == 200
    login_body = login.json()
    access_token = login_body["access_token"]
    refresh_token = login_body["refresh_token"]

    me = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert me.status_code == 200
    assert me.json()["user"]["email"] == "a@example.com"

    refreshed = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert refreshed.status_code == 200
    new_refresh_token = refreshed.json()["refresh_token"]
    assert new_refresh_token != refresh_token

    logout = client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": new_refresh_token},
    )
    assert logout.status_code == 200
    assert logout.json()["success"] is True


def test_me_requires_bearer_token(auth_client: tuple[TestClient, FakeAuthService]):
    client, _ = auth_client

    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401

