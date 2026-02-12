from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import chat, dashboard, notebook, settings


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(notebook.router, prefix="/api/v1/notebook", tags=["notebook"])
    app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["dashboard"])
    app.include_router(settings.router, prefix="/api/v1/settings", tags=["settings"])
    app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
    return TestClient(app)


def test_notebook_requires_auth():
    client = _build_client()
    response = client.get("/api/v1/notebook/list")
    assert response.status_code == 401


def test_dashboard_requires_auth():
    client = _build_client()
    response = client.get("/api/v1/dashboard/recent")
    assert response.status_code == 401


def test_settings_requires_auth():
    client = _build_client()
    response = client.get("/api/v1/settings")
    assert response.status_code == 401


def test_chat_sessions_requires_auth():
    client = _build_client()
    response = client.get("/api/v1/chat/sessions")
    assert response.status_code == 401
