from __future__ import annotations

import asyncio

import pytest
from fastapi import WebSocketDisconnect

from src.api.routers import knowledge


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False

    async def accept(self) -> None:
        return None

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)

    async def receive_text(self) -> str:
        raise WebSocketDisconnect()

    async def close(self) -> None:
        self.closed = True


class _FakeBroadcaster:
    async def connect(self, kb_name: str, websocket: _FakeWebSocket) -> None:
        return None

    async def disconnect(self, kb_name: str, websocket: _FakeWebSocket) -> None:
        return None


class _FakeProgressTracker:
    def __init__(self, kb_name: str, base_dir) -> None:
        self.kb_name = kb_name
        self.base_dir = base_dir

    def get_progress(self):
        return None


class _FakeKBManager:
    def get_kb_info(self, kb_name: str) -> dict:
        return {"name": kb_name, "status": "ready"}


@pytest.mark.asyncio
async def test_progress_ws_does_not_emit_name_error(monkeypatch: pytest.MonkeyPatch):
    fake_ws = _FakeWebSocket()
    fake_broadcaster = _FakeBroadcaster()

    monkeypatch.setattr(
        "src.api.routers.knowledge.ProgressBroadcaster.get_instance",
        lambda: fake_broadcaster,
    )
    monkeypatch.setattr("src.api.routers.knowledge.ProgressTracker", _FakeProgressTracker)
    monkeypatch.setattr("src.api.routers.knowledge.get_kb_manager", lambda: _FakeKBManager())

    await knowledge.websocket_progress(fake_ws, "RAG面试")

    assert fake_ws.closed is True
    assert not any(
        msg.get("type") == "error" and "manager" in str(msg.get("message", ""))
        for msg in fake_ws.sent
    )
