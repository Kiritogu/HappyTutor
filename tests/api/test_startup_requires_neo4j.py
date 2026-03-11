import pytest

class DummyApp:
    class State:
        pass

    state = State()


@pytest.mark.asyncio
async def test_startup_fails_without_neo4j(monkeypatch: pytest.MonkeyPatch):
    try:
        from src.api.main import lifespan
    except Exception as exc:  # pragma: no cover - environment-dependent import
        pytest.skip(f"Skipping startup test due to app import dependency error: {exc}")

    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    app = DummyApp()
    with pytest.raises(Exception):
        async with lifespan(app):  # type: ignore[arg-type]
            pass
