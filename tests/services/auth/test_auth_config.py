import pytest

from src.services.auth.config import get_auth_settings


def test_auth_settings_require_postgres(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEEPTUTOR_STORAGE_BACKEND", "sqlite")

    with pytest.raises(ValueError, match="postgres"):
        get_auth_settings()

