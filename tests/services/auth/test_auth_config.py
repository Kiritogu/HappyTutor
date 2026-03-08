from src.services.auth.config import get_auth_settings


def test_auth_settings_defaults():
    settings = get_auth_settings()
    assert settings.jwt_issuer == "deeptutor"
    assert settings.jwt_access_ttl_seconds == 900
