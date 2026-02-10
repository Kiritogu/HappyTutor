from src.services.auth.config import AuthSettings
from src.services.auth.token_service import TokenService


def _settings() -> AuthSettings:
    return AuthSettings(
        jwt_secret="unit-test-secret",
        jwt_issuer="deeptutor-test",
        jwt_access_ttl_seconds=900,
        jwt_refresh_ttl_seconds=1209600,
        email_verification_ttl_seconds=3600,
        password_reset_ttl_seconds=3600,
    )


def test_access_token_contains_sub_and_type():
    service = TokenService(_settings())

    token = service.create_access_token(user_id="u1", email="a@example.com")
    payload = service.decode_token(token)

    assert payload["sub"] == "u1"
    assert payload["token_type"] == "access"
    assert payload["email"] == "a@example.com"


def test_refresh_token_contains_jti_and_type():
    service = TokenService(_settings())

    token = service.create_refresh_token(user_id="u1", token_id="token-row-id")
    payload = service.decode_token(token)

    assert payload["sub"] == "u1"
    assert payload["token_type"] == "refresh"
    assert payload["jti"] == "token-row-id"

