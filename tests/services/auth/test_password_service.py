from src.services.auth.password_service import PasswordService


def test_hash_and_verify_password():
    service = PasswordService()

    hashed = service.hash_password("StrongPass!23")
    assert hashed != "StrongPass!23"
    assert service.verify_password("StrongPass!23", hashed)
    assert not service.verify_password("wrong-password", hashed)

