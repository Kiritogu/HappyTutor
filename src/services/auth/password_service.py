from __future__ import annotations

from passlib.context import CryptContext


class PasswordService:
    def __init__(self) -> None:
        self._context = CryptContext(schemes=["argon2"], deprecated="auto")

    def hash_password(self, plain_password: str) -> str:
        if not plain_password:
            raise ValueError("Password must not be empty")
        return self._context.hash(plain_password)

    def verify_password(self, plain_password: str, password_hash: str) -> bool:
        if not plain_password or not password_hash:
            return False
        return bool(self._context.verify(plain_password, password_hash))

