"""
core/security.py
----------------
Password hashing (bcrypt) and access tokens (JWT, signed with SECRET_KEY).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache

import bcrypt
import jwt

from src.core.config import settings

# bcrypt only uses the first 72 bytes of a password and bcrypt>=5 rejects
# longer input, so the limit is enforced explicitly.
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LENGTH = 8


class InvalidToken(Exception):
    """Token missing, malformed, expired or signed with another key."""


def validate_new_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must have at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValueError(f"Password must be at most {MAX_PASSWORD_BYTES} bytes.")


def hash_password(password: str) -> str:
    validate_new_password(password)
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except ValueError:          # over-long password or malformed hash
        return False


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    return bcrypt.hashpw(b"timing-equaliser", bcrypt.gensalt()).decode("ascii")


def burn_password_check(password: str) -> None:
    """
    Spend the same time as a real check when the username does not exist,
    so response times do not reveal which usernames are registered.
    """
    verify_password(password, _dummy_hash())


def create_access_token(username: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub":  username,
        "role": role,
        "iat":  now,
        "exp":  now + timedelta(minutes=settings.access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
            options={"require": ["exp", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise InvalidToken(str(exc)) from exc