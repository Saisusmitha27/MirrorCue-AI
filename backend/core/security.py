from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext

from backend.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str, expires_delta_days: int | None = None, extra_claims: dict[str, Any] | None = None) -> str:
    expire_days = expires_delta_days if expires_delta_days is not None else settings.access_token_expire_days
    expire = datetime.now(timezone.utc) + timedelta(days=expire_days)
    payload: dict[str, Any] = {"sub": subject, "exp": expire}
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])


def get_token_subject(token: str) -> UUID:
    payload = decode_access_token(token)
    subject = payload.get("sub")
    if not subject:
        raise JWTError("Token subject missing")
    return UUID(str(subject))
