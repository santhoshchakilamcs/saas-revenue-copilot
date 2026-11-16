"""OAuth2 password flow with JWT bearer tokens, and role-based access (viewer, admin)."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

ROLES = ("viewer", "admin")
ROLE_RANK = {"viewer": 0, "admin": 1}
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


@dataclass
class User:
    username: str
    role: str
    password_hash: bytes


class AuthConfig:
    def __init__(self, secret: str | None = None, expire_minutes: int | None = None):
        self.secret = secret or os.environ.get("JWT_SECRET") or secrets.token_urlsafe(32)
        self.expire_minutes = expire_minutes or int(os.environ.get("JWT_EXPIRE_MINUTES", 30))
        self.algorithm = "HS256"


class UserStore:
    """In-memory user store, bcrypt-hashed. Users come from env vars (see .env.example)."""

    def __init__(self):
        self._users: dict[str, User] = {}

    def add(self, username: str, password: str, role: str) -> None:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        self._users[username] = User(username, role, bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=10)))

    def authenticate(self, username: str, password: str) -> User | None:
        u = self._users.get(username)
        if u is None:
            bcrypt.checkpw(b"timing-equalizer", bcrypt.hashpw(b"x", bcrypt.gensalt(rounds=4)))
            return None
        return u if bcrypt.checkpw(password.encode(), u.password_hash) else None

    def get(self, username: str) -> User | None:
        return self._users.get(username)

    @classmethod
    def from_env(cls) -> "UserStore":
        s = cls()
        for role in ROLES:
            name = os.environ.get(f"{role.upper()}_USERNAME")
            pw = os.environ.get(f"{role.upper()}_PASSWORD")
            if name and pw:
                s.add(name, pw, role)
        return s


def create_token(cfg: AuthConfig, user: User) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": user.username, "role": user.role, "iat": now, "exp": now + timedelta(minutes=cfg.expire_minutes)}
    return jwt.encode(payload, cfg.secret, algorithm=cfg.algorithm)


def build_dependencies(cfg: AuthConfig, users: UserStore) -> tuple[Callable, Callable]:
    def current_user(token: str = Depends(oauth2_scheme)) -> User:
        unauthorized = HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired token",
                                     headers={"WWW-Authenticate": "Bearer"})
        try:
            payload = jwt.decode(token, cfg.secret, algorithms=[cfg.algorithm])
        except jwt.PyJWTError:
            raise unauthorized from None
        user = users.get(payload.get("sub", ""))
        if user is None or user.role != payload.get("role"):
            raise unauthorized
        return user

    def require(role: str) -> Callable:
        def dep(user: User = Depends(current_user)) -> User:
            if ROLE_RANK[user.role] < ROLE_RANK[role]:
                raise HTTPException(status.HTTP_403_FORBIDDEN, f"requires role '{role}'")
            return user

        return dep

    return current_user, require
