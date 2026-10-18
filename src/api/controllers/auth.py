"""
api/controllers/auth.py
------------------------
POST /auth/token  — OAuth2 password flow: username + password → JWT (RN07).
GET  /auth/me     — the logged-in account (username, role).

Dependencies for other routers:
    get_current_user     — any active account; 401 otherwise
    require_role(*roles) — 403 unless the account has one of the roles

Roles are read from the database on every request, so deactivating a user
or changing a role takes effect immediately, even for issued tokens.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.db_models import User
from src.api.models.schemas import TokenOut, UserOut
from src.api.repositories.user_repository import UserRepository
from src.core.database import get_db
from src.core.security import (
    InvalidToken, burn_password_check, create_access_token, decode_access_token, verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired token.",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """FastAPI dependency — validates the JWT and returns the active user."""
    try:
        payload = decode_access_token(token)
    except InvalidToken:
        raise _UNAUTHORIZED
    user = await UserRepository(db).get_by_username(payload["sub"])
    if user is None or not user.is_active:
        raise _UNAUTHORIZED
    return user


def require_role(*roles: str):
    """
    Dependency factory, e.g.  Depends(require_role("admin")).
    Admin and user currently have the same permissions, so no route uses it yet.
    """
    async def _check(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your account does not have permission for this action.",
            )
        return user
    return _check


@router.post("/token", response_model=TokenOut)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> TokenOut:
    """Authenticate with username + password (form fields) and return a bearer token."""
    user = await UserRepository(db).get_by_username(form_data.username)
    # bcrypt is deliberately slow (~0.2 s); run it off the event loop.
    if user is None or not user.is_active:
        await run_in_threadpool(burn_password_check, form_data.password)
        ok = False
    else:
        ok = await run_in_threadpool(verify_password, form_data.password, user.password_hash)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return TokenOut(access_token=create_access_token(user.username, user.role))


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)) -> UserOut:
    return UserOut(username=user.username, role=user.role)