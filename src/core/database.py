"""
core/database.py
----------------
SQLAlchemy async engine, session factory, and declarative base.

Supports both SQLite (local testing) and PostgreSQL (production).
The driver is selected automatically from the DATABASE_URL in .env:
    sqlite+aiosqlite:///./data/sipsa_test.db   → SQLite, no server needed
    postgresql+asyncpg://...                   → PostgreSQL, for production
"""

from sqlalchemy import MetaData, event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from src.core.config import settings

# SQLite needs check_same_thread=False for async use.
# PostgreSQL ignores connect_args entirely, so this is safe for both.
_connect_args = (
    {"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {}
)

# pool_pre_ping not supported by aiosqlite — disable for SQLite
_pool_pre_ping = not settings.database_url.startswith("sqlite")

engine = create_async_engine(
    settings.database_url,
    echo=False,                     # set True to log SQL during development
    connect_args=_connect_args,
    pool_pre_ping=_pool_pre_ping,
)

if settings.database_url.startswith("sqlite"):
    # SQLite ignores foreign keys unless enabled on every connection.
    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# Deterministic constraint names, so Alembic can find and alter them later
# (SQLite batch migrations in particular need named constraints).
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields one async DB session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise