"""
api/repositories/user_repository.py
User account lookups and changes.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.db_models import User


class UserRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_username(self, username: str) -> User | None:
        result = await self.db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    async def add(self, user: User) -> User:
        self.db.add(user)
        await self.db.flush()
        return user