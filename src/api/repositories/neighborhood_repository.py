"""
api/repositories/neighborhood_repository.py
Neighborhood lookups.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.models.db_models import Neighborhood


class NeighborhoodRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all(self) -> list[Neighborhood]:
        result = await self.db.execute(select(Neighborhood).order_by(Neighborhood.name))
        return result.scalars().all()

    async def get_by_id(self, neighborhood_id: int) -> Neighborhood | None:
        result = await self.db.execute(
            select(Neighborhood).where(Neighborhood.id == neighborhood_id)
        )
        return result.scalar_one_or_none()
