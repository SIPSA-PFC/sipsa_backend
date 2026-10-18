"""
api/repositories/neighborhood_repository.py
Neighborhood lookups and prediction cache.
"""
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.models.db_models import Neighborhood, ModelPrediction


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


class PredictionRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_cached(
        self, neighborhood_id: int, model_version: str, input_type: str, lag: int
    ) -> list[ModelPrediction]:
        result = await self.db.execute(
            select(ModelPrediction)
            .where(ModelPrediction.neighborhood_id == neighborhood_id)
            .where(ModelPrediction.model_version   == model_version)
            .where(ModelPrediction.input_type      == input_type)
            .where(ModelPrediction.lag             == lag)
            .order_by(ModelPrediction.epi_year, ModelPrediction.epi_week)
        )
        return result.scalars().all()

    async def save(self, rows: list[ModelPrediction]) -> None:
        if not rows:
            return
        await self.db.execute(
            delete(ModelPrediction)
            .where(ModelPrediction.neighborhood_id == rows[0].neighborhood_id)
            .where(ModelPrediction.model_version   == rows[0].model_version)
            .where(ModelPrediction.input_type      == rows[0].input_type)
            .where(ModelPrediction.lag             == rows[0].lag)
        )
        self.db.add_all(rows)
        await self.db.flush()