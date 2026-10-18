"""
api/repositories/forecast_repository.py
Stored forecasts: one row per model run, area, as-of week and target week.
"""
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.db_models import EpiWeek, Forecast


class ForecastRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    def _area(self, stmt, scope: str, district_id: int | None, neighborhood_id: int | None):
        stmt = stmt.where(Forecast.scope == scope)
        stmt = stmt.where(Forecast.district_id.is_(None) if district_id is None else Forecast.district_id == district_id)
        return stmt.where(Forecast.neighborhood_id.is_(None) if neighborhood_id is None
                          else Forecast.neighborhood_id == neighborhood_id)

    async def get(self, model_run_id: int, as_of_week_id: int, scope: str = "city",
                  district_id: int | None = None, neighborhood_id: int | None = None) -> list:
        """(Forecast, target EpiWeek) pairs ordered by horizon."""
        stmt = (
            select(Forecast, EpiWeek)
            .join(EpiWeek, Forecast.target_week_id == EpiWeek.id)
            .where(Forecast.model_run_id == model_run_id, Forecast.as_of_week_id == as_of_week_id)
            .order_by(Forecast.horizon)
        )
        result = await self.db.execute(self._area(stmt, scope, district_id, neighborhood_id))
        return result.all()

    async def replace(self, model_run_id: int, as_of_week_id: int, rows: list[Forecast], scope: str = "city",
                      district_id: int | None = None, neighborhood_id: int | None = None) -> None:
        stmt = delete(Forecast).where(Forecast.model_run_id == model_run_id, Forecast.as_of_week_id == as_of_week_id)
        await self.db.execute(self._area(stmt, scope, district_id, neighborhood_id))
        self.db.add_all(rows)
        await self.db.flush()
