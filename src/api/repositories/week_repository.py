"""
api/repositories/week_repository.py
Epidemiological-week rows, created on demand for future (forecast) weeks.
"""
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.db_models import EpiWeek
from src.core import epiweek


class WeekRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get(self, epi_year: int, epi_week: int) -> EpiWeek | None:
        result = await self.db.execute(
            select(EpiWeek).where(EpiWeek.epi_year == epi_year, EpiWeek.epi_week == epi_week)
        )
        return result.scalar_one_or_none()

    async def ensure(self, weeks: list[tuple[int, int]]) -> dict[tuple[int, int], EpiWeek]:
        """Rows for the given (epi_year, epi_week) pairs, creating any that are missing."""
        found = {}
        for y, w in weeks:
            row = await self.get(y, w)
            if row is None:
                start = epiweek.week_start(y, w)
                row = EpiWeek(epi_year=y, epi_week=w, start_date=start, end_date=start + timedelta(days=6))
                self.db.add(row)
            found[(y, w)] = row
        await self.db.flush()
        return found
