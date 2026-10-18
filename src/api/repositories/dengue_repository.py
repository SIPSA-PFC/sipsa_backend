"""
api/repositories/dengue_repository.py
Dengue case queries — never mixed with ovitrap data.

Rows are returned joined with their epidemiological week, so callers get
epi_year / epi_week / week_start without a second query.
"""
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.db_models import DengueCase, EpiWeek

DISEASE = "dengue"


class DengueRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_series(
        self,
        neighborhood_id: int,
        epi_year_from:   int | None = None,
        epi_year_to:     int | None = None,
    ) -> list:
        """Weekly rows: neighborhood_id, case_count, epi_year, epi_week, week_start."""
        stmt = (
            select(
                DengueCase.neighborhood_id,
                DengueCase.case_count,
                EpiWeek.epi_year,
                EpiWeek.epi_week,
                EpiWeek.start_date.label("week_start"),
            )
            .join(EpiWeek, DengueCase.epi_week_id == EpiWeek.id)
            .where(DengueCase.neighborhood_id == neighborhood_id)
            .where(DengueCase.disease == DISEASE)
            .order_by(EpiWeek.start_date)
        )
        if epi_year_from is not None:
            stmt = stmt.where(EpiWeek.epi_year >= epi_year_from)
        if epi_year_to is not None:
            stmt = stmt.where(EpiWeek.epi_year <= epi_year_to)
        result = await self.db.execute(stmt)
        return result.all()

    async def get_city_wide_mean(self) -> list[dict]:
        """
        City-wide weekly mean across neighborhoods — replicates MATLAB
        nanmean(DengueOcc, 2). AVG ignores NULLs, like nanmean.
        Used by AggregationService to build the LSTM input array.
        """
        stmt = (
            select(
                EpiWeek.epi_year,
                EpiWeek.epi_week,
                EpiWeek.start_date,
                func.avg(DengueCase.case_count).label("mean_cases"),
            )
            .join(EpiWeek, DengueCase.epi_week_id == EpiWeek.id)
            .where(DengueCase.disease == DISEASE)
            .where(DengueCase.case_count.isnot(None))
            .group_by(EpiWeek.id, EpiWeek.epi_year, EpiWeek.epi_week, EpiWeek.start_date)
            .order_by(EpiWeek.start_date)
        )
        result = await self.db.execute(stmt)
        return [
            {
                "epi_year":   r.epi_year,
                "epi_week":   r.epi_week,
                "week_start": r.start_date,
                "mean_cases": float(r.mean_cases),
            }
            for r in result.all()
        ]

    async def count_available_weeks(self, neighborhood_id: int) -> int:
        stmt = (
            select(func.count())
            .select_from(DengueCase)
            .where(DengueCase.neighborhood_id == neighborhood_id)
            .where(DengueCase.disease == DISEASE)
            .where(DengueCase.case_count.isnot(None))
        )
        result = await self.db.execute(stmt)
        return result.scalar_one()

    async def get_latest_week(self) -> dict | None:
        stmt = (
            select(EpiWeek.epi_year, EpiWeek.epi_week, EpiWeek.start_date)
            .join(DengueCase, DengueCase.epi_week_id == EpiWeek.id)
            .where(DengueCase.disease == DISEASE)
            .order_by(EpiWeek.start_date.desc())
            .limit(1)
        )
        row = (await self.db.execute(stmt)).one_or_none()
        if row is None:
            return None
        return {"epi_year": row.epi_year, "epi_week": row.epi_week, "week_start": row.start_date}