"""
api/repositories/ovitrap_repository.py
OPI and EDI queries — never mixed with dengue case data.
"""
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.db_models import EpiWeek, OvitrapReading


class OvitrapRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_series(
        self,
        neighborhood_id: int,
        epi_year_from:   int | None = None,
        epi_year_to:     int | None = None,
    ) -> list:
        """Weekly rows: neighborhood_id, opi, edi, epi_year, epi_week, week_start."""
        stmt = (
            select(
                OvitrapReading.neighborhood_id,
                OvitrapReading.opi,
                OvitrapReading.edi,
                EpiWeek.epi_year,
                EpiWeek.epi_week,
                EpiWeek.start_date.label("week_start"),
            )
            .join(EpiWeek, OvitrapReading.epi_week_id == EpiWeek.id)
            .where(OvitrapReading.neighborhood_id == neighborhood_id)
            .order_by(EpiWeek.start_date)
        )
        if epi_year_from is not None:
            stmt = stmt.where(EpiWeek.epi_year >= epi_year_from)
        if epi_year_to is not None:
            stmt = stmt.where(EpiWeek.epi_year <= epi_year_to)
        result = await self.db.execute(stmt)
        return result.all()

    async def get_city_wide_edi_mean(self) -> list[dict]:
        """City-wide weekly mean EDI — replicates MATLAB nanmean(ODI_Mat, 2)."""
        stmt = (
            select(
                EpiWeek.epi_year,
                EpiWeek.epi_week,
                EpiWeek.start_date,
                func.avg(OvitrapReading.edi).label("mean_edi"),
            )
            .join(EpiWeek, OvitrapReading.epi_week_id == EpiWeek.id)
            .where(OvitrapReading.edi.isnot(None))
            .group_by(EpiWeek.id, EpiWeek.epi_year, EpiWeek.epi_week, EpiWeek.start_date)
            .order_by(EpiWeek.start_date)
        )
        result = await self.db.execute(stmt)
        return [
            {
                "epi_year":   r.epi_year,
                "epi_week":   r.epi_week,
                "week_start": r.start_date,
                "mean_edi":   float(r.mean_edi),
            }
            for r in result.all()
        ]