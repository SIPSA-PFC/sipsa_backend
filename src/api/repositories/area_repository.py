"""
api/repositories/area_repository.py
-----------------------------------
Weekly data for any area: the whole city, a health district, or one
neighborhood. An area is a set of neighborhoods; its weekly cases are the
sum over those neighborhoods, its ovitrap indices the mean.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.db_models import (
    DengueCase, District, EpiWeek, Neighborhood, OvitrapReading, Socioeconomic,
)

CENSUS_YEAR = 2010


@dataclass(frozen=True)
class Area:
    scope:            str                 # "city" | "district" | "neighborhood"
    id:               int | None
    name:             str
    neighborhood_ids: tuple[int, ...]
    population:       int | None          # 2010 Census; None if any neighborhood lacks it


class AreaRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ── Areas ─────────────────────────────────────────────────────────────────

    async def neighborhoods(self) -> list:
        """Every neighborhood with its district and 2010 population."""
        stmt = (
            select(Neighborhood, District.name.label("district_name"), Socioeconomic.population)
            .outerjoin(District, Neighborhood.district_id == District.id)
            .outerjoin(Socioeconomic, (Socioeconomic.neighborhood_id == Neighborhood.id)
                       & (Socioeconomic.reference_year == CENSUS_YEAR))
            .order_by(Neighborhood.name)
        )
        return (await self.db.execute(stmt)).all()

    async def districts(self) -> list[District]:
        return (await self.db.execute(select(District).order_by(District.name))).scalars().all()

    async def area(self, scope: str, area_id: int | None = None) -> Area | None:
        rows = await self.neighborhoods()
        if scope == "city":
            members, name = rows, "Natal"
        elif scope == "district":
            district = await self.db.get(District, area_id)
            if district is None:
                return None
            members, name = [r for r in rows if r.Neighborhood.district_id == area_id], district.name
        else:
            members = [r for r in rows if r.Neighborhood.id == area_id]
            if not members:
                return None
            name = members[0].Neighborhood.name
        pops = [r.population for r in members]
        return Area(scope, area_id, name, tuple(r.Neighborhood.id for r in members),
                    None if any(p is None for p in pops) or not pops else int(sum(pops)))

    # ── Weekly series ─────────────────────────────────────────────────────────

    def _week_filter(self, stmt, year_from: int | None, year_to: int | None, until: date | None):
        if year_from is not None:
            stmt = stmt.where(EpiWeek.epi_year >= year_from)
        if year_to is not None:
            stmt = stmt.where(EpiWeek.epi_year <= year_to)
        if until is not None:
            stmt = stmt.where(EpiWeek.start_date <= until)
        return stmt

    async def weekly_cases(self, area: Area, year_from: int | None = None, year_to: int | None = None,
                           until: date | None = None) -> list:
        """Per week: total cases in the area and how many neighborhoods reported."""
        stmt = (
            select(EpiWeek.epi_year, EpiWeek.epi_week, EpiWeek.start_date,
                   func.sum(DengueCase.case_count).label("cases"),
                   func.count(DengueCase.case_count).label("reporting"))
            .join(DengueCase, DengueCase.epi_week_id == EpiWeek.id)
            .where(DengueCase.disease == "dengue", DengueCase.neighborhood_id.in_(area.neighborhood_ids))
            .group_by(EpiWeek.id, EpiWeek.epi_year, EpiWeek.epi_week, EpiWeek.start_date)
            .order_by(EpiWeek.start_date)
        )
        return (await self.db.execute(self._week_filter(stmt, year_from, year_to, until))).all()

    async def weekly_ovitrap(self, area: Area, year_from: int | None = None, year_to: int | None = None,
                             until: date | None = None) -> list:
        """Per week: mean OPI and EDI over the area's neighborhoods with readings."""
        stmt = (
            select(EpiWeek.epi_year, EpiWeek.epi_week, EpiWeek.start_date,
                   func.avg(OvitrapReading.opi).label("opi"), func.avg(OvitrapReading.edi).label("edi"),
                   func.count(OvitrapReading.edi).label("reporting"))
            .join(OvitrapReading, OvitrapReading.epi_week_id == EpiWeek.id)
            .where(OvitrapReading.neighborhood_id.in_(area.neighborhood_ids))
            .group_by(EpiWeek.id, EpiWeek.epi_year, EpiWeek.epi_week, EpiWeek.start_date)
            .order_by(EpiWeek.start_date)
        )
        return (await self.db.execute(self._week_filter(stmt, year_from, year_to, until))).all()

    async def cases_by_neighborhood(self, start: date, end: date) -> dict[int, float]:
        """Total cases per neighborhood between two week-start dates (inclusive)."""
        stmt = (
            select(DengueCase.neighborhood_id, func.sum(DengueCase.case_count))
            .join(EpiWeek, DengueCase.epi_week_id == EpiWeek.id)
            .where(DengueCase.disease == "dengue", EpiWeek.start_date >= start, EpiWeek.start_date <= end)
            .group_by(DengueCase.neighborhood_id)
        )
        return {nid: float(total or 0) for nid, total in (await self.db.execute(stmt)).all()}

    async def weekly_cases_all(self, until: date) -> list:
        """(neighborhood_id, week start, cases) for every neighborhood up to a week."""
        stmt = (
            select(DengueCase.neighborhood_id, EpiWeek.start_date, DengueCase.case_count)
            .join(EpiWeek, DengueCase.epi_week_id == EpiWeek.id)
            .where(DengueCase.disease == "dengue", EpiWeek.start_date <= until, DengueCase.case_count.isnot(None))
        )
        return (await self.db.execute(stmt)).all()
