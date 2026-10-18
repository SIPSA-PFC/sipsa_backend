"""
api/services/area_service.py
Weekly history for the city, a health district or a neighborhood.
"""
from __future__ import annotations

from datetime import date

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.schemas import (
    AreaCasesOut, AreaOvitrapOut, AreaWeekCasesOut, AreaWeekOvitrapOut, DistrictListOut, NeighborhoodListOut,
)
from src.api.repositories.area_repository import Area, AreaRepository


class AreaService:
    def __init__(self, db: AsyncSession):
        self.repo = AreaRepository(db)

    async def _area(self, scope: str, area_id: int | None) -> Area:
        area = await self.repo.area(scope, area_id)
        if area is None:
            raise HTTPException(status_code=404, detail=f"No {scope} with id {area_id}.")
        return area

    async def neighborhoods(self) -> list[NeighborhoodListOut]:
        return [
            NeighborhoodListOut(
                id=r.Neighborhood.id, name=r.Neighborhood.name, code=r.Neighborhood.code,
                district_id=r.Neighborhood.district_id, district_name=r.district_name,
                latitude=r.Neighborhood.latitude, longitude=r.Neighborhood.longitude, population=r.population,
            )
            for r in await self.repo.neighborhoods()
        ]

    async def districts(self) -> list[DistrictListOut]:
        rows = await self.repo.neighborhoods()
        out = []
        for d in await self.repo.districts():
            members = [r for r in rows if r.Neighborhood.district_id == d.id]
            pops = [r.population for r in members]
            out.append(DistrictListOut(
                id=d.id, name=d.name, neighborhood_ids=[r.Neighborhood.id for r in members],
                population=None if not pops or any(p is None for p in pops) else int(sum(pops)),
            ))
        return out

    async def cases(self, scope: str, area_id: int | None, year_from: int | None = None,
                    year_to: int | None = None, until: date | None = None) -> AreaCasesOut:
        area = await self._area(scope, area_id)
        series = [
            AreaWeekCasesOut(
                week_start=r.start_date, epi_year=r.epi_year, epi_week=r.epi_week,
                cases=int(r.cases or 0), neighborhoods_reporting=r.reporting,
                mean_per_neighborhood=round((r.cases or 0) / r.reporting, 3) if r.reporting else 0.0,
                incidence_per_100k=(round((r.cases or 0) / area.population * 100_000, 2)
                                    if area.population else None),
            )
            for r in await self.repo.weekly_cases(area, year_from, year_to, until)
        ]
        return AreaCasesOut(scope=scope, area_id=area.id, area_name=area.name,
                            population=area.population, series=series)

    async def ovitrap(self, scope: str, area_id: int | None, year_from: int | None = None,
                      year_to: int | None = None, until: date | None = None) -> AreaOvitrapOut:
        area = await self._area(scope, area_id)
        series = [
            AreaWeekOvitrapOut(
                week_start=r.start_date, epi_year=r.epi_year, epi_week=r.epi_week,
                opi=None if r.opi is None else round(float(r.opi), 4),
                edi=None if r.edi is None else round(float(r.edi), 3),
                neighborhoods_reporting=r.reporting,
            )
            for r in await self.repo.weekly_ovitrap(area, year_from, year_to, until)
        ]
        return AreaOvitrapOut(scope=scope, area_id=area.id, area_name=area.name, series=series)
