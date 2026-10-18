"""
Routes for the city, health districts and neighborhoods.

    GET /bairros                       neighborhoods (code, district, coordinates, population)
    GET /bairros/{id}/casos            weekly cases (+ incidence per 100,000)
    GET /bairros/{id}/ovitrampas       weekly OPI / EDI
    GET /bairros/{id}/previsao         forecast (city forecast x neighborhood share)
    GET /distritos …                   same for health districts
    GET /cidade/casos | /ovitrampas | /previsao   same for the whole city
"""
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.schemas import (
    AreaCasesOut, AreaOvitrapOut, DistrictListOut, ForecastOut, Lag, NeighborhoodListOut,
)
from src.api.services.area_service import AreaService
from src.api.services.forecast_service import ForecastService
from src.core.database import get_db


class YearRange:
    def __init__(
        self,
        year_from: Optional[int] = Query(default=None, description="First epidemiological year to include"),
        year_to:   Optional[int] = Query(default=None, description="Last epidemiological year to include"),
    ):
        self.year_from, self.year_to = year_from, year_to


class ForecastParams:
    def __init__(
        self,
        input_type: Literal["dengue", "edi"] = Query(
            default="edi", description="Predictor: past ovitrap egg density (edi) or past dengue cases"),
        lag: Lag = Query(default=Lag.SIX, description="Input window: 1, 3 (weeks 3–1), 4 (4–2), 5 (5–3), 6 (6–4)"),
        n_weeks: Optional[int] = Query(
            default=None, ge=1, le=4,
            description="Weeks ahead; at most the window's newest lag (lag 1/3: 1, 4: 2, 5: 3, 6: 4). Default: maximum."),
    ):
        self.input_type, self.lag, self.n_weeks = input_type, int(lag), n_weeks


# ── City ──────────────────────────────────────────────────────────────────────
cidade = APIRouter(prefix="/cidade", tags=["cidade"])


@cidade.get("/casos", response_model=AreaCasesOut)
async def city_cases(years: YearRange = Depends(), db: AsyncSession = Depends(get_db)):
    """Weekly dengue cases for the whole city."""
    return await AreaService(db).cases("city", None, years.year_from, years.year_to)


@cidade.get("/ovitrampas", response_model=AreaOvitrapOut)
async def city_ovitrap(years: YearRange = Depends(), db: AsyncSession = Depends(get_db)):
    """Weekly ovitrap indices, averaged over the city's neighborhoods."""
    return await AreaService(db).ovitrap("city", None, years.year_from, years.year_to)


@cidade.get("/previsao", response_model=ForecastOut)
async def city_forecast(p: ForecastParams = Depends(), db: AsyncSession = Depends(get_db)):
    """City-wide forecast: the LSTM's own output (method 'model')."""
    return await ForecastService(db).for_area("city", None, p.input_type, p.lag, p.n_weeks)


# ── Districts ─────────────────────────────────────────────────────────────────
distritos = APIRouter(prefix="/distritos", tags=["distritos"])


@distritos.get("", response_model=list[DistrictListOut])
async def list_districts(db: AsyncSession = Depends(get_db)):
    return await AreaService(db).districts()


@distritos.get("/{district_id}/casos", response_model=AreaCasesOut)
async def district_cases(district_id: int, years: YearRange = Depends(), db: AsyncSession = Depends(get_db)):
    return await AreaService(db).cases("district", district_id, years.year_from, years.year_to)


@distritos.get("/{district_id}/ovitrampas", response_model=AreaOvitrapOut)
async def district_ovitrap(district_id: int, years: YearRange = Depends(), db: AsyncSession = Depends(get_db)):
    return await AreaService(db).ovitrap("district", district_id, years.year_from, years.year_to)


@distritos.get("/{district_id}/previsao", response_model=ForecastOut)
async def district_forecast(district_id: int, p: ForecastParams = Depends(), db: AsyncSession = Depends(get_db)):
    """City forecast x the district's share of cases in the training period (method 'split')."""
    return await ForecastService(db).for_area("district", district_id, p.input_type, p.lag, p.n_weeks)


# ── Neighborhoods ─────────────────────────────────────────────────────────────
bairros = APIRouter(prefix="/bairros", tags=["bairros"])


@bairros.get("", response_model=list[NeighborhoodListOut])
async def list_neighborhoods(db: AsyncSession = Depends(get_db)):
    return await AreaService(db).neighborhoods()


@bairros.get("/{neighborhood_id}/casos", response_model=AreaCasesOut)
async def neighborhood_cases(neighborhood_id: int, years: YearRange = Depends(), db: AsyncSession = Depends(get_db)):
    return await AreaService(db).cases("neighborhood", neighborhood_id, years.year_from, years.year_to)


@bairros.get("/{neighborhood_id}/ovitrampas", response_model=AreaOvitrapOut)
async def neighborhood_ovitrap(neighborhood_id: int, years: YearRange = Depends(), db: AsyncSession = Depends(get_db)):
    return await AreaService(db).ovitrap("neighborhood", neighborhood_id, years.year_from, years.year_to)


@bairros.get("/{neighborhood_id}/previsao", response_model=ForecastOut)
async def neighborhood_forecast(neighborhood_id: int, p: ForecastParams = Depends(), db: AsyncSession = Depends(get_db)):
    """City forecast x the neighborhood's share of cases in the training period (method 'split')."""
    return await ForecastService(db).for_area("neighborhood", neighborhood_id, p.input_type, p.lag, p.n_weeks)
