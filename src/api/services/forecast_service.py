"""
api/services/forecast_service.py
Forecasts from saved models: look up the registered run, reuse stored
forecasts when present, otherwise run the models (off the event loop) and
store the result.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.db_models import DengueCase, EpiWeek, Forecast
from src.api.models.schemas import ForecastOut, ForecastPointOut, WeekOut
from src.api.repositories.forecast_repository import ForecastRepository
from src.api.repositories.week_repository import WeekRepository
from src.core import epiweek
from src.ml import data as ml_data
from src.ml.inference import ModelNotReadyError, ModelRegistry, RunInfo, forecast, registry
from src.ml.preprocessing import MissingDataError


def _utc(value: datetime | None) -> datetime:
    """SQLite returns timestamps without a timezone; they are stored in UTC."""
    if value is None:
        return datetime.now(timezone.utc)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def risk_level(mean_per_neighborhood: float) -> str:
    """Interim rule on the model's unit (mean weekly cases per neighborhood)."""
    if mean_per_neighborhood > 20:
        return "high"
    if mean_per_neighborhood > 10:
        return "moderate"
    return "low"


class ForecastService:
    def __init__(self, db: AsyncSession, models: ModelRegistry = registry):
        self.db = db
        self.models = models
        self.weeks = WeekRepository(db)
        self.forecasts = ForecastRepository(db)

    def _run(self, input_type: str, lag: int) -> RunInfo:
        try:
            return self.models.get(input_type, lag)
        except ModelNotReadyError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    async def _neighborhoods_reporting(self, week_id: int) -> int:
        """How many neighborhoods the city mean averages over (to turn it into a city total)."""
        stmt = (select(func.count()).select_from(DengueCase)
                .where(DengueCase.epi_week_id == week_id, DengueCase.disease == "dengue",
                       DengueCase.case_count.isnot(None)))
        return (await self.db.execute(stmt)).scalar_one()

    async def city(self, input_type: str, lag: int, n_weeks: int | None = None) -> ForecastOut:
        run = self._run(input_type, lag)
        n_weeks = n_weeks or run.max_horizon
        if n_weeks > run.max_horizon:
            raise HTTPException(
                status_code=422,
                detail=(f"With lag {lag} the newest input is {run.max_horizon} week(s) old, so the model can "
                        f"forecast at most {run.max_horizon} week(s) ahead from observed data. "
                        "Use lag 6 for 4 weeks ahead."),
            )

        series = await ml_data.load_from_db(self.db)
        as_of_index = len(series.weeks) - 1
        as_of = await self.weeks.get(*series.weeks[as_of_index])

        n_neigh = await self._neighborhoods_reporting(as_of.id)
        stored = await self.forecasts.get(run.model_run_id, as_of.id)
        if len(stored) < run.max_horizon:
            stored = await self._compute_and_store(run, series, as_of_index, as_of, n_neigh)
        return self._response(run, as_of, stored[:n_weeks], area_name="Natal", n_neighborhoods=n_neigh)

    async def _compute_and_store(self, run: RunInfo, series: ml_data.CitySeries, as_of_index: int,
                                 as_of: EpiWeek, n_neigh: int) -> list:
        """Run the models; city totals = mean per neighborhood x neighborhoods reporting."""
        def compute():
            return forecast(series, as_of_index, run, run.max_horizon, self.models.models(run))
        try:
            result = await run_in_threadpool(compute)
        except MissingDataError as exc:
            raise HTTPException(status_code=422,
                                detail=f"Cannot forecast: {exc}.") from exc

        targets = [epiweek.add_weeks(as_of.epi_year, as_of.epi_week, h) for h in range(1, run.max_horizon + 1)]
        target_rows = await self.weeks.ensure(targets)
        now = datetime.now(timezone.utc)
        rows = [
            Forecast(model_run_id=run.model_run_id, scope="city", as_of_week_id=as_of.id,
                     target_week_id=target_rows[t].id, horizon=h,
                     predicted=float(result["mean"][h - 1]) * n_neigh,
                     lower=float(result["lower"][h - 1]) * n_neigh,
                     upper=float(result["upper"][h - 1]) * n_neigh,
                     method="model", generated_at=now)
            for h, t in enumerate(targets, start=1)
        ]
        await self.forecasts.replace(run.model_run_id, as_of.id, rows)
        await self.db.commit()
        return [(r, target_rows[t]) for r, t in zip(rows, targets)]

    def _response(self, run: RunInfo, as_of: EpiWeek, stored: list, area_name: str, *,
                  scope: str = "city", area_id: int | None = None,
                  n_neighborhoods: int | None = None) -> ForecastOut:
        points = [
            ForecastPointOut(
                horizon=f.horizon, week_start=week.start_date, epi_year=week.epi_year, epi_week=week.epi_week,
                predicted_cases=round(f.predicted, 2),
                lower_bound=None if f.lower is None else round(f.lower, 2),
                upper_bound=None if f.upper is None else round(f.upper, 2),
                mean_per_neighborhood=round(f.predicted / n_neighborhoods, 3) if n_neighborhoods else None,
            )
            for f, week in stored
        ]
        mean_unit = [p.mean_per_neighborhood for p in points if p.mean_per_neighborhood is not None]
        return ForecastOut(
            scope=scope, area_id=area_id, area_name=area_name, input_type=run.input_type, lag=run.lag,
            max_horizon=run.max_horizon, model_version=run.model_version,
            method=stored[0][0].method,
            as_of=WeekOut(epi_year=as_of.epi_year, epi_week=as_of.epi_week, week_start=as_of.start_date),
            generated_at=_utc(stored[0][0].generated_at),
            risk_level=risk_level(sum(mean_unit) / len(mean_unit)) if mean_unit else "low",
            series=points,
        )
