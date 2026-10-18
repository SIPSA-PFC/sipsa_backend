"""
api/services/forecast_service.py
--------------------------------
Forecasts for the city, a health district or a neighborhood.

City: the LSTM's own output (method 'model'), converted from mean cases per
neighborhood to a city total.

District / neighborhood: the paper's model is city-wide only, so these are
the city forecast multiplied by the area's share of the city's cases in all
weeks observed up to the as-of week (method 'split'). It is an estimate that
assumes each area keeps its usual share; responses say so through `method`.

As-of week: by default the latest complete week (every neighborhood
reported and the model's input exists); newly ingested partial weeks are
skipped until complete.

Replay: every forecast can be made "as of" an earlier week. Only data
observed up to that week is used (model inputs, shares, risk history), and
each point carries the cases actually observed, when known. If a target
week falls in the model's training period, `in_sample` is true: the model
saw that outcome, so the comparison is not a fair test.

Risk: compares the highest forecast week with the area's own weekly history
up to the as-of week (settings.risk_percentiles), so the same rule works for
a small neighborhood and the whole city. Below settings.risk_min_cases the
risk is always low.

Forecasts are stored in the forecast table and reused.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from fastapi import HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.db_models import DengueCase, EpiWeek, Forecast, ModelRun
from src.api.models.schemas import ForecastOut, ForecastPointOut, RiskThresholdsOut, WeekOut
from src.api.repositories.area_repository import Area, AreaRepository
from src.api.repositories.forecast_repository import ForecastRepository
from src.api.repositories.week_repository import WeekRepository
from src.core import epiweek
from src.core.config import settings
from src.ml import data as ml_data
from src.ml.inference import ModelNotReadyError, ModelRegistry, RunInfo, forecast, registry
from src.ml.preprocessing import MissingDataError


def _utc(value: datetime | None) -> datetime:
    """SQLite returns timestamps without a timezone; they are stored in UTC."""
    if value is None:
        return datetime.now(timezone.utc)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def relative_risk(history: np.ndarray, peak: float) -> tuple[str, RiskThresholdsOut]:
    lo_p, hi_p = settings.risk_percentiles
    history = np.asarray(history, dtype=float)
    moderate, high = (np.percentile(history, [lo_p, hi_p]) if len(history) else (0.0, 0.0))
    thresholds = RiskThresholdsOut(moderate_from=round(float(moderate), 2), high_from=round(float(high), 2),
                                   percentiles=[lo_p, hi_p], history_weeks=len(history))
    if peak < settings.risk_min_cases:
        return "low", thresholds
    if peak > high:
        return "high", thresholds
    if peak > moderate:
        return "moderate", thresholds
    return "low", thresholds


class ForecastService:
    def __init__(self, db: AsyncSession, models: ModelRegistry = registry):
        self.db = db
        self.models = models
        self.weeks = WeekRepository(db)
        self.forecasts = ForecastRepository(db)
        self.areas = AreaRepository(db)

    # ── Public ────────────────────────────────────────────────────────────────

    async def for_area(self, scope: str, area_id: int | None, input_type: str, lag: int,
                       n_weeks: int | None = None, as_of: tuple[int, int] | None = None) -> ForecastOut:
        area = await self.areas.area(scope, area_id)
        if area is None:
            raise HTTPException(status_code=404, detail=f"No {scope} with id {area_id}.")
        run = self._run(input_type, lag)
        n_weeks = self._check_horizon(run, lag, n_weeks)

        as_of_row, n_neigh, city_rows = await self._city_rows(run, as_of)
        if scope == "city":
            rows = city_rows
        else:
            rows = await self._split_rows(run, area, as_of_row, city_rows)

        history = [r.cases for r in await self.areas.weekly_cases(area, until=as_of_row.start_date)]
        peak = max(f.predicted for f, _ in rows[:n_weeks])
        level, thresholds = relative_risk(history, peak)
        actual = await self.actual_cases(area, [w for _, w in rows[:n_weeks]])
        return self._response(run, area, as_of_row, rows[:n_weeks], level, thresholds,
                              n_neighborhoods=n_neigh if scope == "city" else None,
                              actual=actual, in_sample=await self.in_sample(run, rows[:n_weeks]))

    async def actual_cases(self, area: Area, weeks: list[EpiWeek]) -> dict:
        """Observed cases in the area for the given weeks (only weeks that have data)."""
        if not weeks:
            return {}
        last = max(w.start_date for w in weeks)
        wanted = {w.start_date for w in weeks}
        return {r.start_date: int(r.cases) for r in await self.areas.weekly_cases(area, until=last)
                if r.start_date in wanted and r.cases is not None}

    async def in_sample(self, run: RunInfo, rows: list) -> bool:
        """True if any target week is inside the model's training period."""
        mr = await self.db.get(ModelRun, run.model_run_id)
        train_end = (await self.db.get(EpiWeek, mr.train_to_week_id)).start_date
        return any(w.start_date <= train_end for _, w in rows)

    async def city_and_neighborhoods(self, input_type: str, lag: int, as_of: tuple[int, int] | None = None):
        """City rows plus split rows for every neighborhood, for the dashboard."""
        run = self._run(input_type, lag)
        as_of_row, n_neigh, city_rows = await self._city_rows(run, as_of)
        shares = await self._shares(as_of_row)
        out = {}
        for n in await self.areas.neighborhoods():
            area = Area("neighborhood", n.Neighborhood.id, n.Neighborhood.name, (n.Neighborhood.id,), n.population)
            out[area.id] = await self._split_rows(run, area, as_of_row, city_rows, shares)
        return run, as_of_row, n_neigh, city_rows, out

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _run(self, input_type: str, lag: int) -> RunInfo:
        try:
            return self.models.get(input_type, lag)
        except ModelNotReadyError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    @staticmethod
    def _check_horizon(run: RunInfo, lag: int, n_weeks: int | None) -> int:
        n_weeks = n_weeks or run.max_horizon
        if n_weeks > run.max_horizon:
            raise HTTPException(
                status_code=422,
                detail=(f"With lag {lag} the newest input is {run.max_horizon} week(s) old, so the model can "
                        f"forecast at most {run.max_horizon} week(s) ahead from observed data. "
                        "Use lag 6 for 4 weeks ahead."),
            )
        return n_weeks

    async def _neighborhoods_reporting(self, week_id: int) -> int:
        stmt = (select(func.count()).select_from(DengueCase)
                .where(DengueCase.epi_week_id == week_id, DengueCase.disease == "dengue",
                       DengueCase.case_count.isnot(None)))
        return (await self.db.execute(stmt)).scalar_one()

    async def _complete_weeks(self, series: ml_data.CitySeries, input_type: str) -> tuple[list[bool], list[str]]:
        """
        Which weeks can serve as an as-of week: every neighborhood reported
        cases and the model's input series has a value. Newly ingested weeks
        are often partial; a city mean over a few neighborhoods would mislead.
        """
        city = await self.areas.area("city")
        reporting = {(r.epi_year, r.epi_week): r.reporting for r in await self.areas.weekly_cases(city)}
        n_total = len(city.neighborhood_ids)
        x = series.series(input_type)
        complete, why = [], []
        for i, w in enumerate(series.weeks):
            problems = []
            if reporting.get(w, 0) < n_total:
                problems.append(f"{reporting.get(w, 0)} of {n_total} neighborhoods reported cases")
            if np.isnan(x[i]):
                problems.append(f"no {input_type.upper()} value")
            complete.append(not problems)
            why.append("; ".join(problems))
        return complete, why

    async def _city_rows(self, run: RunInfo, as_of: tuple[int, int] | None):
        """City forecast at the as-of week (default: latest complete week), computed and stored if needed."""
        series = await ml_data.load_from_db(self.db)
        complete, why = await self._complete_weeks(series, run.input_type)
        if as_of is None:
            if not any(complete):
                raise HTTPException(status_code=422, detail="No complete week to forecast from.")
            as_of_index = max(i for i, ok in enumerate(complete) if ok)
        elif as_of in series.weeks:
            as_of_index = series.weeks.index(as_of)
            if not complete[as_of_index]:
                raise HTTPException(status_code=422, detail=(
                    f"Epidemiological week {as_of[0]}-{as_of[1]:02d} is incomplete ({why[as_of_index]}); "
                    "pick an earlier week or omit as_of to use the latest complete one."))
        else:
            raise HTTPException(status_code=422, detail=f"No data for epidemiological week {as_of[0]}-{as_of[1]:02d}.")
        as_of_row = await self.weeks.get(*series.weeks[as_of_index])
        n_neigh = await self._neighborhoods_reporting(as_of_row.id)

        stored = await self.forecasts.get(run.model_run_id, as_of_row.id)
        if len(stored) >= run.max_horizon:
            return as_of_row, n_neigh, stored

        def compute():
            return forecast(series, as_of_index, run, run.max_horizon, self.models.models(run))
        try:
            result = await run_in_threadpool(compute)
        except MissingDataError as exc:
            raise HTTPException(status_code=422, detail=f"Cannot forecast: {exc}.") from exc

        targets = [epiweek.add_weeks(as_of_row.epi_year, as_of_row.epi_week, h) for h in range(1, run.max_horizon + 1)]
        target_rows = await self.weeks.ensure(targets)
        now = datetime.now(timezone.utc)
        rows = [
            Forecast(model_run_id=run.model_run_id, scope="city", as_of_week_id=as_of_row.id,
                     target_week_id=target_rows[t].id, horizon=h,
                     predicted=float(result["mean"][h - 1]) * n_neigh,
                     lower=float(result["lower"][h - 1]) * n_neigh,
                     upper=float(result["upper"][h - 1]) * n_neigh,
                     method="model", generated_at=now)
            for h, t in enumerate(targets, start=1)
        ]
        await self.forecasts.replace(run.model_run_id, as_of_row.id, rows)
        await self.db.commit()
        return as_of_row, n_neigh, [(r, target_rows[t]) for r, t in zip(rows, targets)]

    async def _shares(self, as_of_row: EpiWeek) -> dict[int, float]:
        """Each neighborhood's share of city cases in all weeks observed up to the as-of week."""
        first = (await self.db.execute(select(func.min(EpiWeek.start_date)))).scalar_one()
        totals = await self.areas.cases_by_neighborhood(first, as_of_row.start_date)
        city = sum(totals.values())
        return {nid: (v / city if city else 0.0) for nid, v in totals.items()}

    async def _split_rows(self, run: RunInfo, area: Area, as_of_row: EpiWeek, city_rows: list,
                          shares: dict[int, float] | None = None) -> list:
        district_id = area.id if area.scope == "district" else None
        neighborhood_id = area.id if area.scope == "neighborhood" else None
        stored = await self.forecasts.get(run.model_run_id, as_of_row.id, area.scope, district_id, neighborhood_id)
        if len(stored) >= len(city_rows):
            return stored
        shares = shares if shares is not None else await self._shares(as_of_row)
        share = sum(shares.get(n, 0.0) for n in area.neighborhood_ids)
        now = datetime.now(timezone.utc)
        rows = [
            (Forecast(model_run_id=run.model_run_id, scope=area.scope, district_id=district_id,
                      neighborhood_id=neighborhood_id, as_of_week_id=as_of_row.id, target_week_id=week.id,
                      horizon=f.horizon, predicted=f.predicted * share,
                      lower=None if f.lower is None else f.lower * share,
                      upper=None if f.upper is None else f.upper * share,
                      method="split", generated_at=now), week)
            for f, week in city_rows
        ]
        await self.forecasts.replace(run.model_run_id, as_of_row.id, [r for r, _ in rows],
                                     area.scope, district_id, neighborhood_id)
        await self.db.commit()
        return rows

    def _response(self, run: RunInfo, area: Area, as_of: EpiWeek, rows: list, level: str,
                  thresholds: RiskThresholdsOut, n_neighborhoods: int | None,
                  actual: dict | None = None, in_sample: bool = False) -> ForecastOut:
        points = [
            ForecastPointOut(
                horizon=f.horizon, week_start=week.start_date, epi_year=week.epi_year, epi_week=week.epi_week,
                predicted_cases=round(f.predicted, 2),
                lower_bound=None if f.lower is None else round(f.lower, 2),
                upper_bound=None if f.upper is None else round(f.upper, 2),
                mean_per_neighborhood=round(f.predicted / n_neighborhoods, 3) if n_neighborhoods else None,
                actual_cases=(actual or {}).get(week.start_date),
            )
            for f, week in rows
        ]
        return ForecastOut(
            scope=area.scope, area_id=area.id, area_name=area.name, input_type=run.input_type, lag=run.lag,
            max_horizon=run.max_horizon, model_version=run.model_version, method=rows[0][0].method,
            as_of=WeekOut(epi_year=as_of.epi_year, epi_week=as_of.epi_week, week_start=as_of.start_date),
            generated_at=_utc(rows[0][0].generated_at), in_sample=in_sample,
            risk_level=level, risk_thresholds=thresholds,
            series=points,
        )
