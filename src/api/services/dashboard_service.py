"""
api/services/dashboard_service.py
Everything the main panel needs: this week's cases, the city forecast, and
per-neighborhood forecasts with risk levels, alerts and map bubbles.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.schemas import AlertOut, DashboardOut, ForecastPointOut, MapBubbleOut, WeekOut
from src.api.repositories.area_repository import AreaRepository
from src.api.services.forecast_service import ForecastService, relative_risk

_ORDER = {"high": 0, "moderate": 1, "low": 2}


class DashboardService:
    def __init__(self, db: AsyncSession):
        self.forecasts = ForecastService(db)
        self.areas = AreaRepository(db)

    async def build(self, input_type: str = "edi", lag: int = 6,
                    as_of: tuple[int, int] | None = None) -> DashboardOut:
        run, as_of_row, n_neigh, city_rows, by_neighborhood = \
            await self.forecasts.city_and_neighborhoods(input_type, lag, as_of)

        history: dict[int, list[float]] = defaultdict(list)
        this_week: dict[int, int] = {}
        for nid, start, cases in await self.areas.weekly_cases_all(until=as_of_row.start_date):
            history[nid].append(cases)
            if start == as_of_row.start_date:
                this_week[nid] = cases

        alerts, bubbles = [], []
        for n in await self.areas.neighborhoods():
            nb = n.Neighborhood
            rows = by_neighborhood[nb.id]
            peak_f, peak_week = max(rows, key=lambda fw: fw[0].predicted)
            level, thr = relative_risk(np.array(history[nb.id]), peak_f.predicted)
            if level != "low":
                alerts.append(AlertOut(
                    neighborhood_id=nb.id, name=nb.name, risk_level=level, week_start=peak_week.start_date,
                    message=(f"Forecast of {peak_f.predicted:.1f} cases in epidemiological week "
                             f"{peak_week.epi_year}-{peak_week.epi_week:02d}, above this neighborhood's "
                             f"{thr.percentiles[0 if level == 'moderate' else 1]}th-percentile week "
                             f"({thr.moderate_from if level == 'moderate' else thr.high_from:.1f} cases). "
                             "Estimated from the city forecast and the neighborhood's usual share of cases."),
                ))
            if nb.latitude is not None and nb.longitude is not None:
                bubbles.append(MapBubbleOut(
                    neighborhood_id=nb.id, name=nb.name, latitude=nb.latitude, longitude=nb.longitude,
                    total_cases_this_week=int(this_week.get(nb.id, 0)),
                    predicted_next_week=round(rows[0][0].predicted, 2),
                    predicted_peak=round(peak_f.predicted, 2), risk_level=level,
                ))
        alerts.sort(key=lambda a: (_ORDER[a.risk_level], a.name))

        city_points = [
            ForecastPointOut(horizon=f.horizon, week_start=w.start_date, epi_year=w.epi_year, epi_week=w.epi_week,
                             predicted_cases=round(f.predicted, 2),
                             lower_bound=None if f.lower is None else round(f.lower, 2),
                             upper_bound=None if f.upper is None else round(f.upper, 2),
                             mean_per_neighborhood=round(f.predicted / n_neigh, 3) if n_neigh else None)
            for f, w in city_rows
        ]
        return DashboardOut(
            as_of=WeekOut(epi_year=as_of_row.epi_year, epi_week=as_of_row.epi_week, week_start=as_of_row.start_date),
            model_version=run.model_version, input_type=run.input_type, lag=run.lag,
            total_cases_this_week=int(sum(this_week.values())),
            neighborhoods_on_alert=len(alerts),
            mean_predicted_next_weeks=round(float(np.mean([p.predicted_cases for p in city_points])), 2),
            city_forecast=city_points, alerts=alerts, map=bubbles,
        )
