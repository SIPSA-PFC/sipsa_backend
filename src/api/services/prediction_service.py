"""
api/services/prediction_service.py
Orchestrates: validate → aggregate → LSTM → cache → return.
"""
from __future__ import annotations
from datetime import datetime, timezone
from fastapi import HTTPException, status

from src.api.models.db_models import ModelPrediction
from src.api.models.schemas import PredictionOut, PredictionPointOut
from src.api.repositories.neighborhood_repository import NeighborhoodRepository, PredictionRepository
from src.api.repositories.dengue_repository import DengueRepository
from src.api.repositories.ovitrap_repository import OvitrapRepository
from src.api.services.validation_service import ValidationService
from src.api.services.aggregation_service import AggregationService
from src.core import epiweek
from src.ml.inference import ModelNotReadyError, run_prediction

MODEL_VERSION = "sipsa-lstm-v1"


def _risk_level(mean_cases: float) -> str:
    if mean_cases > 20: return "high"
    if mean_cases > 10: return "moderate"
    return "low"


def _next_week(epi_year: int, epi_week: int):
    """Following epidemiological week: (epi_year, epi_week, start Sunday)."""
    ny, nw = epiweek.add_weeks(epi_year, epi_week, 1)
    return ny, nw, epiweek.week_start(ny, nw)


class PredictionService:
    def __init__(
        self,
        validation_svc:    ValidationService,
        aggregation_svc:   AggregationService,
        neighborhood_repo: NeighborhoodRepository,
        prediction_repo:   PredictionRepository,
    ):
        self.validation_svc    = validation_svc
        self.aggregation_svc   = aggregation_svc
        self.neighborhood_repo = neighborhood_repo
        self.prediction_repo   = prediction_repo

    async def predict(
        self,
        neighborhood_id: int,
        input_type:      str = "dengue",
        lag:             int = 1,
        n_weeks:         int = 4,
    ) -> PredictionOut:

        # 1. Validate
        await self.validation_svc.assert_exists(neighborhood_id)
        await self.validation_svc.assert_has_geolocation(neighborhood_id)
        await self.validation_svc.assert_sufficient_history(neighborhood_id)

        neighborhood = await self.neighborhood_repo.get_by_id(neighborhood_id)

        # 2. Return cache if it covers the requested horizon.
        #    The cache holds the longest horizon computed so far for this
        #    configuration; shorter requests are served from its first weeks.
        cached = await self.prediction_repo.get_cached(neighborhood_id, MODEL_VERSION, input_type, lag)
        if len(cached) >= n_weeks:
            return self._format(cached[:n_weeks], neighborhood.name, input_type, lag)

        # 3. Build input series
        series = await self.aggregation_svc.get_aligned_series()

        # 4. Run LSTM (503 if the model weights have not been generated yet)
        try:
            result = run_prediction(
                dengue_series  = series["dengue"],
                edi_series     = series["edi"],
                input_type     = input_type,
                lag            = lag,
                n_future_weeks = n_weeks,
                n_reps         = 5,
            )
        except ModelNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

        # 5. Build future week rows
        last_year = series["epi_years"][-1]
        last_week = series["epi_weeks"][-1]
        rows: list[ModelPrediction] = []

        for i in range(n_weeks):
            y = last_year if i == 0 else rows[-1].epi_year
            w = last_week if i == 0 else rows[-1].epi_week
            ny, nw, nd = _next_week(y, w)
            rows.append(ModelPrediction(
                neighborhood_id = neighborhood_id,
                week_start      = nd,
                epi_year        = ny,
                epi_week        = nw,
                predicted_cases = float(result["mean"][i]),
                lower_bound     = float(result["lower"][i]),
                upper_bound     = float(result["upper"][i]),
                model_version   = MODEL_VERSION,
                input_type      = input_type,
                lag             = lag,
                generated_at    = datetime.now(timezone.utc),
            ))

        # 6. Persist cache
        await self.prediction_repo.save(rows)

        return self._format(rows, neighborhood.name, input_type, lag)

    @staticmethod
    def _format(rows, name, input_type, lag) -> PredictionOut:
        mean = sum(float(r.predicted_cases) for r in rows) / len(rows)
        return PredictionOut(
            neighborhood_id   = rows[0].neighborhood_id,
            neighborhood_name = name,
            input_type        = input_type,
            lag               = lag,
            model_version     = MODEL_VERSION,
            generated_at      = rows[0].generated_at or datetime.now(timezone.utc),
            risk_level        = _risk_level(mean),
            series=[
                PredictionPointOut(
                    week_start      = r.week_start,
                    epi_year        = r.epi_year,
                    epi_week        = r.epi_week,
                    predicted_cases = float(r.predicted_cases),
                    lower_bound     = float(r.lower_bound) if r.lower_bound is not None else None,
                    upper_bound     = float(r.upper_bound) if r.upper_bound is not None else None,
                )
                for r in rows
            ],
        )