"""GET /predicao/{neighborhood_id}"""
from typing import Literal
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.api.models.schemas import Lag, PredictionOut
from src.api.repositories.neighborhood_repository import NeighborhoodRepository, PredictionRepository
from src.api.repositories.dengue_repository import DengueRepository
from src.api.repositories.ovitrap_repository import OvitrapRepository
from src.api.services.validation_service import ValidationService
from src.api.services.aggregation_service import AggregationService
from src.api.services.prediction_service import PredictionService

router = APIRouter(prefix="/predicao", tags=["predicao"])

def _svc(db: AsyncSession) -> PredictionService:
    nr  = NeighborhoodRepository(db)
    dr  = DengueRepository(db)
    or_ = OvitrapRepository(db)
    return PredictionService(
        validation_svc    = ValidationService(nr, dr),
        aggregation_svc   = AggregationService(dr, or_),
        neighborhood_repo = nr,
        prediction_repo   = PredictionRepository(db),
    )

@router.get("/{neighborhood_id}", response_model=PredictionOut)
async def get_predicao(
    neighborhood_id: int,
    input_type: Literal["dengue", "edi"] = Query(default="dengue"),
    lag:        Lag = Query(default=Lag.ONE),
    n_weeks:    int = Query(default=4, ge=1, le=12),
    db: AsyncSession = Depends(get_db),
):
    """
    Predict dengue incidence for the next n_weeks.
    - **input_type**: `dengue` or `edi`
    - **lag**: past-sample window 1, 3, 4, 5 or 6
    - **n_weeks**: forecast horizon 1–12
    """
    return await _svc(db).predict(neighborhood_id, input_type, int(lag), n_weeks)