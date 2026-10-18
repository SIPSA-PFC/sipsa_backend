"""GET /predicao — city-wide dengue forecast from the saved LSTM models."""
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.schemas import ForecastOut, Lag
from src.api.services.forecast_service import ForecastService
from src.core.database import get_db

router = APIRouter(prefix="/predicao", tags=["predicao"])


@router.get("", response_model=ForecastOut)
async def get_predicao(
    input_type: Literal["dengue", "edi"] = Query(
        default="edi", description="Predictor: past ovitrap egg density (edi) or past dengue cases"),
    lag: Lag = Query(default=Lag.SIX, description="Input window: 1, 3 (weeks 3–1), 4 (4–2), 5 (5–3), 6 (6–4)"),
    n_weeks: Optional[int] = Query(
        default=None, ge=1, le=4,
        description="Weeks ahead; at most the window's newest lag (lag 1/3: 1, 4: 2, 5: 3, 6: 4). "
                    "Default: the maximum."),
    db: AsyncSession = Depends(get_db),
):
    """
    Forecast weekly dengue cases for the whole city after the latest observed week.

    The default (EDI, lag 6) uses ovitrap egg density from 6 to 4 weeks earlier,
    so it forecasts 4 weeks ahead; in the paper's test period it was the only
    kind of configuration that beat a naive forecast with the same lead time.
    """
    return await ForecastService(db).city(input_type, int(lag), n_weeks)
