"""GET /predicao — deprecated alias of GET /cidade/previsao."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.controllers.areas import ForecastParams
from src.api.models.schemas import ForecastOut
from src.api.services.forecast_service import ForecastService
from src.core.database import get_db

router = APIRouter(prefix="/predicao", tags=["predicao"])


@router.get("", response_model=ForecastOut, deprecated=True)
async def get_predicao(p: ForecastParams = Depends(), db: AsyncSession = Depends(get_db)):
    """Deprecated: use GET /cidade/previsao (same response)."""
    return await ForecastService(db).for_area("city", None, p.input_type, p.lag, p.n_weeks)
