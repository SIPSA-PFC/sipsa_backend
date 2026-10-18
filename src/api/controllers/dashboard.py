"""GET /dashboard — the main panel in one request."""
from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.schemas import DashboardOut, Lag
from src.api.services.dashboard_service import DashboardService
from src.core.database import get_db

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("", response_model=DashboardOut)
async def get_dashboard(
    input_type: Literal["dengue", "edi"] = Query(default="edi"),
    lag: Lag = Query(default=Lag.SIX),
    db: AsyncSession = Depends(get_db),
):
    """Latest week's cases, city forecast, and neighborhoods at moderate or high forecast risk."""
    return await DashboardService(db).build(input_type, int(lag))
