"""GET /dashboard — the main panel in one request."""
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.controllers.areas import parse_as_of
from src.api.models.schemas import DashboardOut, Lag
from src.api.services.dashboard_service import DashboardService
from src.core.database import get_db

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("", response_model=DashboardOut)
async def get_dashboard(
    input_type: Literal["dengue", "edi"] = Query(default="edi"),
    lag: Lag = Query(default=Lag.SIX),
    as_of: Optional[tuple[int, int]] = Depends(parse_as_of),
    db: AsyncSession = Depends(get_db),
):
    """
    The week's cases, city forecast, and neighborhoods at moderate or high forecast risk —
    for the latest week, or replayed as of an earlier week (as_of=YYYY-WW).
    """
    return await DashboardService(db).build(input_type, int(lag), as_of)
