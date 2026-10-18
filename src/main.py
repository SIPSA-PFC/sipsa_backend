"""
main.py
-------
FastAPI application entry point.

Run with:
    uvicorn src.main:app --reload --port 8000

The API starts without trained models; /predicao returns 503 until
`python -m src.ml.train` has registered some. Other endpoints work immediately.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.api.controllers import predicao, casos, ovitrampas, auth, resultados
from src.api.controllers.auth import get_current_user
from src.core.security import burn_password_check
from src.core.database import AsyncSessionLocal
from src.ml.inference import registry

# uvicorn only prints its own loggers; use its error logger (stderr) so the
# startup messages appear in the server output.
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Find the trained models registered in model_run. They are loaded into
    # memory on first use. /predicao returns 503 for configurations without one.
    async with AsyncSessionLocal() as session:
        await registry.refresh(session)
    if registry.available:
        logger.info("Trained models: %s", ", ".join(
            f"{r.input_type}/lag {r.lag} ({r.model_version}, {r.n_reps} reps)" for r in registry.available))
        # Load the default configuration (EDI, lag 6) in the background, so the
        # first forecast request does not wait for its saved models.
        default = next((r for r in registry.available if (r.input_type, r.lag) == ("edi", 6)), None)
        if default is not None:
            asyncio.get_running_loop().run_in_executor(None, registry.models, default)
    else:
        logger.warning("No trained models registered; /predicao will return 503. "
                       "Train with: python -m src.ml.train")
    # Build the bcrypt dummy hash now, so the first failed login is not slower
    # than later ones (equal timing hides which usernames exist).
    burn_password_check("")
    yield


app = FastAPI(
    title="SIPSA — Dengue Forecasting API",
    description=(
        "1D-LSTM model for dengue incidence forecasting — Natal, RN-Brazil.\n\n"
        "Based on: *Data-Driven Computational Intelligence Applied to "
        "Dengue Outbreak Forecasting.*"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Public: login, health check, and the paper's published evaluation results.
app.include_router(auth.router)
app.include_router(resultados.router)

# Data routes require a logged-in user (any role).
_login_required = [Depends(get_current_user)]
app.include_router(casos.router,      dependencies=_login_required)
app.include_router(ovitrampas.router, dependencies=_login_required)
app.include_router(predicao.router,   dependencies=_login_required)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}