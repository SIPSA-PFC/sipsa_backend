"""
main.py
-------
FastAPI application entry point.

Run with:
    uvicorn src.main:app --reload --port 8000

The API starts without model weights. The /predicao endpoint will return
503 if weights are not available. All other endpoints work immediately.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.api.controllers import predicao, casos, ovitrampas, auth, resultados
from src.ml.inference import is_model_ready, load_weights

logger = logging.getLogger("sipsa")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load weights without blocking startup if they are missing.
    # load_weights() does not raise when the file is absent, so check the
    # registry afterwards instead of relying on an exception.
    load_weights(settings.model_weights_path)
    if is_model_ready():
        logger.info("Model weights loaded from %s", settings.model_weights_path)
    else:
        logger.warning(
            "Model weights not found at %s; /predicao will return 503 until "
            "they exist. Generate them with: python -m src.ml.train",
            settings.model_weights_path,
        )
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
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(casos.router)
app.include_router(ovitrampas.router)
app.include_router(predicao.router)
app.include_router(resultados.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}