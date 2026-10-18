"""
api/models/schemas.py
---------------------
Pydantic schemas — the typed contract between FastAPI and the Next.js frontend.

Design rule: every field name here is what the frontend will receive in JSON.
The frontend dev should import these types as-is to TypeScript interfaces.
"""

from __future__ import annotations
from datetime import date, datetime
from enum import IntEnum
from typing import Optional
from pydantic import BaseModel


# =============================================================================
# Model configuration
# =============================================================================

class Lag(IntEnum):
    """
    Past-sample windows the LSTM was trained with (MATLAB NBackSamples = [1 3:6]).
    1 -> i-1;  3 -> i-3..i-1;  4 -> i-4..i-2;  5 -> i-5..i-3;  6 -> i-6..i-4.
    Any other value is invalid: lag=2 would include week i itself (target leakage).
    """
    ONE   = 1
    THREE = 3
    FOUR  = 4
    FIVE  = 5
    SIX   = 6


# =============================================================================
# Neighborhood
# =============================================================================

class NeighborhoodOut(BaseModel):
    id:        int
    name:      str
    latitude:  Optional[float] = None
    longitude: Optional[float] = None

    class Config: from_attributes = True


# =============================================================================
# Dengue cases  — GET /casos/{neighborhood_id}
# =============================================================================

class DengueCaseOut(BaseModel):
    neighborhood_id: int
    week_start:      date
    iso_year:        int
    iso_week:        int
    case_count:      Optional[float] = None

    class Config: from_attributes = True


class DengueCaseSeriesOut(BaseModel):
    neighborhood_id:   int
    neighborhood_name: str
    series:            list[DengueCaseOut]


# =============================================================================
# Ovitrap readings  — GET /ovitrampas/{neighborhood_id}
# OPI and EDI are returned together (same source) but labelled separately
# so the frontend can choose which to display
# =============================================================================

class OvitrapReadingOut(BaseModel):
    neighborhood_id: int
    week_start:      date
    iso_year:        int
    iso_week:        int
    opi:             Optional[float] = None
    edi:             Optional[float] = None

    class Config: from_attributes = True


class OvitrapSeriesOut(BaseModel):
    neighborhood_id:   int
    neighborhood_name: str
    series:            list[OvitrapReadingOut]


# =============================================================================
# Prediction  — GET /predicao/{neighborhood_id}
# predicted_cases is already back-transformed (2^YPred) — real case units
# =============================================================================

class PredictionPointOut(BaseModel):
    week_start:       date
    iso_year:         int
    iso_week:         int
    predicted_cases:  float
    lower_bound:      Optional[float] = None
    upper_bound:      Optional[float] = None


class PredictionOut(BaseModel):
    neighborhood_id:   int
    neighborhood_name: str
    input_type:        str        # "dengue" | "edi"
    lag:               int
    model_version:     str
    generated_at:      datetime
    risk_level:        str        # "high" | "moderate" | "low"
    series:            list[PredictionPointOut]


# =============================================================================
# Dashboard  — GET /dashboard
# Aggregates everything the main panel needs in one request
# =============================================================================

class MapBubbleOut(BaseModel):
    """One bubble per neighborhood on the Leaflet map."""
    neighborhood_id:       int
    name:                  str
    latitude:              float
    longitude:             float
    total_cases_this_week: int
    predicted_next_week:   Optional[float] = None
    risk_level:            str


class AlertOut(BaseModel):
    """One card per high/moderate-risk neighborhood."""
    neighborhood_id: int
    name:            str
    risk_level:      str
    message:         str
    week_start:      date


class DashboardOut(BaseModel):
    total_cases_this_week:   int
    neighborhoods_on_alert:  int
    mean_predicted_4_weeks:  float
    alerts:                  list[AlertOut]
    map:                     list[MapBubbleOut]


# =============================================================================
# Auth  — POST /auth/token
# =============================================================================

class TokenOut(BaseModel):
    access_token: str
    token_type:   str = "bearer"


# =============================================================================
# Pre-computed results  — GET /resultados
# Served directly from the .mat file, no training required
# =============================================================================

class ModelMetricsOut(BaseModel):
    """Performance metrics for one model configuration."""
    model_index:  int
    input_type:   str        # "dengue" | "edi"
    lag_label:    str        # "1 past sample" | "3:1 past samples" etc.
    mean_rmse:    float
    std_err_rmse: float
    mean_r:       float
    std_err_r:    float
    risk_level:   str        # "high" | "moderate" | "low" based on RMSE


class TimeSeriesPointOut(BaseModel):
    """One data point in a predicted vs actual series."""
    week_start:      str     # ISO date string "2019-03-18"
    actual:          float
    predicted_mean:  float
    lower_bound:     float
    upper_bound:     float


class BestModelSeriesOut(BaseModel):
    """Full time series for one of the two best models."""
    model_index:  int
    input_type:   str
    lag_label:    str
    mean_r:       float
    mean_rmse:    float
    series:       list[TimeSeriesPointOut]


class ResultadosOut(BaseModel):
    """
    Full response for GET /resultados.
    Contains everything the frontend needs to populate:
        - model performance bar charts (metrics)
        - predicted vs actual line charts (best_models)
        - model accuracy gauge (best_r, best_rmse)
    """
    metrics:      list[ModelMetricsOut]     # all 10 models
    best_models:  list[BestModelSeriesOut]  # D→D best + O→D best
    best_r:       float                     # highest r across all models
    best_rmse:    float                     # lowest RMSE across all models
    test_period_start: str
    test_period_end:   str
    n_reps:       int                       # 30 — number of training repetitions