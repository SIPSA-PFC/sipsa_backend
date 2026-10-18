"""
api/models/db_models.py
-----------------------
SQLAlchemy ORM models.

Reference data
    data_source       — where each dataset came from (file/URL, license, citation)
    district          — Natal's health districts (Norte I, Norte II, Sul, Leste, Oeste)
    neighborhood      — the 36 bairros, with official code and coordinates
    socioeconomic     — 2010 Census snapshot per neighborhood
    epi_week          — calendar of epidemiological weeks (Sunday–Saturday)

Weekly measurements (one row per neighborhood per epidemiological week)
    dengue_case       — notified cases (whole numbers)
    ovitrap_reading   — ovitrap positivity (OPI) and egg density (EDI)

Access
    users             — accounts that can log in to the API (role admin | user)

Models
    model_run         — one trained model configuration (input, lag) of a
                        training run: training window, metrics, saved files

Forecasts
    forecast          — model output per area, as-of week and target week

Dengue cases and ovitrap readings are different measurement systems and are
stored and queried separately.
"""

from datetime import date, datetime

from sqlalchemy import (
    JSON, Boolean, CheckConstraint, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text,
    UniqueConstraint, func, text, true,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.core.database import Base


# =============================================================================
# Reference data
# =============================================================================

class DataSource(Base):
    """Provenance: every loaded dataset points at one row here."""
    __tablename__ = "data_source"

    id:          Mapped[int]             = mapped_column(Integer, primary_key=True)
    name:        Mapped[str]             = mapped_column(String(200), nullable=False, unique=True)
    file_or_url: Mapped[str | None]      = mapped_column(String(500))
    license:     Mapped[str | None]      = mapped_column(String(120))
    citation:    Mapped[str | None]      = mapped_column(Text)
    imported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())


class District(Base):
    """Health district (distrito sanitário) grouping several neighborhoods."""
    __tablename__ = "district"

    id:   Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(60), nullable=False, unique=True)

    neighborhoods: Mapped[list["Neighborhood"]] = relationship(back_populates="district")


class Neighborhood(Base):
    __tablename__ = "neighborhood"

    id:                 Mapped[int]          = mapped_column(Integer, primary_key=True)
    code:               Mapped[str | None]   = mapped_column(String(10), unique=True)   # official municipal code
    name:               Mapped[str]          = mapped_column(String(120), nullable=False, unique=True)
    district_id:        Mapped[int | None]   = mapped_column(ForeignKey("district.id"), index=True)
    latitude:           Mapped[float | None] = mapped_column(Float)
    longitude:          Mapped[float | None] = mapped_column(Float)
    coord_source_id:    Mapped[int | None]   = mapped_column(ForeignKey("data_source.id"))
    coord_retrieved_at: Mapped[date | None]  = mapped_column(Date)

    district:         Mapped["District | None"]       = relationship(back_populates="neighborhoods")
    socioeconomic:    Mapped[list["Socioeconomic"]]   = relationship(back_populates="neighborhood")
    dengue_cases:     Mapped[list["DengueCase"]]      = relationship(back_populates="neighborhood")
    ovitrap_readings: Mapped[list["OvitrapReading"]]  = relationship(back_populates="neighborhood")


class Socioeconomic(Base):
    """
    Census snapshot per neighborhood. Source columns (DADOS GERAIS_01.xlsx,
    sheet 'Renda por Bairro') are from the 2010 Census.
    """
    __tablename__ = "socioeconomic"
    __table_args__ = (UniqueConstraint("neighborhood_id", "reference_year"),)

    id:                    Mapped[int]          = mapped_column(Integer, primary_key=True)
    neighborhood_id:       Mapped[int]          = mapped_column(ForeignKey("neighborhood.id"), nullable=False)
    reference_year:        Mapped[int]          = mapped_column(Integer, nullable=False, default=2010)
    population:            Mapped[int | None]   = mapped_column(Integer)
    total_income_brl:      Mapped[float | None] = mapped_column(Float)   # 'Renda Mensal Total R$'
    income_per_capita_brl: Mapped[float | None] = mapped_column(Float)   # 'Renda Mensal por Pessoa R$'
    source_id:             Mapped[int | None]   = mapped_column(ForeignKey("data_source.id"))

    neighborhood: Mapped["Neighborhood"] = relationship(back_populates="socioeconomic")


class EpiWeek(Base):
    """Calendar of epidemiological weeks; see src/core/epiweek.py for the rules."""
    __tablename__ = "epi_week"
    __table_args__ = (
        UniqueConstraint("epi_year", "epi_week"),
        CheckConstraint("epi_week BETWEEN 1 AND 53", name="week_range"),
    )

    id:         Mapped[int]  = mapped_column(Integer, primary_key=True)
    epi_year:   Mapped[int]  = mapped_column(Integer, nullable=False)
    epi_week:   Mapped[int]  = mapped_column(Integer, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False, unique=True)   # Sunday
    end_date:   Mapped[date] = mapped_column(Date, nullable=False)                # Saturday


# =============================================================================
# Weekly measurements
# =============================================================================

class DengueCase(Base):
    """Notified cases per neighborhood per epidemiological week."""
    __tablename__ = "dengue_case"
    __table_args__ = (
        UniqueConstraint("neighborhood_id", "epi_week_id", "disease"),
        CheckConstraint("case_count >= 0", name="non_negative"),
    )

    id:              Mapped[int]        = mapped_column(Integer, primary_key=True)
    neighborhood_id: Mapped[int]        = mapped_column(ForeignKey("neighborhood.id"), nullable=False, index=True)
    epi_week_id:     Mapped[int]        = mapped_column(ForeignKey("epi_week.id"), nullable=False, index=True)
    disease:         Mapped[str]        = mapped_column(String(20), nullable=False, default="dengue", server_default="dengue")
    case_count:      Mapped[int | None] = mapped_column(Integer)
    source_id:       Mapped[int | None] = mapped_column(ForeignKey("data_source.id"))

    neighborhood: Mapped["Neighborhood"] = relationship(back_populates="dengue_cases")
    week:         Mapped["EpiWeek"]      = relationship()


class OvitrapReading(Base):
    """
    Weekly ovitrap indices per neighborhood: OPI (share of positive traps,
    0–1) and EDI (egg density index). Same source, so stored together.
    """
    __tablename__ = "ovitrap_reading"
    __table_args__ = (
        UniqueConstraint("neighborhood_id", "epi_week_id"),
        CheckConstraint("opi IS NULL OR (opi >= 0 AND opi <= 1)", name="opi_range"),
    )

    id:              Mapped[int]          = mapped_column(Integer, primary_key=True)
    neighborhood_id: Mapped[int]          = mapped_column(ForeignKey("neighborhood.id"), nullable=False, index=True)
    epi_week_id:     Mapped[int]          = mapped_column(ForeignKey("epi_week.id"), nullable=False, index=True)
    opi:             Mapped[float | None] = mapped_column(Float)
    edi:             Mapped[float | None] = mapped_column(Float)
    source_id:       Mapped[int | None]   = mapped_column(ForeignKey("data_source.id"))

    neighborhood: Mapped["Neighborhood"] = relationship(back_populates="ovitrap_readings")
    week:         Mapped["EpiWeek"]      = relationship()


# =============================================================================
# Models
# =============================================================================

class ModelRun(Base):
    """
    One model configuration from one training run (python -m src.ml.train).
    A run trains 10 configurations (2 inputs x 5 lags), each with n_reps
    independent repetitions saved under artifact_path.
    """
    __tablename__ = "model_run"
    __table_args__ = (
        UniqueConstraint("model_version", "input_type", "lag"),
        CheckConstraint("input_type IN ('dengue', 'edi')", name="input_type"),
    )

    id:                 Mapped[int]             = mapped_column(Integer, primary_key=True)
    model_version:      Mapped[str]             = mapped_column(String(60), nullable=False)   # training run id
    input_type:         Mapped[str]             = mapped_column(String(10), nullable=False)
    lag:                Mapped[int]             = mapped_column(Integer, nullable=False)
    train_from_week_id: Mapped[int]             = mapped_column(ForeignKey("epi_week.id"), nullable=False)
    train_to_week_id:   Mapped[int]             = mapped_column(ForeignKey("epi_week.id"), nullable=False)
    n_reps:             Mapped[int]             = mapped_column(Integer, nullable=False)
    metrics:            Mapped[dict | None]     = mapped_column(JSON)
    artifact_path:      Mapped[str]             = mapped_column(String(500), nullable=False)  # relative to repo root
    created_at:         Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())

    train_from_week: Mapped["EpiWeek"] = relationship(foreign_keys=[train_from_week_id])
    train_to_week:   Mapped["EpiWeek"] = relationship(foreign_keys=[train_to_week_id])


# =============================================================================
# Access
# =============================================================================

USER_ROLES = ("admin", "user")


class User(Base):
    """
    API account. Passwords are stored only as bcrypt hashes.
    Table is 'users' because 'user' is a reserved word in PostgreSQL.
    """
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('admin', 'user')", name="role"),
    )

    id:            Mapped[int]             = mapped_column(Integer, primary_key=True)
    username:      Mapped[str]             = mapped_column(String(80), nullable=False, unique=True)
    password_hash: Mapped[str]             = mapped_column(String(100), nullable=False)
    role:          Mapped[str]             = mapped_column(String(10), nullable=False, default="user", server_default="user")
    is_active:     Mapped[bool]            = mapped_column(Boolean, nullable=False, default=True, server_default=true())
    created_at:    Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())


# =============================================================================
# Forecasts
# =============================================================================

FORECAST_SCOPES = ("city", "district", "neighborhood")
FORECAST_METHODS = ("model", "split")


class Forecast(Base):
    """
    One forecast value: what a model run predicted for one area and target
    week, using data observed up to the as-of week.

    predicted / lower / upper are weekly cases in the area (for the city: all
    neighborhoods together). method is 'model' for the LSTM output itself and
    'split' for values derived from it (district / neighborhood shares).

    Target weeks may lie after the last observed week, so they are rows of
    epi_week created on demand.
    """
    __tablename__ = "forecast"
    __table_args__ = (
        CheckConstraint("scope IN ('city', 'district', 'neighborhood')", name="scope"),
        CheckConstraint("method IN ('model', 'split')", name="method"),
        CheckConstraint(
            "(scope = 'city' AND district_id IS NULL AND neighborhood_id IS NULL)"
            " OR (scope = 'district' AND district_id IS NOT NULL AND neighborhood_id IS NULL)"
            " OR (scope = 'neighborhood' AND neighborhood_id IS NOT NULL AND district_id IS NULL)",
            name="area_matches_scope",
        ),
        CheckConstraint("horizon >= 1", name="horizon_positive"),
        # One value per run, area, as-of week and target week. COALESCE because
        # NULLs never collide in a unique index (the city has no area id).
        Index(
            "uq_forecast_key",
            "model_run_id", "scope",
            func.coalesce(text("district_id"), 0), func.coalesce(text("neighborhood_id"), 0),
            "as_of_week_id", "target_week_id",
            unique=True,
        ),
    )

    id:              Mapped[int]             = mapped_column(Integer, primary_key=True)
    model_run_id:    Mapped[int]             = mapped_column(ForeignKey("model_run.id", ondelete="CASCADE"), nullable=False, index=True)
    scope:           Mapped[str]             = mapped_column(String(12), nullable=False)
    district_id:     Mapped[int | None]      = mapped_column(ForeignKey("district.id"))
    neighborhood_id: Mapped[int | None]      = mapped_column(ForeignKey("neighborhood.id"))
    as_of_week_id:   Mapped[int]             = mapped_column(ForeignKey("epi_week.id"), nullable=False)
    target_week_id:  Mapped[int]             = mapped_column(ForeignKey("epi_week.id"), nullable=False)
    horizon:         Mapped[int]             = mapped_column(Integer, nullable=False)    # weeks after as-of
    predicted:       Mapped[float]           = mapped_column(Float, nullable=False)
    lower:           Mapped[float | None]    = mapped_column(Float)
    upper:           Mapped[float | None]    = mapped_column(Float)
    method:          Mapped[str]             = mapped_column(String(10), nullable=False, default="model")
    generated_at:    Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())

    model_run:   Mapped["ModelRun"] = relationship()
    target_week: Mapped["EpiWeek"]  = relationship(foreign_keys=[target_week_id])
