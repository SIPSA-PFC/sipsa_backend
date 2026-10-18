"""
api/models/db_models.py
-----------------------
SQLAlchemy ORM models — clean relational schema derived directly from
the original research data (36 neighborhoods, 208 weeks, two measurement
types, one static socioeconomic snapshot).

Tables
------
    neighborhood      — the 36 bairros of Natal, RN
    socioeconomic     — static income/population snapshot per neighborhood
    dengue_case       — weekly case counts (Dados_Modelagem.xlsx sheet 4)
    ovitrap_reading   — weekly OPI + EDI (Rearranged_Data.mat)
    model_prediction  — cached LSTM output (derived, not source data)
"""

from datetime import date, datetime
from sqlalchemy import (
    Boolean, CheckConstraint, Date, DateTime, Numeric,
    ForeignKey, Integer, String, UniqueConstraint, func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.core.database import Base


class Neighborhood(Base):
    __tablename__ = "neighborhood"

    id:        Mapped[int]          = mapped_column(Integer, primary_key=True, autoincrement=True)
    name:      Mapped[str]          = mapped_column(String(120), nullable=False, unique=True)
    latitude:  Mapped[float | None] = mapped_column(Numeric(10, 8))
    longitude: Mapped[float | None] = mapped_column(Numeric(11, 8))

    socioeconomic:    Mapped[list["Socioeconomic"]]   = relationship(back_populates="neighborhood")
    dengue_cases:     Mapped[list["DengueCase"]]       = relationship(back_populates="neighborhood")
    ovitrap_readings: Mapped[list["OvitrapReading"]]   = relationship(back_populates="neighborhood")
    predictions:      Mapped[list["ModelPrediction"]]  = relationship(back_populates="neighborhood")


class Socioeconomic(Base):
    """
    Static snapshot per neighborhood.
    reference_year allows future snapshots without schema changes.
    """
    __tablename__ = "socioeconomic"
    __table_args__ = (
        UniqueConstraint("neighborhood_id", "reference_year", name="uq_socio_year"),
    )

    id:               Mapped[int]          = mapped_column(Integer, primary_key=True, autoincrement=True)
    neighborhood_id:  Mapped[int]          = mapped_column(ForeignKey("neighborhood.id"), nullable=False)
    reference_year:   Mapped[int]          = mapped_column(Integer, nullable=False, default=2016)
    population:       Mapped[int | None]   = mapped_column(Integer)
    income_index:     Mapped[float | None] = mapped_column(Numeric(10, 2))

    neighborhood: Mapped["Neighborhood"] = relationship(back_populates="socioeconomic")


class DengueCase(Base):
    """
    Weekly dengue incidence per neighborhood.
    Source: Dados_Modelagem.xlsx sheet 4 — nanmean across the neighborhood matrix.
    case_count is NUMERIC(10,4) not INT because the source is already a float mean.
    """
    __tablename__ = "dengue_case"
    __table_args__ = (
        UniqueConstraint("neighborhood_id", "iso_year", "iso_week", name="uq_dengue_week"),
        CheckConstraint("iso_week BETWEEN 1 AND 53", name="chk_dengue_iso_week"),
    )

    id:              Mapped[int]          = mapped_column(Integer, primary_key=True, autoincrement=True)
    neighborhood_id: Mapped[int]          = mapped_column(ForeignKey("neighborhood.id"), nullable=False)
    week_start:      Mapped[date]         = mapped_column(Date, nullable=False)
    iso_year:        Mapped[int]          = mapped_column(Integer, nullable=False)
    iso_week:        Mapped[int]          = mapped_column(Integer, nullable=False)
    case_count:      Mapped[float | None] = mapped_column(Numeric(10, 4))

    neighborhood: Mapped["Neighborhood"] = relationship(back_populates="dengue_cases")


class OvitrapReading(Base):
    """
    Weekly OPI + EDI per neighborhood.
    Source: Rearranged_Data.mat EggIndice_Agregated — slice 0 = OPI, slice 1 = EDI.
    Stored together because they come from the same mat file slice.
    Exposed separately through the API (never combined in queries).
    """
    __tablename__ = "ovitrap_reading"
    __table_args__ = (
        UniqueConstraint("neighborhood_id", "iso_year", "iso_week", name="uq_ovitrap_week"),
        CheckConstraint("iso_week BETWEEN 1 AND 53", name="chk_ovitrap_iso_week"),
    )

    id:              Mapped[int]          = mapped_column(Integer, primary_key=True, autoincrement=True)
    neighborhood_id: Mapped[int]          = mapped_column(ForeignKey("neighborhood.id"), nullable=False)
    week_start:      Mapped[date]         = mapped_column(Date, nullable=False)
    iso_year:        Mapped[int]          = mapped_column(Integer, nullable=False)
    iso_week:        Mapped[int]          = mapped_column(Integer, nullable=False)
    opi:             Mapped[float | None] = mapped_column(Numeric(10, 4))
    edi:             Mapped[float | None] = mapped_column(Numeric(10, 4))

    neighborhood: Mapped["Neighborhood"] = relationship(back_populates="ovitrap_readings")


class ModelPrediction(Base):
    """
    Cached LSTM predictions — derived data, not source data.
    input_type + lag stored so multiple model configurations can coexist
    for the same neighborhood without overwriting each other.
    """
    __tablename__ = "model_prediction"
    __table_args__ = (
        # input_type and lag are part of the key: each configuration has its
        # own forecast for the same weeks.
        UniqueConstraint(
            "neighborhood_id", "iso_year", "iso_week",
            "model_version", "input_type", "lag",
            name="uq_prediction_config",
        ),
        CheckConstraint("input_type IN ('dengue','edi')", name="chk_input_type"),
    )

    id:               Mapped[int]          = mapped_column(Integer, primary_key=True, autoincrement=True)
    neighborhood_id:  Mapped[int]          = mapped_column(ForeignKey("neighborhood.id"), nullable=False)
    week_start:       Mapped[date]         = mapped_column(Date, nullable=False)
    iso_year:         Mapped[int]          = mapped_column(Integer, nullable=False)
    iso_week:         Mapped[int]          = mapped_column(Integer, nullable=False)
    predicted_cases:  Mapped[float]        = mapped_column(Numeric(10, 4), nullable=False)
    lower_bound:      Mapped[float | None] = mapped_column(Numeric(10, 4))
    upper_bound:      Mapped[float | None] = mapped_column(Numeric(10, 4))
    model_version:    Mapped[str]          = mapped_column(String(50), nullable=False)
    input_type:       Mapped[str]          = mapped_column(String(10), nullable=False)
    lag:              Mapped[int]          = mapped_column(Integer, nullable=False)
    generated_at:     Mapped[datetime]     = mapped_column(DateTime, server_default=func.now())

    neighborhood: Mapped["Neighborhood"] = relationship(back_populates="predictions")