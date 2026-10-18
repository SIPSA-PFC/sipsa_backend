"""
etl/migrate.py
--------------
Loads the original research files into the database.

Prerequisite: the schema must exist —  run  `alembic upgrade head`  first.

Run: python -m src.etl.migrate
"""
from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sqlalchemy import inspect, select

from src.api.models.db_models import (
    DengueCase, EpiWeek, Neighborhood, OvitrapReading, Socioeconomic,
)
from src.core import epiweek
from src.core.database import AsyncSessionLocal, engine

log = logging.getLogger("sipsa.etl")

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"

REQUIRED_TABLES = {"epi_week", "neighborhood", "socioeconomic", "dengue_case", "ovitrap_reading"}


# =============================================================================
# Readers
# =============================================================================

def _load_dengue() -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """
    Dados_Modelagem.xlsx, sheet 'Casos de Dengue': one row per neighborhood,
    one column per epidemiological week (S201601 … S201952), whole case counts.
    """
    df = pd.read_excel(DATA_RAW / "Dados_Modelagem.xlsx", sheet_name="Casos de Dengue", index_col=0)
    df.index = [str(n).strip() for n in df.index]
    weeks = [epiweek.parse_label(c) for c in df.columns]
    return df, weeks


def _load_egg_indices() -> tuple[np.ndarray, np.ndarray]:
    """Rearranged_Data.mat EggIndice_Agregated (36, 208, 2): [:, :, 0] OPI, [:, :, 1] EDI."""
    mat = sio.loadmat(str(DATA_RAW / "Rearranged_Data.mat"), squeeze_me=True)
    egg = mat["EggIndice_Agregated"]
    return egg[:, :, 0], egg[:, :, 1]


def _load_socio() -> pd.DataFrame:
    """DADOS GERAIS_01.xlsx, sheet 'Renda por Bairro' (2010 Census)."""
    df = pd.read_excel(DATA_RAW / "DADOS GERAIS_01.xlsx", sheet_name="Renda por Bairro")
    df.columns = ["name", "total_income_brl", "population"] + list(df.columns[3:])
    df["name"] = df["name"].astype(str).str.strip()
    df["population"]       = pd.to_numeric(df["population"], errors="coerce")
    df["total_income_brl"] = pd.to_numeric(df["total_income_brl"], errors="coerce")
    return df[["name", "population", "total_income_brl"]]


def _none_if_nan(value) -> float | None:
    return None if value is None or np.isnan(value) else float(value)


# =============================================================================
# Load
# =============================================================================

async def _assert_schema() -> None:
    async with engine.connect() as conn:
        tables = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))
    missing = REQUIRED_TABLES - tables
    if missing:
        raise SystemExit(
            f"Database schema missing tables {sorted(missing)}. "
            "Run  alembic upgrade head  before the ETL."
        )


async def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    missing = [f for f in ("Dados_Modelagem.xlsx", "DADOS GERAIS_01.xlsx", "Rearranged_Data.mat")
               if not (DATA_RAW / f).exists()]
    if missing:
        raise SystemExit(f"Missing files in {DATA_RAW}: {missing}")
    await _assert_schema()

    dengue_df, weeks = _load_dengue()
    opi_mat, edi_mat = _load_egg_indices()
    socio_df         = _load_socio()
    names            = list(dengue_df.index)

    async with AsyncSessionLocal() as session:
        # ── Epidemiological weeks ─────────────────────────────────────────────
        week_rows = []
        for y, w in weeks:
            start = epiweek.week_start(y, w)
            week_rows.append(EpiWeek(epi_year=y, epi_week=w, start_date=start,
                                     end_date=start + timedelta(days=6)))
        session.add_all(week_rows)
        await session.flush()
        log.info("%d epidemiological weeks (%s to %s)", len(week_rows),
                 week_rows[0].start_date, week_rows[-1].end_date)

        # ── Neighborhoods ─────────────────────────────────────────────────────
        session.add_all([Neighborhood(name=n) for n in names])
        await session.flush()
        name_to_id = {n.name: n.id for n in (await session.execute(select(Neighborhood))).scalars()}
        log.info("%d neighborhoods", len(name_to_id))

        # ── Socioeconomic (2010 Census) ───────────────────────────────────────
        n_socio = 0
        for _, row in socio_df.iterrows():
            nid = name_to_id.get(row["name"])
            if nid is None:
                continue
            session.add(Socioeconomic(
                neighborhood_id  = nid,
                reference_year   = 2010,
                population       = None if pd.isna(row["population"]) else int(row["population"]),
                total_income_brl = _none_if_nan(row["total_income_brl"]),
            ))
            n_socio += 1
        await session.flush()
        log.info("%d socioeconomic rows", n_socio)

        # ── Dengue cases ──────────────────────────────────────────────────────
        dengue_rows = []
        for wi, wk in enumerate(week_rows):
            for ni, name in enumerate(names):
                val = dengue_df.iat[ni, wi]
                dengue_rows.append(DengueCase(
                    neighborhood_id = name_to_id[name],
                    epi_week_id     = wk.id,
                    case_count      = None if pd.isna(val) else int(val),
                ))
        session.add_all(dengue_rows)
        await session.flush()
        log.info("%d dengue_case rows", len(dengue_rows))

        # ── Ovitrap readings ──────────────────────────────────────────────────
        # NOTE: rows of EggIndice_Agregated are matched to neighborhoods by
        # position here. fix/etl-alignment replaces this with a name-based match.
        ovitrap_rows = []
        for wi, wk in enumerate(week_rows):
            for ni, name in enumerate(names):
                ovitrap_rows.append(OvitrapReading(
                    neighborhood_id = name_to_id[name],
                    epi_week_id     = wk.id,
                    opi             = _none_if_nan(opi_mat[ni, wi]),
                    edi             = _none_if_nan(edi_mat[ni, wi]),
                ))
        session.add_all(ovitrap_rows)
        await session.flush()
        log.info("%d ovitrap_reading rows", len(ovitrap_rows))

        await session.commit()

    log.info("Migration complete.")


if __name__ == "__main__":
    asyncio.run(run())