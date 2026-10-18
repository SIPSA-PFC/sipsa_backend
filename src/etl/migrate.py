"""
etl/migrate.py
--------------
Loads the original research files into the database.

Prerequisites
- Schema:   alembic upgrade head
- Files in data/raw/:  Dados_Modelagem.xlsx, DADOS GERAIS_01.xlsx, Rearranged_Data.mat
- Reference file:      data/reference/neighborhoods.csv (committed)

Run: python -m src.etl.migrate

Safe to run again: reference rows (sources, districts, neighborhoods, weeks)
are updated in place, and measurements are replaced, all in one transaction.
Stored forecasts are cleared because they depend on the data.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sqlalchemy import delete, inspect, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.db_models import (
    DataSource, DengueCase, District, EpiWeek, Forecast, Neighborhood,
    OvitrapReading, Socioeconomic,
)
from src.core import epiweek
from src.core.database import AsyncSessionLocal, engine
from src.core.names import name_key

log = logging.getLogger("sipsa.etl")

ROOT          = Path(__file__).resolve().parent.parent.parent
DATA_RAW      = ROOT / "data" / "raw"
REFERENCE_CSV = ROOT / "data" / "reference" / "neighborhoods.csv"

REQUIRED_TABLES = {"data_source", "district", "epi_week", "neighborhood",
                   "socioeconomic", "dengue_case", "ovitrap_reading"}

CENSUS_YEAR = 2010

_PAPER = ("Data-Driven Computational Intelligence Applied to Dengue Outbreak Forecasting: "
          "a case study at the scale of the city of Natal, RN-Brazil. "
          "Code and data: https://github.com/isabgmb/dengueknowledgePaper_pi")
_PAPER_LICENSE = "CC BY-NC-SA 4.0"


@dataclass(frozen=True)
class SourceSpec:
    key:         str
    name:        str
    file_or_url: str
    license:     str
    citation:    str


SOURCES = {
    s.key: s for s in [
        SourceSpec("dengue", "Dengue cases by neighborhood, 2016–2019",
                   "Dados_Modelagem.xlsx, sheet 'Casos de Dengue'", _PAPER_LICENSE, _PAPER),
        SourceSpec("ovitrap", "Ovitrap indices (OPI, EDI) by neighborhood, 2016–2019",
                   "Rearranged_Data.mat, variable EggIndice_Agregated", _PAPER_LICENSE, _PAPER),
        SourceSpec("census", "Income and population by neighborhood, 2010 Census",
                   "DADOS GERAIS_01.xlsx, sheet 'Renda por Bairro'", _PAPER_LICENSE, _PAPER),
        SourceSpec("centroids", "Neighborhood centroids from IBGE 2010 boundaries",
                   "https://github.com/ipea/geobr_prep_data/releases/download/v2.0.0/neighborhoods_2010.parquet",
                   "IBGE public data; cite IBGE as the source",
                   "IBGE, Censo Demográfico 2010, malha de bairros; distributed by geobr (Ipea). "
                   "Centroids computed by scripts/build_reference_data.py; see data/reference/README.md."),
    ]
}
CENTROIDS_RETRIEVED = date(2026, 10, 16)


# =============================================================================
# Readers
# =============================================================================

def _read_reference() -> pd.DataFrame:
    ref = pd.read_csv(REFERENCE_CSV, dtype={"code": str, "ibge_code": str})
    ref["key"] = ref["name"].map(name_key)
    return ref


def _read_dengue() -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """One row per neighborhood, one column per epidemiological week (S201601…)."""
    df = pd.read_excel(DATA_RAW / "Dados_Modelagem.xlsx", sheet_name="Casos de Dengue", index_col=0)
    df.index = [str(n).strip() for n in df.index]
    return df, [epiweek.parse_label(c) for c in df.columns]


def _read_egg_indices() -> tuple[np.ndarray, np.ndarray]:
    """EggIndice_Agregated (36 rows × 208 weeks × 2): [:, :, 0] OPI, [:, :, 1] EDI."""
    egg = sio.loadmat(str(DATA_RAW / "Rearranged_Data.mat"), squeeze_me=True)["EggIndice_Agregated"]
    return egg[:, :, 0], egg[:, :, 1]


def _read_census() -> pd.DataFrame:
    """Columns located by header text, not position."""
    df = pd.read_excel(DATA_RAW / "DADOS GERAIS_01.xlsx", sheet_name="Renda por Bairro")

    def col(prefix: str) -> str:
        matches = [c for c in df.columns if str(c).strip().startswith(prefix)]
        if len(matches) != 1:
            raise SystemExit(f"Expected one column starting with {prefix!r} in 'Renda por Bairro', found {matches}")
        return matches[0]

    out = pd.DataFrame({
        "name":                  df[col("Bairros")].astype(str).str.strip(),
        "population":            pd.to_numeric(df[col("População")], errors="coerce"),
        "total_income_brl":      pd.to_numeric(df[col("Renda Mensal Total")], errors="coerce"),
        "income_per_capita_brl": pd.to_numeric(df[col("Renda Mensal por Pessoa")], errors="coerce"),
    })
    out["key"] = out["name"].map(name_key)
    return out


def _num(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


# =============================================================================
# Loaders (each one idempotent)
# =============================================================================

async def _upsert_sources(session: AsyncSession) -> dict[str, int]:
    ids = {}
    now = datetime.now(timezone.utc)
    for spec in SOURCES.values():
        row = (await session.execute(select(DataSource).where(DataSource.name == spec.name))).scalar_one_or_none()
        if row is None:
            row = DataSource(name=spec.name)
            session.add(row)
        row.file_or_url, row.license, row.citation, row.imported_at = (
            spec.file_or_url, spec.license, spec.citation, now)
        await session.flush()
        ids[spec.key] = row.id
    return ids


async def _upsert_weeks(session: AsyncSession, weeks: list[tuple[int, int]]) -> list[int]:
    existing = {(w.epi_year, w.epi_week): w.id for w in (await session.execute(select(EpiWeek))).scalars()}
    new = []
    for y, w in weeks:
        if (y, w) not in existing:
            start = epiweek.week_start(y, w)
            row = EpiWeek(epi_year=y, epi_week=w, start_date=start, end_date=start + timedelta(days=6))
            session.add(row)
            new.append(((y, w), row))
    await session.flush()
    existing.update({k: r.id for k, r in new})
    return [existing[w] for w in weeks]


async def _upsert_neighborhoods(session: AsyncSession, ref: pd.DataFrame, centroid_source: int) -> dict[str, int]:
    """Districts and neighborhoods from the reference file. Returns name_key -> neighborhood id."""
    districts = {d.name: d for d in (await session.execute(select(District))).scalars()}
    for name in sorted(ref["district"].unique()):
        if name not in districts:
            districts[name] = District(name=name)
            session.add(districts[name])
    await session.flush()

    by_key = {name_key(n.name): n for n in (await session.execute(select(Neighborhood))).scalars()}
    for r in ref.itertuples():
        n = by_key.get(r.key)
        if n is None:
            n = Neighborhood(name=r.name)
            session.add(n)
            by_key[r.key] = n
        n.name, n.code, n.district_id = r.name, r.code, districts[r.district].id
        n.latitude, n.longitude = float(r.latitude), float(r.longitude)
        n.coord_source_id, n.coord_retrieved_at = centroid_source, CENTROIDS_RETRIEVED
    await session.flush()

    extra = sorted(n.name for k, n in by_key.items() if k not in set(ref["key"]))
    if extra:
        log.warning("Neighborhoods in the database but not in the reference file (kept): %s", extra)
    return {k: n.id for k, n in by_key.items()}


async def _load_census(session: AsyncSession, census: pd.DataFrame, ids: dict[str, int], source: int) -> int:
    await session.execute(delete(Socioeconomic).where(Socioeconomic.reference_year == CENSUS_YEAR))
    unmatched = sorted(census.loc[~census["key"].isin(ids), "name"])
    if unmatched:
        log.warning("Census rows with no matching neighborhood (skipped): %s", unmatched)
    missing = sorted(k for k in ids if k not in set(census["key"]))
    if missing:
        log.warning("Neighborhoods with no census row: %s", missing)
    rows = [
        Socioeconomic(
            neighborhood_id       = ids[r.key],
            reference_year        = CENSUS_YEAR,
            population            = None if pd.isna(r.population) else int(r.population),
            total_income_brl      = _num(r.total_income_brl),
            income_per_capita_brl = _num(r.income_per_capita_brl),
            source_id             = source,
        )
        for r in census.itertuples() if r.key in ids
    ]
    session.add_all(rows)
    await session.flush()
    return len(rows)


async def _load_dengue(session: AsyncSession, df: pd.DataFrame, week_ids: list[int],
                       ids: dict[str, int], source: int) -> int:
    unmatched = [n for n in df.index if name_key(n) not in ids]
    if unmatched:
        raise SystemExit(f"Dengue sheet neighborhoods missing from the reference file: {unmatched}")
    await session.execute(delete(DengueCase).where(DengueCase.disease == "dengue"))
    rows = []
    for name, counts in zip(df.index, df.to_numpy()):
        nid = ids[name_key(name)]
        for wid, val in zip(week_ids, counts):
            rows.append(DengueCase(neighborhood_id=nid, epi_week_id=wid, disease="dengue",
                                   case_count=None if pd.isna(val) else int(val), source_id=source))
    session.add_all(rows)
    await session.flush()
    return len(rows)


async def _load_ovitrap(session: AsyncSession, opi: np.ndarray, edi: np.ndarray, week_ids: list[int],
                        ref: pd.DataFrame, ids: dict[str, int], source: int) -> int:
    """Each neighborhood reads its own row of the egg arrays (egg_row, 1-based)."""
    if opi.shape[1] != len(week_ids):
        raise SystemExit(f"Ovitrap data has {opi.shape[1]} weeks, dengue data has {len(week_ids)}")
    await session.execute(delete(OvitrapReading))
    rows = []
    for r in ref.itertuples():
        i = int(r.egg_row) - 1
        for wi, wid in enumerate(week_ids):
            rows.append(OvitrapReading(neighborhood_id=ids[r.key], epi_week_id=wid,
                                       opi=_num(opi[i, wi]), edi=_num(edi[i, wi]), source_id=source))
    session.add_all(rows)
    await session.flush()
    return len(rows)


# =============================================================================
# Entry point
# =============================================================================

async def _assert_ready() -> None:
    missing_files = [f for f in ("Dados_Modelagem.xlsx", "DADOS GERAIS_01.xlsx", "Rearranged_Data.mat")
                     if not (DATA_RAW / f).exists()]
    if missing_files:
        raise SystemExit(f"Missing files in {DATA_RAW}: {missing_files}")
    if not REFERENCE_CSV.exists():
        raise SystemExit(f"Missing {REFERENCE_CSV}; it is committed in the repository.")
    async with engine.connect() as conn:
        tables = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))
    if REQUIRED_TABLES - tables:
        raise SystemExit(f"Database schema missing tables {sorted(REQUIRED_TABLES - tables)}. "
                         "Run  alembic upgrade head  before the ETL.")


async def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    await _assert_ready()

    ref              = _read_reference()
    dengue_df, weeks = _read_dengue()
    opi, edi         = _read_egg_indices()
    census           = _read_census()

    try:
        async with AsyncSessionLocal() as session, session.begin():
            src      = await _upsert_sources(session)
            week_ids = await _upsert_weeks(session, weeks)
            ids      = await _upsert_neighborhoods(session, ref, src["centroids"])
            n_census = await _load_census(session, census, ids, src["census"])
            n_dengue = await _load_dengue(session, dengue_df, week_ids, ids, src["dengue"])
            n_ovi    = await _load_ovitrap(session, opi, edi, week_ids, ref, ids, src["ovitrap"])
            cleared  = (await session.execute(delete(Forecast))).rowcount
    finally:
        await engine.dispose()

    log.info("Sources: %d | weeks: %d (%s to %s) | neighborhoods: %d | districts: %d",
             len(src), len(weeks), epiweek.week_start(*weeks[0]),
             epiweek.week_start(*weeks[-1]) + timedelta(days=6), len(ids), ref["district"].nunique())
    log.info("Census rows: %d | dengue_case rows: %d | ovitrap_reading rows: %d | stored forecasts cleared: %d",
             n_census, n_dengue, n_ovi, cleared)


if __name__ == "__main__":
    asyncio.run(run())
