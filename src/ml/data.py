"""
ml/data.py
----------
City-wide weekly series for modelling, from the raw files or the database.

The paper models the city as a whole: for each epidemiological week, the
mean over the 36 neighborhoods of dengue cases (MATLAB: nanmean(DengueOcc, 2))
and of the egg density index (nanmean(ODI_Mat, 2)).

    python -m src.ml.data        # checks that both sources give the same series
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from src.core import epiweek

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"


@dataclass(frozen=True)
class CitySeries:
    dengue: np.ndarray                 # mean cases per neighborhood, per week
    edi:    np.ndarray                 # mean egg density index, per week
    weeks:  list[tuple[int, int]]      # (epi_year, epi_week)

    @property
    def week_starts(self) -> list[date]:
        return [epiweek.week_start(y, w) for y, w in self.weeks]

    def series(self, input_type: str) -> np.ndarray:
        return self.dengue if input_type == "dengue" else self.edi


def load_from_raw(data_raw: Path = DATA_RAW) -> CitySeries:
    """Exactly what Run1DLSTM_01.m computes from the same files."""
    df = pd.read_excel(data_raw / "Dados_Modelagem.xlsx", sheet_name="Casos de Dengue", index_col=0)
    weeks = [epiweek.parse_label(c) for c in df.columns]
    dengue = np.nanmean(df.to_numpy(dtype=float), axis=0)
    egg = sio.loadmat(str(data_raw / "Rearranged_Data.mat"), squeeze_me=True)["EggIndice_Agregated"]
    edi = np.nanmean(egg[:, :, 1], axis=0)      # row order does not affect the mean
    if len(edi) != len(weeks):
        raise ValueError(f"EDI has {len(edi)} weeks, dengue has {len(weeks)}")
    return CitySeries(dengue=dengue, edi=edi, weeks=weeks)


async def load_from_db(session) -> CitySeries:
    """
    Same series from the database. Weeks without EDI are kept as NaN (no
    filling: filling with the series mean would use future weeks).
    """
    from src.api.repositories.dengue_repository import DengueRepository
    from src.api.repositories.ovitrap_repository import OvitrapRepository

    dengue_rows = await DengueRepository(session).get_city_wide_mean()
    edi_rows    = await OvitrapRepository(session).get_city_wide_edi_mean()
    if not dengue_rows:
        raise ValueError("No dengue data in the database. Run the ETL first.")
    edi_by_week = {(r["epi_year"], r["epi_week"]): r["mean_edi"] for r in edi_rows}
    weeks = [(r["epi_year"], r["epi_week"]) for r in dengue_rows]
    return CitySeries(
        dengue=np.array([r["mean_cases"] for r in dengue_rows], dtype=float),
        edi=np.array([edi_by_week.get(w, np.nan) for w in weeks], dtype=float),
        weeks=weeks,
    )


def compare(a: CitySeries, b: CitySeries) -> list[str]:
    """Differences between two series (empty list = identical)."""
    problems = []
    if a.weeks != b.weeks:
        problems.append(f"weeks differ ({len(a.weeks)} vs {len(b.weeks)})")
    else:
        for name in ("dengue", "edi"):
            x, y = getattr(a, name), getattr(b, name)
            if not np.allclose(x, y, equal_nan=True, rtol=1e-9, atol=1e-9):
                worst = int(np.nanargmax(np.abs(x - y)))
                problems.append(f"{name} differs, largest gap at {a.weeks[worst]}: {x[worst]} vs {y[worst]}")
    return problems


async def _check() -> None:
    from src.core.database import AsyncSessionLocal, engine
    raw = load_from_raw()
    try:
        async with AsyncSessionLocal() as session:
            db = await load_from_db(session)
    finally:
        await engine.dispose()
    problems = compare(raw, db)
    print(f"{len(raw.weeks)} weeks, {raw.weeks[0]} to {raw.weeks[-1]}")
    print("Raw files and database give identical series." if not problems else "\n".join(problems))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(_check())
