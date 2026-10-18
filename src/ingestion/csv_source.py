"""
ingestion/csv_source.py
-----------------------
Weekly cases from a CSV file, e.g. a spreadsheet exported each week.

Required columns (header names, any order):
    neighborhood   official code or name
    epi_year       epidemiological year
    epi_week       epidemiological week (1–53)
    cases          notified cases (blank = unknown)
Optional:
    disease        defaults to 'dengue'
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.core import epiweek
from src.ingestion.base import WeeklyCases

REQUIRED = {"neighborhood", "epi_year", "epi_week", "cases"}


@dataclass
class CsvCasesSource:
    path:     Path
    name:     str
    license:  str | None = None
    citation: str | None = None

    @property
    def file_or_url(self) -> str:
        return str(self.path)

    def fetch(self) -> Iterable[WeeklyCases]:
        df = pd.read_csv(self.path, dtype={"neighborhood": str})
        df.columns = [c.strip().lower() for c in df.columns]
        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"{self.path}: missing columns {sorted(missing)}")
        for i, r in df.iterrows():
            year, week = int(r["epi_year"]), int(r["epi_week"])
            epiweek.week_start(year, week)          # rejects impossible weeks
            cases = None if pd.isna(r["cases"]) else int(r["cases"])
            if cases is not None and cases < 0:
                raise ValueError(f"{self.path}, row {i + 2}: negative case count")
            yield WeeklyCases(neighborhood=str(r["neighborhood"]), epi_year=year, epi_week=week, cases=cases,
                              disease=str(r.get("disease") or "dengue") if "disease" in df.columns else "dengue")
