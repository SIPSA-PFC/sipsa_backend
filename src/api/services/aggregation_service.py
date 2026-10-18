"""
api/services/aggregation_service.py
Builds city-wide numpy arrays for the LSTM from DB rows.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.api.repositories.dengue_repository import DengueRepository
from src.api.repositories.ovitrap_repository import OvitrapRepository


class AggregationService:
    def __init__(self, dengue_repo: DengueRepository, ovitrap_repo: OvitrapRepository):
        self.dengue_repo  = dengue_repo
        self.ovitrap_repo = ovitrap_repo

    async def get_aligned_series(self) -> dict:
        """
        Returns city-wide dengue and EDI series aligned on the same ISO week index.

        Keys: dengue (ndarray), edi (ndarray), epi_years, epi_weeks, week_starts
        """
        dengue_rows = await self.dengue_repo.get_city_wide_mean()
        edi_rows    = await self.ovitrap_repo.get_city_wide_edi_mean()

        if not dengue_rows:
            raise ValueError("No dengue data in database. Run the ETL first.")

        dengue_s = pd.Series(
            {(r["epi_year"], r["epi_week"]): r["mean_cases"] for r in dengue_rows}
        ).sort_index()

        edi_s = pd.Series(
            {(r["epi_year"], r["epi_week"]): r["mean_edi"] for r in edi_rows}
        ).sort_index()

        df = pd.DataFrame({"dengue": dengue_s, "edi": edi_s})
        # Weeks without EDI stay NaN. Filling them with the series mean (as
        # before) used values from future weeks; the model code now refuses
        # missing values with a clear error instead.

        date_map = {(r["epi_year"], r["epi_week"]): r["week_start"] for r in dengue_rows}

        return {
            "dengue":      df["dengue"].to_numpy(dtype=float),
            "edi":         df["edi"].to_numpy(dtype=float),
            "epi_years":   [idx[0] for idx in df.index],
            "epi_weeks":   [idx[1] for idx in df.index],
            "week_starts": [date_map.get((y, w)) for y, w in df.index],
        }
