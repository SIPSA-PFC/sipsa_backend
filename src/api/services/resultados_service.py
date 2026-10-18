"""
api/services/resultados_service.py
------------------------------------
Reads pre-computed MATLAB eval results and returns all metrics and
time series the frontend needs — no database, no training required.

The .mat file contains:
    YPred : (42 weeks, 30 reps, 10 models)  log2-scale predictions
    YTest : (42,)                            real-scale actual values
    tTest : (42,)                            MATLAB datenums

Model index mapping (matches Run1DLSTM_01.m):
    0–4  → D→D  lags [1, 3, 4, 5, 6]
    5–9  → O→D  lags [1, 3, 4, 5, 6]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ml.utils import load_eval_results, compute_rmse, compute_r, matlab_datenum_to_datetime
from src.api.models.schemas import (
    ResultadosOut, ModelMetricsOut, BestModelSeriesOut, TimeSeriesPointOut,
)

# Lag configurations matching MATLAB's NBackSamples = [1 3:6]
N_BACK_SAMPLES = [1, 3, 4, 5, 6]

LAG_LABELS = {
    1: "1 past sample",
    3: "3:1 past samples",
    4: "4:2 past samples",
    5: "5:3 past samples",
    6: "6:4 past samples",
}


def _risk_from_rmse(rmse: float) -> str:
    """Map RMSE to a risk-quality label for the frontend gauge."""
    if rmse < 4:   return "low"       # good model
    if rmse < 7:   return "moderate"
    return "high"                     # poor model


# Parsed results per file, reused until the file changes (path, mtime).
_CACHE: dict[tuple[str, float], ResultadosOut] = {}


class ResultadosService:

    def __init__(self, mat_path: str | Path):
        self.mat_path = Path(mat_path)

    def _assert_available(self) -> None:
        if not self.mat_path.exists():
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Eval results file not found at {self.mat_path}. "
                    "Place Models_1DLSTM_2022_02_04_EvalResults.mat in data/raw/."
                )
            )

    def get_resultados(self) -> ResultadosOut:
        self._assert_available()
        key = (str(self.mat_path), self.mat_path.stat().st_mtime)
        if key not in _CACHE:
            _CACHE.clear()
            _CACHE[key] = self._compute()
        return _CACHE[key]

    def _compute(self) -> ResultadosOut:

        res   = load_eval_results(self.mat_path)
        YPred = res["YPred"]    # (42, 30, 10)
        YTest = res["YTest"]    # (42,)
        tTest = res["tTest"]    # (42,) MATLAB datenums

        n_models    = YPred.shape[2]
        n_half      = len(N_BACK_SAMPLES)   # 5
        dates       = matlab_datenum_to_datetime(tTest)
        date_strs   = [d.strftime("%Y-%m-%d") for d in dates]

        # ── Compute metrics for all 10 models ─────────────────────────────────
        metrics: list[ModelMetricsOut] = []

        for i in range(n_models):
            input_type = "dengue" if i < n_half else "edi"
            lag        = N_BACK_SAMPLES[i % n_half]
            mean_rmse, std_rmse = compute_rmse(YPred[:, :, i], YTest)
            mean_r,    std_r    = compute_r(   YPred[:, :, i], YTest)

            metrics.append(ModelMetricsOut(
                model_index  = i,
                input_type   = input_type,
                lag_label    = LAG_LABELS[lag],
                mean_rmse    = round(float(mean_rmse), 4),
                std_err_rmse = round(float(std_rmse),  4),
                mean_r       = round(float(mean_r),    4),
                std_err_r    = round(float(std_r),     4),
                risk_level   = _risk_from_rmse(mean_rmse),
            ))

        # ── Build time series for the two best models ─────────────────────────
        # Best dengue-input (D→D) and best EDI-input (O→D) model by mean r,
        # computed rather than hard-coded (for the paper's file: models 0 and 9).
        best_indices = [
            max(range(n_half), key=lambda i: metrics[i].mean_r),
            max(range(n_half, n_models), key=lambda i: metrics[i].mean_r),
        ]
        best_models: list[BestModelSeriesOut] = []

        for idx in best_indices:
            m          = metrics[idx]
            y_pred_lin = 2.0 ** YPred[:, :, idx]          # back-transform
            pred_mean  = y_pred_lin.mean(axis=1)
            pred_std   = y_pred_lin.std(axis=1)

            series = [
                TimeSeriesPointOut(
                    week_start     = date_strs[t],
                    actual         = round(float(YTest[t]),      4),
                    predicted_mean = round(float(pred_mean[t]),  4),
                    lower_bound    = round(max(float(pred_mean[t] - pred_std[t]), 0), 4),
                    upper_bound    = round(float(pred_mean[t] + pred_std[t]),     4),
                )
                for t in range(len(date_strs))
            ]

            best_models.append(BestModelSeriesOut(
                model_index = idx,
                input_type  = m.input_type,
                lag_label   = m.lag_label,
                mean_r      = m.mean_r,
                mean_rmse   = m.mean_rmse,
                series      = series,
            ))

        return ResultadosOut(
            metrics            = metrics,
            best_models        = best_models,
            best_r             = max(m.mean_r    for m in metrics),
            best_rmse          = min(m.mean_rmse for m in metrics),
            test_period_start  = date_strs[0],
            test_period_end    = date_strs[-1],
            n_reps             = YPred.shape[1],
        )