"""
ml/inference.py
---------------
Forecasts from saved models. Nothing is trained here: models come from
`python -m src.ml.train` and are found through the model_run table.

    registry = ModelRegistry()
    await registry.refresh(session)                 # at startup (and after training)
    run = registry.get("edi", 6)                    # newest run for that configuration
    result = forecast(series, as_of_index, run, horizon=4)

How a forecast is made (same procedure as the paper's test period):
the LSTM reads the input window of every week from the first one with a
full window up to the target weeks, carrying its state forward, and the
outputs for the target weeks are the forecast. Only data observed up to the
as-of week is used, which limits the horizon to the configuration's newest
input lag (see preprocessing.max_horizon). Each saved repetition gives one
forecast; the result is their mean, with mean ± 1 standard deviation as the
band (spread between repetitions, not a calibrated interval).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.ml.data import CitySeries
from src.ml.preprocessing import (
    MissingDataError, back_transform, build_inputs, lag_offsets, log2_transform, max_horizon,
)

log = logging.getLogger("sipsa.inference")

ROOT = Path(__file__).resolve().parent.parent.parent


class ModelNotReadyError(RuntimeError):
    """No trained model is registered for the requested configuration."""


@dataclass(frozen=True)
class RunInfo:
    model_run_id:  int
    model_version: str
    input_type:    str
    lag:           int
    n_reps:        int
    artifact_path: Path

    @property
    def max_horizon(self) -> int:
        return max_horizon(self.lag)


class ModelRegistry:
    """
    Newest registered run per (input_type, lag), with its repetitions loaded
    on first use and kept in memory. Thread-safe: forecasts run in a
    threadpool.
    """

    def __init__(self) -> None:
        self._runs: dict[tuple[str, int], RunInfo] = {}
        self._models: dict[int, list] = {}
        self._lock = threading.Lock()

    async def refresh(self, session) -> None:
        from sqlalchemy import select
        from src.api.models.db_models import ModelRun

        rows = (await session.execute(
            select(ModelRun).order_by(ModelRun.created_at.desc(), ModelRun.id.desc())
        )).scalars().all()
        runs: dict[tuple[str, int], RunInfo] = {}
        for r in rows:
            key = (r.input_type, r.lag)
            path = ROOT / r.artifact_path
            if key in runs:
                continue
            if not path.is_dir() or not any(path.glob("rep_*.keras")):
                log.warning("model_run %s (%s lag %d): no saved models at %s; skipped",
                            r.model_version, r.input_type, r.lag, path)
                continue
            runs[key] = RunInfo(r.id, r.model_version, r.input_type, r.lag, r.n_reps, path)
        with self._lock:
            self._runs = runs
            self._models = {k: v for k, v in self._models.items() if k in {i.model_run_id for i in runs.values()}}

    @property
    def available(self) -> list[RunInfo]:
        return sorted(self._runs.values(), key=lambda r: (r.input_type, r.lag))

    def get(self, input_type: str, lag: int) -> RunInfo:
        run = self._runs.get((input_type, lag))
        if run is None:
            raise ModelNotReadyError(
                f"No trained model for input '{input_type}' with lag {lag}. "
                "Train with  python -m src.ml.train  (it registers the models), then restart the API."
            )
        return run

    def models(self, run: RunInfo) -> list:
        """The run's repetitions, loaded once (a few seconds on first use)."""
        with self._lock:
            if run.model_run_id not in self._models:
                from src.ml.model import load
                files = sorted(run.artifact_path.glob("rep_*.keras"))
                log.info("Loading %d repetitions of %s (%s lag %d)", len(files), run.model_version,
                         run.input_type, run.lag)
                self._models[run.model_run_id] = [load(f) for f in files]
            return self._models[run.model_run_id]


def forecast(series: CitySeries, as_of_index: int, run: RunInfo, horizon: int,
             models: list) -> dict[str, np.ndarray]:
    """
    Forecast the `horizon` weeks after week `as_of_index` (0-based index into
    series), in the model's unit: mean cases per neighborhood.
    Returns mean, lower, upper (each of length `horizon`) and the per-rep values.
    """
    from src.ml.model import predict_sequence

    if not 1 <= horizon <= run.max_horizon:
        raise ValueError(f"horizon must be between 1 and {run.max_horizon} for lag {run.lag}")
    first_target = max(lag_offsets(run.lag))
    if as_of_index < first_target:
        raise MissingDataError(f"at least {first_target + 1} observed weeks are needed before forecasting")

    observed = series.series(run.input_type)[: as_of_index + 1]       # nothing after the as-of week
    x = log2_transform(observed, run.input_type)
    targets = np.arange(first_target, as_of_index + horizon + 1)
    inputs = build_inputs(x, targets, run.lag)                         # every input is <= as_of_index

    per_rep = np.stack([back_transform(predict_sequence(m, inputs)[-horizon:]) for m in models])
    mean, std = per_rep.mean(axis=0), per_rep.std(axis=0)
    return {"mean": mean, "lower": np.maximum(mean - std, 0.0), "upper": mean + std, "per_rep": per_rep}


registry = ModelRegistry()
