"""
ml/preprocessing.py
-------------------
Transformations and lag windows shared by training (train.py) and
inference (inference.py), so both always build inputs the same way.

Translated from Run1DLSTM_01.m:

    NTrain      = round(.8*NSamples)
    IndTRainOut = iBack+1:NTrain                 % training targets
    IndTestOut  = NTrain+1:NSamples              % test targets
    vector      = iBack            (iBack == 1)
                = (iBack-2):iBack  (otherwise)   % input offsets
    IndIn(cnt,:) = IndOut - vector(cnt)
    x1 = log2(Dengue) or log2(EDI);  y1 = log2(Dengue)

Indices here are 0-based.
"""

from __future__ import annotations

import numpy as np

SUPPORTED_LAGS = (1, 3, 4, 5, 6)
INPUT_TYPES = ("dengue", "edi")


class MissingDataError(ValueError):
    """The series has gaps or values the model cannot use."""


def lag_offsets(lag: int) -> list[int]:
    """
    How many weeks before the target week each input is taken.
    lag 1 -> [1];  lag 3 -> [1, 2, 3];  lag 6 -> [4, 5, 6].
    Order matches MATLAB's feature rows: (iBack-2):iBack.
    """
    if lag not in SUPPORTED_LAGS:
        raise ValueError(f"lag must be one of {SUPPORTED_LAGS}, got {lag}")
    return [1] if lag == 1 else [lag - 2, lag - 1, lag]


def max_horizon(lag: int) -> int:
    """
    Furthest week ahead this configuration can forecast from observed data only:
    the target week t needs inputs at t - offset for every offset, all of
    which must already be observed.
    """
    return min(lag_offsets(lag))


def n_train_samples(n_samples: int, train_split: float = 0.8) -> int:
    """MATLAB round(): halves away from zero (Python's round() rounds to even)."""
    return int(np.floor(train_split * n_samples + 0.5))


def log2_transform(values: np.ndarray, name: str = "series") -> np.ndarray:
    """
    log2 as in the paper. Zero or negative values have no log; the city-wide
    means used by the paper are always positive (minimum 0.67 cases).
    """
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        raise MissingDataError(f"{name} has missing weeks; fill the data before modelling")
    if (values <= 0).any():
        raise MissingDataError(f"{name} has values <= 0, which log2 cannot transform")
    return np.log2(values)


def back_transform(log2_values: np.ndarray) -> np.ndarray:
    return np.power(2.0, log2_values)


def build_inputs(x: np.ndarray, targets: np.ndarray, lag: int) -> np.ndarray:
    """
    Input matrix (n_features, n_targets): column j holds x at
    targets[j] - offset for each offset of the lag window.
    """
    targets = np.asarray(targets)
    offsets = lag_offsets(lag)
    if (targets - max(offsets) < 0).any():
        raise ValueError("a target week has no full input window before it")
    return np.stack([x[targets - o] for o in offsets])


def split_targets(n_samples: int, lag: int, train_split: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """(training targets, test targets) as 0-based week indices."""
    n_train = n_train_samples(n_samples, train_split)
    return np.arange(lag, n_train), np.arange(n_train, n_samples)
