"""
utils.py
--------
Shared helpers for loading and preprocessing the dengue / EDI data.
Replaces the data-loading sections at the top of Run1DLSTM_01.m.
"""

import numpy as np
import scipy.io as sio
import h5py
from pathlib import Path


# ---------------------------------------------------------------------------
# .mat loaders
# ---------------------------------------------------------------------------

def load_mat_v73(path: str | Path) -> dict:
    """Load a MATLAB v7.3 (.mat) file via h5py and return a plain dict."""
    out = {}
    with h5py.File(path, "r") as hf:
        for key in hf.keys():
            if key.startswith("#"):
                continue
            obj = hf[key]
            if isinstance(obj, h5py.Dataset):
                out[key] = obj[()]          # load into numpy array
            # Groups (cell arrays etc.) are skipped for now
    return out


def load_mat_legacy(path: str | Path) -> dict:
    """Load a pre-v7.3 MATLAB .mat file via scipy.io."""
    return sio.loadmat(str(path), squeeze_me=True)


# ---------------------------------------------------------------------------
# EDI / Ovitrap data  (Rearranged_Data.mat)
# ---------------------------------------------------------------------------

def load_rearranged_data(path: str | Path) -> dict:
    """
    Load Rearranged_Data.mat and return the two EDI matrices.

    Returns
    -------
    dict with keys:
        'OPI'  : (208, 36) float array  – Ovitrap Positivity Index
        'EDI'  : (208, 36) float array  – Egg Density Index
    """
    mat = load_mat_legacy(path)
    # EggIndice_Agregated is (36 neighborhoods, 208 weeks, 2 indices)
    egg = mat["EggIndice_Agregated"]          # shape (36, 208, 2)
    return {
        "OPI": egg[:, :, 0].T,               # (208, 36)
        "EDI": egg[:, :, 1].T,               # (208, 36)
    }


# ---------------------------------------------------------------------------
# Evaluation results  (Models_1DLSTM_*_EvalResults.mat)
# ---------------------------------------------------------------------------

def load_eval_results(path: str | Path) -> dict:
    """
    Load a v7.3 eval-results .mat file.

    Returns
    -------
    dict with keys:
        'YPred' : (n_timesteps, n_reps, n_models) float array
        'YTest' : (n_timesteps,) float array
        'tTest' : (n_timesteps,) float array  – MATLAB datenums
    """
    raw = load_mat_v73(path)
    # HDF5 stores arrays transposed relative to MATLAB
    return {
        "YPred": raw["YPred"].T,             # (n_timesteps, n_reps, n_models)
        "YTest": raw["YTest"].squeeze(),
        "tTest": raw["tTest"].squeeze(),
    }


# ---------------------------------------------------------------------------
# Metrics  (replaces ComputeRMSE / ComputeR in ModelsPerform_1DLSTM_01.m)
# ---------------------------------------------------------------------------

def compute_rmse(y_pred_log2: np.ndarray, y_test: np.ndarray):
    """
    Parameters
    ----------
    y_pred_log2 : (n_timesteps, n_reps)  – log2-scale predictions
    y_test      : (n_timesteps,)

    Returns
    -------
    mean_rmse, std_err_rmse
    """
    n_reps = y_pred_log2.shape[1]
    y_pred = 2.0 ** y_pred_log2
    y_test_rep = np.tile(y_test[:, None], (1, n_reps))
    rmse_per_rep = np.sqrt(np.mean((y_test_rep - y_pred) ** 2, axis=0))
    return rmse_per_rep.mean(), rmse_per_rep.std() / np.sqrt(n_reps)


def compute_r(y_pred_log2: np.ndarray, y_test: np.ndarray):
    """
    Parameters
    ----------
    y_pred_log2 : (n_timesteps, n_reps)
    y_test      : (n_timesteps,)

    Returns
    -------
    mean_r, std_err_r
    """
    n_reps = y_pred_log2.shape[1]
    y_pred = 2.0 ** y_pred_log2
    r_vals = np.array([
        np.corrcoef(y_test, y_pred[:, i])[0, 1]
        for i in range(n_reps)
    ])
    return r_vals.mean(), r_vals.std() / np.sqrt(n_reps)


# ---------------------------------------------------------------------------
# MATLAB datenum  →  pandas/numpy datetime
# ---------------------------------------------------------------------------

def matlab_datenum_to_datetime(datenums: np.ndarray):
    """Convert MATLAB serial datenums to numpy datetime64[D]."""
    import pandas as pd
    # MATLAB epoch: Jan 0, 0000  →  offset vs Python epoch (Jan 1, 1970)
    # 719529 days between MATLAB epoch and Unix epoch
    days = datenums.astype(float) - 719529
    return pd.to_datetime(days, unit="D", origin="unix")


# ---------------------------------------------------------------------------
# Baselines and plain metrics (linear scale)
# ---------------------------------------------------------------------------

def rmse_and_r(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """RMSE and Pearson r of one prediction series, both in case units."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))), float(np.corrcoef(y_true, y_pred)[0, 1])


def persistence_baseline(series: np.ndarray, targets: np.ndarray, lead: int) -> np.ndarray:
    """
    Naive forecast: the value observed `lead` weeks before each target.
    With the same lead as a model's newest input, it uses the same
    information as that model, so it is the fair yardstick.
    """
    return np.asarray(series, dtype=float)[np.asarray(targets) - lead]
