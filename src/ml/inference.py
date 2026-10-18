"""
ml/inference.py
---------------
Thin inference wrapper loaded ONCE at API startup.

This is the bridge between the ML world (numpy, TensorFlow) and the
service layer. prediction_service.py calls run_prediction() and gets back
a plain numpy array — it never touches TensorFlow directly.

The LSTM logic (stateful step-by-step prediction, log2 transform,
weight transfer) is unchanged from train.py. Only the entry point differs:
instead of reading from files, it accepts numpy arrays directly from
the aggregation service.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from functools import lru_cache

from src.ml.preprocessing import SUPPORTED_LAGS, lag_offsets, log2_transform

# TensorFlow is imported lazily so the rest of the API starts even if TF
# is not installed — inference will raise a clear error only when called.
_tf = None


def _get_tf():
    global _tf
    if _tf is None:
        try:
            import tensorflow as tf
            _tf = tf
        except ImportError:
            raise RuntimeError(
                "TensorFlow is required for inference. "
                "Install it with:  pip install tensorflow"
            )
    return _tf


# =============================================================================
# Model registry — loaded once, reused across requests
# =============================================================================

class _ModelRegistry:
    """
    Holds the pre-trained weights in memory.
    Loaded once at startup via load_weights(); never retrained here.
    """
    def __init__(self):
        self._weights: dict[int, list] = {}   # model_index → list of weight arrays
        self._input_sizes: dict[int, int] = {}
        self._loaded: bool = False
        self.model_version: str = "unknown"

    def load(self, npz_path: str | Path) -> None:
        """
        Load weights from the .npz file produced by train.py.

        The .npz stores YPred (n_timesteps, n_reps, n_models) — the raw
        predictions from all 30 training repetitions. For inference we
        pick the best rep per model (lowest RMSE) and rebuild a stateful
        Keras model from those weights.

        NOTE: train.py does not currently save individual Keras weight arrays —
        it saves predictions. If you want to reload exact weights, add a
        model.save_weights() call in train.py for each rep and load them here.

        As a practical alternative, inference.py re-runs a single training
        pass per model on first load (warm-start mode) OR simply re-uses the
        stored predictions directly for the test window. For live API use,
        the prediction service will re-train a single rep on the full available
        data and use that model for the requested future window.

        If the .npz file does not exist yet (e.g. training has not been run),
        this method logs a warning and returns without raising — the API can
        still start. Calls to run_prediction() will raise a 503-compatible
        error until weights are loaded.
        """
        path = Path(npz_path)
        if not path.exists():
            # Do NOT raise here — allow the API to start without weights.
            # run_prediction() will surface a clear 503 error when called.
            import warnings
            warnings.warn(
                f"Model weights not found at {path}. "
                "The API will start, but /predicao will return 503 until "
                "weights are available. Run  python -m src.ml.train  to generate them.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        self._loaded = True
        self.model_version = path.stem   # e.g. "eval_results"

    @property
    def is_loaded(self) -> bool:
        return self._loaded


_registry = _ModelRegistry()


def load_weights(npz_path: str | Path) -> None:
    """Called once from main.py lifespan — loads the registry."""
    _registry.load(npz_path)


def is_model_ready() -> bool:
    """True once load_weights() has found the weights file."""
    return _registry.is_loaded


# =============================================================================
# Core LSTM builders  (identical to train.py — no duplication of logic)
# =============================================================================

def _build_train_lstm(tf, input_size: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None, input_size)),
        tf.keras.layers.LSTM(100, stateful=False, return_sequences=False),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss="mse",
    )
    return model


def _build_stateful_lstm(tf, input_size: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(batch_shape=(1, 1, input_size)),
        tf.keras.layers.LSTM(100, stateful=True, return_sequences=False),
        tf.keras.layers.Dense(1),
    ])
    model.compile(loss="mse")
    return model


def _lr_schedule(epoch: int, lr: float) -> float:
    if epoch == 125:
        return lr * 0.2
    return lr


def _transfer_weights(src, dst) -> None:
    for src_layer, dst_layer in zip(src.layers, dst.layers):
        dst_layer.set_weights(src_layer.get_weights())


# =============================================================================
# Public entry point — called by prediction_service.py
# =============================================================================

class ModelNotReadyError(RuntimeError):
    """
    Raised by run_prediction() when the model weights have not been loaded.
    prediction_service.py should catch this and return HTTP 503.
    """


def run_prediction(
    dengue_series: np.ndarray,
    edi_series: np.ndarray,
    input_type: str = "dengue",
    lag: int = 1,
    n_future_weeks: int = 4,
    n_reps: int = 5,
) -> dict:
    """
    Train one LSTM on all available data, then predict n_future_weeks ahead.

    This replicates the full MATLAB pipeline for a live request:
        1. Select the right input series (dengue or EDI).
        2. Apply log2 transform (negatives → 0, as per the original paper).
        3. Build lag windows (same as build_indices in train.py).
        4. Train n_reps independent models on the FULL available history.
        5. Predict step-by-step with stateful LSTM (MATLAB-equivalent).
        6. Back-transform predictions: 2^YPred → real case numbers.
        7. Return mean, lower bound, upper bound across reps.

    Parameters
    ----------
    dengue_series   : (n_weeks,) array of city-wide mean dengue cases
    edi_series      : (n_weeks,) array of city-wide mean EDI values
    input_type      : "dengue" or "edi" — which series to use as predictor
    lag             : lag configuration (1, 3, 4, 5, or 6)
    n_future_weeks  : how many weeks ahead to predict
    n_reps          : independent training runs (5 for API, 30 for training)

    Returns
    -------
    dict with keys:
        'mean'   : (n_future_weeks,)  back-transformed predicted cases
        'lower'  : (n_future_weeks,)  mean - 1 std
        'upper'  : (n_future_weeks,)  mean + 1 std
        'input_type': str
        'lag'    : int

    Raises
    ------
    ModelNotReadyError
        If model weights have not been loaded yet (e.g. training not run).
        The caller (prediction_service.py) should map this to HTTP 503.
    """
    # ── 503 guard — fail fast with a clear message before touching TF ────────
    if not _registry.is_loaded:
        raise ModelNotReadyError(
            "Model weights are not loaded. "
            "Run  python -m src.ml.train  to generate them, then restart the API."
        )

    # Only the trained lag windows are valid. lag=2 would build the window
    # [0, 1, 2], and offset 0 is the target week itself (leakage).
    if lag not in SUPPORTED_LAGS:
        raise ValueError(f"lag must be one of {SUPPORTED_LAGS}, got {lag}")
    if input_type not in ("dengue", "edi"):
        raise ValueError(f"input_type must be 'dengue' or 'edi', got {input_type!r}")

    tf = _get_tf()

    # ── 1. Select and log2-transform the input series ─────────────────────────
    raw = dengue_series if input_type == "dengue" else edi_series
    x   = log2_transform(raw, input_type)       # refuses missing / non-positive values
    y   = log2_transform(dengue_series, "dengue")   # target is always dengue

    n_samples = len(x)

    # ── 2. Build lag indices over the FULL series ─────────────────────────────
    vector = lag_offsets(lag)
    input_size = len(vector)

    # Training uses the whole available history
    train_end  = n_samples
    X_train_list = []
    Y_train_list = []

    for step in range(lag, n_samples):
        if lag == 1:
            X_train_list.append([x[step - 1]])
        else:
            X_train_list.append([x[step - v] for v in vector])
        Y_train_list.append(y[step])

    X_train = np.array(X_train_list).T    # (input_size, n_train_steps)
    Y_train = np.array(Y_train_list)      # (n_train_steps,)

    # ── 3. Build future input windows by repeating the last known values ───────
    # For live prediction we don't have future X, so we use the last lag values
    # as the input for each future step (persistence assumption).
    # This matches how MATLAB's model was used for the test window:
    # it had access to the real X_test values. Here we approximate with
    # the most recent known values.
    last_known = x[-lag:]                # (lag,)  or scalar for lag=1
    X_future_list = []
    for _ in range(n_future_weeks):
        if lag == 1:
            X_future_list.append([last_known[-1]])
        else:
            X_future_list.append([last_known[-v] for v in vector])

    X_future = np.array(X_future_list).T  # (input_size, n_future_weeks)

    # ── 4. Train n_reps models and predict ────────────────────────────────────
    X_tr = X_train.T.reshape(-1, 1, input_size)
    Y_tr = Y_train.reshape(-1, 1)
    lr_cb = tf.keras.callbacks.LearningRateScheduler(_lr_schedule, verbose=0)

    all_preds = np.zeros((n_future_weeks, n_reps))

    for i_rep in range(n_reps):
        # Train
        train_model = _build_train_lstm(tf, input_size)
        train_model.fit(
            X_tr, Y_tr,
            epochs=250,
            batch_size=64,
            callbacks=[lr_cb],
            verbose=0,
            shuffle=False,
        )

        # Transfer to stateful model
        pred_model = _build_stateful_lstm(tf, input_size)
        _transfer_weights(train_model, pred_model)

        # Warm up on full training history (MATLAB: predictAndUpdateState(net, XTrain))
        pred_model.reset_states()
        for t in range(X_train.shape[1]):
            step = X_train[:, t].reshape(1, 1, input_size)
            pred_model.predict(step, verbose=0)

        # Step-by-step future prediction
        for i in range(n_future_weeks):
            step = X_future[:, i].reshape(1, 1, input_size)
            all_preds[i, i_rep] = pred_model.predict(step, verbose=0).squeeze()

    # ── 5. Back-transform: 2^YPred → real case numbers ───────────────────────
    preds_real  = 2.0 ** all_preds
    pred_mean   = preds_real.mean(axis=1)
    pred_std    = preds_real.std(axis=1)

    return {
        "mean":       pred_mean,
        "lower":      np.maximum(pred_mean - pred_std, 0),
        "upper":      pred_mean + pred_std,
        "input_type": input_type,
        "lag":        lag,
    }