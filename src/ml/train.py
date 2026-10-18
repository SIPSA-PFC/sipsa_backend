"""
train.py
--------
Python equivalent of Run1DLSTM_01.m.

Trains 1D-LSTM models to forecast aggregate dengue incidence for the
municipality of Natal, RN-Brazil. Two predictor types are tested
(past dengue incidence and past EDI), each with five lag configurations.

Usage
-----
    python -m src.ml.train

Outputs
-------
    data/models/eval_results.npz
        YPred  : (n_timesteps, n_reps, n_models)
        YTest  : (n_timesteps,)
        tTest  : (n_timesteps,)  – as MATLAB datenums (float)
"""

from __future__ import annotations

import time
import random
from pathlib import Path

import numpy as np
import scipy.io as sio

# ── optional: set seeds for reproducibility ──────────────────────────────────
SEED = 1
random.seed(SEED)
np.random.seed(SEED)

try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not found – install it with:  pip install tensorflow")
    raise

# ── project paths ─────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent.parent   # repo root (src/ml/ -> src/ -> root)
DATA_RAW    = ROOT / "data" / "raw"
DATA_MODELS = ROOT / "data" / "models"
DATA_MODELS.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data loading
# =============================================================================

def load_data(rearranged_path: Path, dengue_xlsx: Path) -> dict:
    """
    Load EDI and dengue incidence data.

    NOTE: The dengue xlsx file (Dados_Modelagem.xlsx, sheet 4) is read when
    available.  If it is absent the script falls back to the EDI data only
    and raises a clear error – the xlsx is required for training.
    """
    # --- EDI / Ovitrap (Rearranged_Data.mat) --------------------------------
    mat = sio.loadmat(str(rearranged_path), squeeze_me=True)
    egg = mat["EggIndice_Agregated"]           # (36, 208, 2)
    EDI_mat = egg[:, :, 1].T                   # (208, 36)  EDI per neighborhood

    # --- Dengue incidence (xlsx) --------------------------------------------
    try:
        import pandas as pd
        df = pd.read_excel(str(dengue_xlsx), sheet_name=3,
                           index_col=0, header=0)
        dengue_mat = df.values.astype(float).T  # (weeks, neighborhoods)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dengue data not found at {dengue_xlsx}.\n"
            "Please copy 'Dados_Modelagem.xlsx' into data/raw/"
        )

    # --- Aggregate over neighborhoods (nanmean) ------------------------------
    Dengue = np.nanmean(dengue_mat, axis=1)    # (208,)
    EDI    = np.nanmean(EDI_mat,    axis=1)    # (208,)

    return {"Dengue": Dengue, "EDI": EDI}


# =============================================================================
# Index helpers  (direct translation of the MATLAB indexing logic)
# =============================================================================

def build_indices(n_samples: int, n_train: int, i_back: int):
    """
    Build training and test index arrays for a given lag configuration.

    Parameters
    ----------
    n_samples : total number of time steps
    n_train   : number of training samples
    i_back    : lag depth (1 → single lag; 3-6 → window of 3 lags)

    Returns
    -------
    IndTrainOut, IndTrainIn, IndTestOut, IndTestIn  – all 0-based
    """
    IndTestOut = np.arange(n_train, n_samples)   # 0-based

    if i_back == 1:
        vector = [i_back]
        IndTrainIn = np.empty((1, n_train - i_back), dtype=int)
        IndTestIn  = np.empty((1, len(IndTestOut)), dtype=int)
    else:
        vector = list(range(i_back - 2, i_back + 1))   # [iBack-2, iBack-1, iBack]
        IndTrainIn = np.empty((3, n_train - i_back), dtype=int)
        IndTestIn  = np.empty((3, len(IndTestOut)), dtype=int)

    IndTrainOut = np.arange(i_back, n_train)     # 0-based

    for cnt, ii_back in enumerate(vector):
        IndTrainIn[cnt, :] = IndTrainOut - ii_back
        IndTestIn[cnt, :]  = IndTestOut  - ii_back

    return IndTrainOut, IndTrainIn, IndTestOut, IndTestIn


# =============================================================================
# LSTM model  (replaces TrainLSTM nested function in MATLAB)
# =============================================================================

def build_train_lstm(input_size: int) -> tf.keras.Model:
    """
    Non-stateful LSTM used during training.
    Accepts full sequences so Keras can run backprop efficiently.

    Architecture matches MATLAB:
        sequenceInputLayer → lstmLayer(100) → fullyConnectedLayer(1)
    """
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


def build_stateful_lstm(input_size: int) -> tf.keras.Model:
    """
    Stateful LSTM used during step-by-step test prediction.

    batch_size=1 because we feed one time step at a time, exactly matching
    MATLAB's predictAndUpdateState behaviour where the hidden state (h, c)
    carries forward between consecutive test steps.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(batch_shape=(1, 1, input_size)),
        tf.keras.layers.LSTM(100, stateful=True, return_sequences=False),
        tf.keras.layers.Dense(1),
    ])
    model.compile(loss="mse")
    return model


def lr_schedule(epoch: int, lr: float) -> float:
    """
    Piecewise LR drop matching MATLAB's LearnRateSchedule:'piecewise':
        epoch 0-124  → 0.005
        epoch 125+   → 0.005 × 0.2 = 0.001
    """
    if epoch == 125:
        return lr * 0.2
    return lr


def transfer_weights(src: tf.keras.Model, dst: tf.keras.Model) -> None:
    """Copy trained weights from the non-stateful model into the stateful one."""
    for src_layer, dst_layer in zip(src.layers, dst.layers):
        dst_layer.set_weights(src_layer.get_weights())


def train_lstm(X_train: np.ndarray, Y_train: np.ndarray,
               X_test: np.ndarray, n_reps: int) -> np.ndarray:
    """
    Train the LSTM n_reps times and collect predictions using stateful
    step-by-step inference — matching MATLAB's predictAndUpdateState logic.

    Training phase
    --------------
    Uses a standard non-stateful LSTM fed the full training sequence at once
    (efficient for backprop).

    After training, the MATLAB script does:
        netLSTM = resetState(netLSTM)
        netLSTM = predictAndUpdateState(netLSTM, XTrain)  ← warm up on training data
        for i = 1:numTimeStepsTest
            [netLSTM, YPred(i)] = predictAndUpdateState(netLSTM, XTest(:,i))

    We replicate this exactly:
        1. Reset hidden state to zeros.
        2. Run all training steps through the stateful model to build up
           the same hidden state the MATLAB model had at the end of training.
        3. Step through each test sample one at a time, carrying (h, c) forward.

    Parameters
    ----------
    X_train : (input_size, n_train_steps)
    Y_train : (n_train_steps,)  – not used in inference, kept for API symmetry
    X_test  : (input_size, n_test_steps)
    n_reps  : number of independent runs

    Returns
    -------
    Y_pred  : (n_test_steps, n_reps)
    """
    input_size   = X_train.shape[0]
    n_test_steps = X_test.shape[1]

    # Shape for non-stateful training: (n_train_steps, 1, input_size)
    # Each time step is presented as a sequence of length 1
    X_tr = X_train.T.reshape(-1, 1, input_size)
    Y_tr = Y_train.reshape(-1, 1)

    Y_pred = np.zeros((n_test_steps, n_reps))
    lr_cb  = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0)

    for i_rep in range(n_reps):
        print(f"  Rep {i_rep + 1}/{n_reps}", end="\r")

        # ── 1. Train on full training sequence (non-stateful, efficient) ──────
        train_model = build_train_lstm(input_size)
        train_model.fit(
            X_tr, Y_tr,
            epochs=250,
            batch_size=64,
            callbacks=[lr_cb],
            verbose=0,
            shuffle=False,    # must preserve temporal order
        )

        # ── 2. Transfer weights to stateful model ─────────────────────────────
        pred_model = build_stateful_lstm(input_size)
        transfer_weights(train_model, pred_model)

        # ── 3. Warm up: run training data through stateful model to replicate
        #       MATLAB's  predictAndUpdateState(netLSTM, XTrain)
        #       This builds up the same (h, c) state as end-of-training. ──────
        pred_model.reset_states()
        for t in range(X_train.shape[1]):
            step = X_train[:, t].reshape(1, 1, input_size)
            pred_model.predict(step, verbose=0)   # discard output, keep state

        # ── 4. Step-by-step test prediction, carrying state forward ───────────
        #       Matches MATLAB's inner loop exactly:
        #           [netLSTM, YPred(i)] = predictAndUpdateState(netLSTM, XTest(:,i))
        for i in range(n_test_steps):
            step = X_test[:, i].reshape(1, 1, input_size)
            Y_pred[i, i_rep] = pred_model.predict(step, verbose=0).squeeze()

    print()
    return Y_pred


# =============================================================================
# Main training loop
# =============================================================================

def main():
    # ── paths ─────────────────────────────────────────────────────────────────
    rearranged_path = DATA_RAW / "Rearranged_Data.mat"
    dengue_xlsx     = DATA_RAW / "Dados_Modelagem.xlsx"

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading data …")
    data   = load_data(rearranged_path, dengue_xlsx)
    Dengue = data["Dengue"]   # (208,)
    EDI    = data["EDI"]      # (208,)

    n_samples = len(Dengue)
    n_train   = round(0.8 * n_samples)

    # ── experiment configuration ───────────────────────────────────────────────
    N_REPS         = 30
    N_BACK_SAMPLES = [1, 3, 4, 5, 6]
    IN_SERIES      = {"Dengue": np.log2(Dengue), "EDI": np.log2(EDI)}
    y1             = np.log2(Dengue)       # always predict log2(Dengue)

    IndTestOut     = np.arange(n_train, n_samples)
    n_test         = len(IndTestOut)
    YTest          = Dengue[IndTestOut]

    n_models = len(IN_SERIES) * len(N_BACK_SAMPLES)
    YPred    = np.full((n_test, N_REPS, n_models), np.nan)

    # ── training ───────────────────────────────────────────────────────────────
    t_start   = time.time()
    cnt_model = 0

    for in_name, x1 in IN_SERIES.items():
        for i_back in N_BACK_SAMPLES:
            print(f"\n=== Model {cnt_model + 1}/{n_models}  "
                  f"input={in_name}  lag={i_back} ===")

            IndTrainOut, IndTrainIn, _, IndTestIn = build_indices(
                n_samples, n_train, i_back)

            X_train = x1[IndTrainIn]          # (input_size, n_train_steps)
            Y_train = y1[IndTrainOut]         # (n_train_steps,)
            X_test  = x1[IndTestIn]           # (input_size, n_test_steps)

            YPred[:, :, cnt_model] = train_lstm(
                X_train, Y_train, X_test, N_REPS)

            cnt_model += 1

    elapsed_hr = (time.time() - t_start) / 3600
    print(f"\nDone. Elapsed: {elapsed_hr:.2f} hr")

    # ── save results ───────────────────────────────────────────────────────────
    out_path = DATA_MODELS / "eval_results.npz"
    np.savez(str(out_path), YPred=YPred, YTest=YTest)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()