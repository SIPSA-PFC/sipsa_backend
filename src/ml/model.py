"""
ml/model.py
-----------
The 1D-LSTM from Run1DLSTM_01.m (function TrainLSTM), in Keras.

MATLAB                                         Keras
sequenceInputLayer(inputSize)                  Input(shape=(None, input_size))
lstmLayer(100)  (sequence output)              LSTM(100, return_sequences=True)
fullyConnectedLayer(1) + regressionLayer       Dense(1), loss = half mean squared error
                                               (regressionLayer's loss for sequences)
'adam', 'InitialLearnRate', 0.005              Adam(learning_rate=0.005, epsilon=1e-8)
L2Regularization 1e-4  (trainingOptions        L2 on kernels and recurrent kernels,
  default, not set in the script)                factor 5e-5 (MATLAB adds lambda/2*|w|^2)
'GradientThreshold', 1  (per parameter, L2)    clipnorm=1.0 (per variable)
'LearnRateSchedule','piecewise', drop 0.2      0.005 for epochs 1-125,
  every 125 epochs                             0.001 for 126-250
'MaxEpochs', 250                               epochs=250

The two defaults MATLAB applies without the script asking (half-MSE loss,
L2 regularization 1e-4) were missing from the first port; adding them
raised test correlation in a 5-rep probe (dengue lag 1: 0.40 -> 0.57;
EDI lag 6: 0.68 -> 0.72). See docs/REPRODUCTION.md.

How MATLAB trains it: trainNetwork(XTrain, YTrain, ...) with XTrain a
numeric matrix (features x weeks) is ONE sequence covering the whole
training period. Each epoch is one gradient step over that sequence, and the
LSTM carries its state from week to week. (MiniBatchSize 64 has no effect
with a single sequence.)

How MATLAB predicts: resetState, then predictAndUpdateState over the
training inputs, then over each test input in turn. Running the trained
network once over [training inputs, test inputs] gives the same outputs.
"""

from __future__ import annotations

import numpy as np

EPOCHS = 250
HIDDEN_UNITS = 100
LEARNING_RATE = 0.005
LR_DROP_EPOCH = 125
LR_DROP_FACTOR = 0.2
GRADIENT_THRESHOLD = 1.0
L2_REGULARIZATION = 1e-4          # MATLAB trainingOptions default
ADAM_EPSILON = 1e-8               # MATLAB default (Keras: 1e-7)

_tf = None


def tf():
    """TensorFlow, imported on first use (it takes seconds and the API may not need it)."""
    global _tf
    if _tf is None:
        try:
            import tensorflow
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required: pip install tensorflow") from exc
        _tf = tensorflow
    return _tf


def half_mse(y_true, y_pred):
    """MATLAB regressionLayer loss for sequences: 1/2 x mean squared error over time steps."""
    t = tf()
    return 0.5 * t.reduce_mean(t.square(y_true - y_pred))


def build_model(input_size: int):
    t = tf()
    l2 = t.keras.regularizers.l2(L2_REGULARIZATION / 2)
    model = t.keras.Sequential([
        t.keras.layers.Input(shape=(None, input_size)),
        t.keras.layers.LSTM(HIDDEN_UNITS, return_sequences=True,
                            kernel_regularizer=l2, recurrent_regularizer=l2),
        t.keras.layers.Dense(1, kernel_regularizer=l2),
    ])
    model.compile(
        optimizer=t.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=ADAM_EPSILON,
                                          clipnorm=GRADIENT_THRESHOLD),
        loss=half_mse,
    )
    return model


def load(path):
    """Load a saved repetition for prediction (the training loss is not needed)."""
    return tf().keras.models.load_model(path, compile=False)


def _learning_rate(epoch: int, _current: float) -> float:
    return LEARNING_RATE if epoch < LR_DROP_EPOCH else LEARNING_RATE * LR_DROP_FACTOR


def train_rep(x_train: np.ndarray, y_train: np.ndarray, seed: int):
    """
    Train one repetition.
    x_train: (n_features, n_weeks) inputs; y_train: (n_weeks,) log2 targets.
    """
    t = tf()
    t.keras.utils.set_random_seed(seed)
    model = build_model(x_train.shape[0])
    model.fit(
        x_train.T[np.newaxis],                  # (1 sequence, weeks, features)
        y_train[np.newaxis, :, np.newaxis],     # (1, weeks, 1)
        epochs=EPOCHS,
        batch_size=1,
        shuffle=False,
        verbose=0,
        callbacks=[t.keras.callbacks.LearningRateScheduler(_learning_rate)],
    )
    return model


def _compiled(model):
    """
    A compiled (graph-mode) forward pass for this model, built once and kept
    on the model. Eager Keras runs the LSTM step by step in Python (~0.35 s
    for 200 weeks); the compiled call takes ~6 ms and gives the same outputs.
    The signature accepts any sequence length, so it is traced only once.
    """
    fn = getattr(model, "_sipsa_forward", None)
    if fn is None:
        t = tf()

        def forward(x):
            return model(x, training=False)

        fn = t.function(forward, input_signature=[t.TensorSpec([1, None, model.input_shape[-1]], t.float32)],
                        reduce_retracing=True)
        model._sipsa_forward = fn
    return fn


def predict_sequence(model, x: np.ndarray) -> np.ndarray:
    """
    Outputs for every week of x (n_features, n_weeks), starting from a reset
    state — equivalent to MATLAB's resetState + predictAndUpdateState.
    """
    out = _compiled(model)(x.T[np.newaxis].astype("float32"))
    return np.asarray(out).reshape(-1)
