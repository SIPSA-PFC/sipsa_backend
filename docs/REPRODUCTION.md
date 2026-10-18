# Reproducing the paper's LSTM results

How the Python port compares with the paper's MATLAB results (`Run1DLSTM_01.m`,
`ModelsPerform_1DLSTM_01.m`), what was changed to get there, and what is still open.

## Result (run `lstm-paper-30r`)

10 configurations × 30 repetitions, seed 1, data from the raw files; 166 weeks for training
(2016-01 to 2019-10), test weeks 2019-11 to 2019-52 — the paper's split. 79 minutes on 2 CPU
cores, TensorFlow 2.21. Reproduce with `python -m src.ml.train --run-id <name>` and
`python -m src.ml.evaluate --run data/models/<name>`.

| Configuration | Paper r | Ours r | Paper RMSE | Ours RMSE | Naive baseline (same lead) r / RMSE |
|---|---|---|---|---|---|
| dengue lag 1 (week i−1) | 0.899 | 0.614 ± 0.029 | 4.18 | 7.68 ± 0.27 | 0.929 / 2.85 |
| dengue lag 3 (i−3…i−1) | 0.845 | 0.333 ± 0.064 | 6.02 | 9.58 ± 0.24 | 0.929 / 2.85 |
| dengue lag 4 (i−4…i−2) | 0.731 | −0.167 ± 0.060 | 7.28 | 10.58 ± 0.12 | 0.863 / 3.95 |
| dengue lag 5 (i−5…i−3) | 0.661 | −0.108 ± 0.071 | 7.71 | 10.43 ± 0.19 | 0.802 / 4.76 |
| dengue lag 6 (i−6…i−4) | 0.537 | −0.244 ± 0.054 | 8.45 | 10.50 ± 0.14 | 0.745 / 5.40 |
| EDI lag 1 | 0.851 | **0.883** ± 0.016 | 4.82 | 7.33 ± 0.39 | 0.929 / 2.85 |
| EDI lag 3 | 0.803 | **0.836** ± 0.025 | 4.88 | 7.26 ± 0.43 | 0.929 / 2.85 |
| EDI lag 4 | 0.784 | **0.824** ± 0.017 | 5.44 | 8.07 ± 0.42 | 0.863 / 3.95 |
| EDI lag 5 | 0.850 | 0.798 ± 0.014 | 4.64 | 8.45 ± 0.42 | 0.802 / 4.76 |
| EDI lag 6 | 0.879 | 0.794 ± 0.014 | 4.35 | 8.77 ± 0.42 | 0.745 / 5.40 |

r and RMSE are averaged over repetitions (± standard error), on mean weekly cases per
neighborhood, as in the paper. The naive baseline predicts each week with the value observed
as many weeks earlier as the configuration's newest input — the same information the model has.

### What this shows

- **EDI (ovitrap) models reproduce the paper's correlations** (0.79–0.88 vs 0.78–0.88): they
  follow the 2019 epidemic curve. Their error is about 1.7× the paper's because they
  **underestimate the peak** — the predicted curve has the right shape at roughly half the
  height (outputs/lstm-paper-30r/Fig2_best_models.png).
- **Dengue-only models do not reproduce**: r 0.61 at lag 1, falling below zero at longer lags.
- **Naive baselines matter.** In the paper's own results, "same as last week" (r 0.93,
  RMSE 2.85) beats every model with a 1-week lead, and the paper's best model (dengue lag 1)
  correlates 0.98 with that baseline — it learned to repeat last week. The case for the system is
  the **4-week lead from ovitrap data**: EDI lag 6 beats "same as 4 weeks earlier" on correlation
  in both the paper (0.879 vs 0.745) and our run (0.794 vs 0.745), though not yet on RMSE in ours.

## What the first port got wrong (fixed)

1. **Training regime.** The port fed ~160 one-week sequences without carrying state, then
   predicted with a stateful copy warmed up over the full history — a regime never seen in
   training (one repetition gave RMSE 309 on a series whose maximum is 41). MATLAB's
   `trainNetwork` with a matrix input trains on **one continuous sequence**; the port now does too.
2. **Missing MATLAB defaults** that the script never sets but MATLAB applies: gradient clipping
   at 1 (`GradientThreshold`, set in the script but not ported), half mean-squared-error loss
   (`regressionLayer`), L2 regularization 1e-4, Adam ε 1e-8. In a 5-repetition probe the loss
   and L2 defaults raised test r from 0.40 to 0.57 (dengue lag 1) and 0.68 to 0.72 (EDI lag 6).
3. **Prediction** now equals MATLAB's `resetState` + `predictAndUpdateState` over training then
   test inputs (one pass over the joined sequence).
4. Data checks: the city-wide series from the raw files equals the database's
   (`python -m src.ml.data`), and `YTest` equals the paper's.

## Still open

The investigation was time-boxed. Hypotheses for the remaining gap, roughly in order of
likelihood:

- **Initialization and randomness.** MATLAB and Keras draw different initial weights and the
  dengue models are unstable across seeds (standard error 0.03–0.07 on r). The paper's 30
  repetitions may have landed differently.
- **Exact loss and clipping arithmetic.** MATLAB's regression loss normalization and its
  per-parameter clipping interact with Adam; small differences change how far a single long
  sequence is fitted (training-period fit is r ≈ 0.98 here, i.e. heavy memorization).
- **MATLAB version behaviour** (the results file is from 2022) — e.g. input normalization defaults
  in `sequenceInputLayer`.

Re-running `Run1DLSTM_01.m` in MATLAB with the training-period fit printed (and one fixed seed)
would settle most of these. Until then, the EDI models are the reproduced part of the paper.
