"""
ml/train.py
-----------
Trains the paper's 1D-LSTM models (Run1DLSTM_01.m) and saves them.

    python -m src.ml.train                 # full run: 10 configurations x 30 reps (~80 min on 2 CPUs)
    python -m src.ml.train --dev           # 3 reps per configuration (~8 min), for development
    python -m src.ml.train --configs dengue:1,edi:6 --reps 5

Configurations: input dengue or edi, x lags 1, 3, 4, 5, 6. The target is
always log2 of city-wide mean dengue cases. Training uses the first 80% of
weeks; the remaining weeks are the test period, as in the paper.

Output (data/models/<run_id>/):
    <input>_lag<lag>/rep_NN.keras   one saved model per repetition
    results.npz                     YPred (test weeks x reps x configs, log2),
                                    YTest, test week labels, configuration names
    metrics.json                    RMSE / r per configuration and naive baselines
    run.json                        settings, data window, timing

Each configuration is also registered in the model_run table (skip with
--no-register), which is how the API finds trained models. To register an
existing run in another (or rebuilt) database without retraining:

    python -m src.ml.train --register data/models/<run_id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.ml import data as ml_data
from src.ml import model as ml_model
from src.ml.preprocessing import (
    INPUT_TYPES, SUPPORTED_LAGS, build_inputs, lag_offsets, log2_transform, n_train_samples, split_targets,
)
from src.ml.utils import compute_r, compute_rmse, persistence_baseline, rmse_and_r

log = logging.getLogger("sipsa.train")

ROOT       = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "data" / "models"
TRAIN_SPLIT = 0.8


def config_name(input_type: str, lag: int) -> str:
    return f"{input_type}_lag{lag}"


def parse_configs(text: str | None) -> list[tuple[str, int]]:
    if not text:
        return [(i, lag) for i in INPUT_TYPES for lag in SUPPORTED_LAGS]
    configs = []
    for item in text.split(","):
        input_type, lag = item.split(":")
        if input_type not in INPUT_TYPES or int(lag) not in SUPPORTED_LAGS:
            raise SystemExit(f"Unknown configuration {item!r}; use input:lag with input in {INPUT_TYPES} and lag in {SUPPORTED_LAGS}")
        configs.append((input_type, int(lag)))
    return configs


def config_metrics(y_pred_log2: np.ndarray, y_test: np.ndarray, dengue: np.ndarray,
                   test_targets: np.ndarray, lag: int) -> dict:
    mean_rmse, se_rmse = compute_rmse(y_pred_log2, y_test)
    mean_r, se_r = compute_r(y_pred_log2, y_test)
    lead = min(lag_offsets(lag))
    b_rmse, b_r = rmse_and_r(y_test, persistence_baseline(dengue, test_targets, lead))
    return {
        "mean_rmse": float(mean_rmse), "std_err_rmse": float(se_rmse),
        "mean_r": float(mean_r), "std_err_r": float(se_r),
        "baseline": {"name": f"same as {lead} week(s) earlier", "lead_weeks": lead,
                     "rmse": b_rmse, "r": b_r},
    }


async def _register(run_id: str, results: list[dict], series: ml_data.CitySeries) -> None:
    from sqlalchemy import select
    from src.api.models.db_models import EpiWeek, ModelRun
    from src.core.database import AsyncSessionLocal, engine

    try:
        async with AsyncSessionLocal() as session, session.begin():
            week_ids = {(w.epi_year, w.epi_week): w.id for w in (await session.execute(select(EpiWeek))).scalars()}
            for r in results:
                first, last = series.weeks[r["train_targets"][0]], series.weeks[r["train_targets"][-1]]
                if first not in week_ids or last not in week_ids:
                    raise SystemExit("Training weeks are not in the database; run the ETL first, or use --no-register.")
                session.add(ModelRun(
                    model_version=run_id, input_type=r["input_type"], lag=r["lag"],
                    train_from_week_id=week_ids[first], train_to_week_id=week_ids[last],
                    n_reps=r["n_reps"], metrics=r["metrics"], artifact_path=r["artifact_path"],
                ))
    finally:
        await engine.dispose()


def register_existing(run_dir: Path) -> None:
    """Record an already-trained run (its run.json / metrics.json) in model_run."""
    run = json.loads((run_dir / "run.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    series = ml_data.load_from_raw()
    results = []
    for name in run["configs"]:
        input_type, lag = name.rsplit("_lag", 1)
        train_targets, _ = split_targets(len(series.weeks), int(lag), run["train_split"])
        results.append({"input_type": input_type, "lag": int(lag), "n_reps": run["n_reps"],
                        "metrics": metrics[name], "train_targets": train_targets.tolist(),
                        "artifact_path": str((run_dir / name).resolve().relative_to(ROOT))})
    asyncio.run(_register(run["run_id"], results, series))
    log.info("Registered %d configurations of %s", len(results), run["run_id"])


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m src.ml.train", description=__doc__.split("\n\n")[0])
    ap.add_argument("--reps", type=int, default=30, help="repetitions per configuration (paper: 30)")
    ap.add_argument("--dev", action="store_true", help="quick run: 3 repetitions")
    ap.add_argument("--configs", help="subset, e.g. dengue:1,edi:6 (default: all 10)")
    ap.add_argument("--source", choices=["raw", "db"], default="raw",
                    help="raw files (as the paper) or the database (default: raw)")
    ap.add_argument("--seed", type=int, default=1, help="base random seed (paper: 1)")
    ap.add_argument("--run-id", help="name of the output folder (default: lstm-<date>-<reps>r)")
    ap.add_argument("--no-register", action="store_true", help="do not record the run in model_run")
    ap.add_argument("--register", type=Path, metavar="RUN_DIR",
                    help="only register an existing run folder in model_run (no training)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.register:
        register_existing(args.register)
        return

    n_reps  = 3 if args.dev else args.reps
    configs = parse_configs(args.configs)
    run_id  = args.run_id or f"lstm-{datetime.now(timezone.utc):%Y%m%d-%H%M}-{n_reps}r"
    out_dir = MODELS_DIR / run_id
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; pick another --run-id")

    if args.source == "raw":
        series = ml_data.load_from_raw()
    else:
        from src.core.database import AsyncSessionLocal, engine

        async def _load():
            try:
                async with AsyncSessionLocal() as session:
                    return await ml_data.load_from_db(session)
            finally:
                await engine.dispose()
        series = asyncio.run(_load())

    y = log2_transform(series.dengue, "dengue")
    n = len(y)
    log.info("Run %s: %d configurations x %d reps, %d weeks (%s to %s), %d for training",
             run_id, len(configs), n_reps, n, series.weeks[0], series.weeks[-1], n_train_samples(n, TRAIN_SPLIT))

    _, test_targets = split_targets(n, 1, TRAIN_SPLIT)
    y_test = series.dengue[test_targets]
    y_pred = np.full((len(test_targets), n_reps, len(configs)), np.nan)
    results, t_start = [], time.time()

    for ci, (input_type, lag) in enumerate(configs):
        t_cfg = time.time()
        x = log2_transform(series.series(input_type), input_type)
        train_targets, test_t = split_targets(n, lag, TRAIN_SPLIT)
        x_train, y_train = build_inputs(x, train_targets, lag), y[train_targets]
        x_all = build_inputs(x, np.concatenate([train_targets, test_t]), lag)
        cfg_dir = out_dir / config_name(input_type, lag)
        cfg_dir.mkdir(parents=True)

        for rep in range(n_reps):
            model = ml_model.train_rep(x_train, y_train, seed=args.seed * 100_000 + ci * 1_000 + rep)
            y_pred[:, rep, ci] = ml_model.predict_sequence(model, x_all)[-len(test_t):]
            model.save(cfg_dir / f"rep_{rep:02d}.keras")

        metrics = config_metrics(y_pred[:, :, ci], y_test, series.dengue, test_t, lag)
        results.append({"input_type": input_type, "lag": lag, "n_reps": n_reps, "metrics": metrics,
                        "train_targets": train_targets.tolist(),
                        "artifact_path": str(cfg_dir.relative_to(ROOT))})
        log.info("%-12s r=%.3f RMSE=%.2f | baseline r=%.3f RMSE=%.2f | %.0fs",
                 config_name(input_type, lag), metrics["mean_r"], metrics["mean_rmse"],
                 metrics["baseline"]["r"], metrics["baseline"]["rmse"], time.time() - t_cfg)

    elapsed = time.time() - t_start
    names = [config_name(i, lag) for i, lag in configs]
    np.savez(out_dir / "results.npz", YPred=y_pred, YTest=y_test, configs=np.array(names),
             test_weeks=np.array([f"{a}-{b:02d}" for a, b in (series.weeks[t] for t in test_targets)]))
    (out_dir / "metrics.json").write_text(json.dumps(
        {name: r["metrics"] for name, r in zip(names, results)}, indent=2))
    (out_dir / "run.json").write_text(json.dumps({
        "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "source": args.source, "n_reps": n_reps, "seed": args.seed, "train_split": TRAIN_SPLIT,
        "weeks": [f"{a}-{b:02d}" for a, b in (series.weeks[0], series.weeks[-1])],
        "test_weeks": [f"{a}-{b:02d}" for a, b in (series.weeks[test_targets[0]], series.weeks[test_targets[-1]])],
        "configs": names, "elapsed_seconds": round(elapsed),
        "tensorflow": ml_model.tf().__version__, "python": platform.python_version(),
        "git_commit": _git_commit(),
    }, indent=2))
    log.info("Done in %.1f min. Saved to %s", elapsed / 60, out_dir)

    if not args.no_register:
        asyncio.run(_register(run_id, results, series))
        log.info("Registered %d configurations in model_run as %s", len(results), run_id)


if __name__ == "__main__":
    main()
