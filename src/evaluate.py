"""
evaluate.py
-----------
Python equivalent of ModelsPerform_1DLSTM_01.m.

Loads pre-computed model predictions from the .mat eval results files
and reproduces all three figures:
    A) Bar chart of RMSE per model
    B) Bar chart of correlation (r) per model
    C) Scatter plot RMSE vs r
    + Time series and scatter plots for the two best models

Usage
-----
    python -m src.ml.evaluate
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.ml.utils import (
    load_eval_results,
    compute_rmse,
    compute_r,
    matlab_datenum_to_datetime,
)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent.parent   # repo root (src/ml/ -> src/ -> root)
DATA_RAW = ROOT / "data" / "raw"
OUT_IMG  = ROOT / "outputs"
OUT_IMG.mkdir(parents=True, exist_ok=True)

MAT_FILE = DATA_RAW / "Models_1DLSTM_2022_02_04_EvalResults.mat"


# =============================================================================
# Helpers
# =============================================================================

def lag_label(i_back: int) -> str:
    """Reproduce the MATLAB MdlNames1 labels."""
    if i_back == 1:
        return "1 past sample"
    return f"{i_back}:{i_back - 2} past samples"


def build_model_meta(n_back_samples: list[int]) -> tuple[list[str], list[str]]:
    """Return (lag_labels, input_labels) for all models."""
    lag_labels   = [lag_label(b) for b in n_back_samples] * 2
    input_labels = (
        [r"$D_{i-j}\rightarrow D_i$"] * len(n_back_samples) +
        [r"$O_{i-j}\rightarrow D_i$"] * len(n_back_samples)
    )
    return lag_labels, input_labels


# =============================================================================
# Figure 1 – bar + scatter performance overview
# =============================================================================

def plot_performance(
    mean_rmse: np.ndarray,
    std_rmse:  np.ndarray,
    mean_r:    np.ndarray,
    std_r:     np.ndarray,
    n_mdls_half: int,
    lag_labels:  list[str],
    save_path: Path | None = None,
) -> None:
    """
    Reproduces the 3-panel figure from ModelsPerform_1DLSTM_01.m.
    """
    rmse_dd = mean_rmse[:n_mdls_half]
    rmse_od = mean_rmse[n_mdls_half:]
    e_rmse_dd = std_rmse[:n_mdls_half]
    e_rmse_od = std_rmse[n_mdls_half:]

    r_dd = mean_r[:n_mdls_half]
    r_od = mean_r[n_mdls_half:]
    e_r_dd = std_r[:n_mdls_half]
    e_r_od = std_r[n_mdls_half:]

    x_dd = np.arange(1, n_mdls_half + 1)
    x_od = np.arange(n_mdls_half + 2, 2 * n_mdls_half + 2)

    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(n_mdls_half)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")

    # ── A) RMSE bar chart ────────────────────────────────────────────────────
    ax = axes[0]
    for i in range(n_mdls_half):
        ax.bar(x_dd[i], rmse_dd[i], color=colors[i], label=lag_labels[i])
        ax.errorbar(x_dd[i], rmse_dd[i], yerr=e_rmse_dd[i], color="k", fmt="none")
        ax.bar(x_od[i], rmse_od[i], color=colors[i])
        ax.errorbar(x_od[i], rmse_od[i], yerr=e_rmse_od[i], color="k", fmt="none")

    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("A) Model errors", fontsize=12)
    ax.set_xticks([x_dd.mean(), x_od.mean()])
    ax.set_xticklabels(
        [r"$D_{i-j}\rightarrow D_i$", r"$O_{i-j}\rightarrow D_i$"],
        fontsize=11,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.box = True

    # ── B) r bar chart ───────────────────────────────────────────────────────
    ax = axes[1]
    for i in range(n_mdls_half):
        ax.bar(x_dd[i], r_dd[i], color=colors[i])
        ax.errorbar(x_dd[i], r_dd[i], yerr=e_r_dd[i], color="k", fmt="none")
        ax.bar(x_od[i], r_od[i], color=colors[i])
        ax.errorbar(x_od[i], r_od[i], yerr=e_r_od[i], color="k", fmt="none")

    ax.set_ylabel("r", fontsize=12)
    ax.set_title("B) Model performance", fontsize=12)
    ax.set_xticks([x_dd.mean(), x_od.mean()])
    ax.set_xticklabels(
        [r"$D_{i-j}\rightarrow D_i$", r"$O_{i-j}\rightarrow D_i$"],
        fontsize=11,
    )
    ax.set_ylim([0.4, 1.0])
    ax.set_aspect("equal", adjustable="box")

    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=colors[i], label=lag_labels[i])
            for i in range(n_mdls_half)
        ],
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
    )

    # ── C) RMSE vs r scatter ─────────────────────────────────────────────────
    ax = axes[2]
    for i in range(n_mdls_half):
        ax.plot(rmse_dd[i], r_dd[i], "o",
                markerfacecolor=colors[i], markeredgecolor="k", ms=8,
                label=r"$D_{i-j}\rightarrow D_i$" if i == 0 else "")
        ax.plot(rmse_od[i], r_od[i], "d",
                markerfacecolor=colors[i], markeredgecolor="k", ms=8,
                label=r"$O_{i-j}\rightarrow D_i$" if i == 0 else "")

    ax.set_xlabel("RMSE", fontsize=12)
    ax.set_ylabel("r",    fontsize=12)
    ax.set_title("C) Error vs Performance", fontsize=12)
    ax.set_ylim([0.55, 1.0])
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        [plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor="gray", markeredgecolor="k", ms=8),
         plt.Line2D([0], [0], marker="d", color="w",
                    markerfacecolor="gray", markeredgecolor="k", ms=8)],
        [r"$D_{i-j}\rightarrow D_i$", r"$O_{i-j}\rightarrow D_i$"],
        fontsize=10, frameon=False,
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


# =============================================================================
# Figure 2 – best model details (scatter + time series)
# =============================================================================

def plot_best_models(
    y_pred_dd: np.ndarray,   # (n_timesteps, n_reps)  log2-scale
    y_pred_od: np.ndarray,
    y_test:    np.ndarray,   # (n_timesteps,)  linear scale
    t_test:    np.ndarray,   # (n_timesteps,)  MATLAB datenums
    color_dd,
    color_od,
    save_path: Path | None = None,
) -> None:
    """
    Reproduces the 2×2 figure: scatter (top) + time series (bottom)
    for the D→D and O→D best models.
    """
    pred_dd_mean = np.mean(2.0 ** y_pred_dd, axis=1)
    pred_od_mean = np.mean(2.0 ** y_pred_od, axis=1)
    dates        = matlab_datenum_to_datetime(t_test)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor("white")

    def scatter_panel(ax, y_true, y_pred_mean, color, subtitle):
        ax.scatter(y_true, y_pred_mean, s=20,
                   facecolor=color, edgecolor="k", zorder=3)
        # least-squares line
        m, b = np.polyfit(y_true, y_pred_mean, 1)
        xs = np.array([y_true.min(), y_true.max()])
        ax.plot(xs, m * xs + b, color="gray", lw=1)
        r = np.corrcoef(y_true, y_pred_mean)[0, 1]
        p_val = _corr_pvalue(r, len(y_true))
        ax.set_title(f"r = {r:.2f}, p = {p_val:.2e}", fontsize=10)
        ax.set_xlabel("Known",     fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.set_xlim(0, 35); ax.set_ylim(0, 35)
        ax.set_aspect("equal")

    def ts_panel(ax, dates, pred_mean, y_true, color, title):
        ax.plot(dates, pred_mean, color=color,   lw=1.5, label="Predicted")
        ax.plot(dates, y_true,   color="black",  lw=1.0, label="Known")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Mean Dengue Occ", fontsize=10)
        ax.set_ylim(0, 35)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.legend(frameon=False, fontsize=9)

    scatter_panel(axes[0, 0], y_test, pred_dd_mean, color_dd, r"$D \rightarrow D$")
    scatter_panel(axes[1, 0], y_test, pred_od_mean, color_od, r"$O \rightarrow D$")
    ts_panel(axes[0, 1], dates, pred_dd_mean, y_test, color_dd,
             r"$D_{i-j} \rightarrow D_i$")
    ts_panel(axes[1, 1], dates, pred_od_mean, y_test, color_od,
             r"$O_{i-j} \rightarrow D_i$")

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


# =============================================================================
# Tiny statistics helper
# =============================================================================

def _corr_pvalue(r: float, n: int) -> float:
    """Two-tailed p-value for a Pearson r."""
    from scipy import stats
    t_stat = r * np.sqrt((n - 2) / (1 - r ** 2 + 1e-12))
    return 2 * stats.t.sf(np.abs(t_stat), df=n - 2)


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"Loading eval results from:\n  {MAT_FILE}\n")
    res = load_eval_results(MAT_FILE)

    YPred = res["YPred"]   # (n_timesteps, n_reps, n_models)
    YTest = res["YTest"]   # (n_timesteps,)
    tTest = res["tTest"]   # (n_timesteps,)

    N_BACK_SAMPLES = [1, 3, 4, 5, 6]
    n_mdls_half    = len(N_BACK_SAMPLES)   # 5
    n_models       = YPred.shape[2]        # should be 10

    lag_labels, _ = build_model_meta(N_BACK_SAMPLES)

    # ── compute metrics for every model ──────────────────────────────────────
    mean_rmse = np.zeros(n_models)
    std_rmse  = np.zeros(n_models)
    mean_r    = np.zeros(n_models)
    std_r     = np.zeros(n_models)

    for i in range(n_models):
        mean_rmse[i], std_rmse[i] = compute_rmse(YPred[:, :, i], YTest)
        mean_r[i],    std_r[i]    = compute_r(   YPred[:, :, i], YTest)

    # ── Figure 1 ─────────────────────────────────────────────────────────────
    plot_performance(
        mean_rmse, std_rmse, mean_r, std_r,
        n_mdls_half=n_mdls_half,
        lag_labels=lag_labels[:n_mdls_half],
        save_path=OUT_IMG / "Fig1_model_performance.png",
    )

    # ── Figure 2: best D→D (model 0) and best O→D (model 9) ────────────────
    # Matches MATLAB: YPredMdls(:,:,1) and YPredMdls(:,:,10)  (1-indexed → 0,9)
    cmap     = plt.get_cmap("tab10")
    color_dd = cmap(0)
    color_od = cmap(4)   # same colour as bar 5 in MATLAB (bRMSE(5))

    plot_best_models(
        y_pred_dd=YPred[:, :, 0],
        y_pred_od=YPred[:, :, 9],
        y_test=YTest,
        t_test=tTest,
        color_dd=color_dd,
        color_od=color_od,
        save_path=OUT_IMG / "Fig2_best_models.png",
    )


if __name__ == "__main__":
    main()