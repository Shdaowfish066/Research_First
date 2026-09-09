"""
=============================================================================
CAMERA-READY FIGURES  (iCONEECT 2026)  --  Reviewer 3, point 1
=============================================================================
Both figures are regenerated as vector PDF, replacing the raster originals,
and both now show seed variability rather than a single run.

  assets/convergence_curves.pdf   Fig. 1 -- per-epoch validation LOSS, mean
                                  over 5 seeds with +/-1 sd bands
  assets/architecture_spread.pdf  Fig. 2 -- the 400 cell means grouped by
                                  architecture, with 95% CI of each
                                  architecture mean
  assets/_preview_*.png           300 DPI rasters for visual inspection only

Data sources -- no value is typed by hand:
  Fig. 1  results/convergence_curves_5seed.csv
  Fig. 2  <ARCH>_5_seed_variance_results/mean_100_cells.csv  (x4)
          <ARCH>_5_seed_variance_results/all_runs.csv        (x4)

WHY FIG. 2 CHANGED. The submitted Fig. 2 plotted the single best activation
pair per architecture. Those peaks are maxima of a noisy single-seed sweep and
do not reproduce (Table~\\ref{tab:reproducibility}), so plotting them would
now be misleading. The replacement shows the full distribution of the 100
replicated cell means per architecture, which is what the data supports: the
architectures separate cleanly, the activation pairs within them do not.

NOTE ON FIG. 1: the y-axis is validation LOSS, not accuracy. The submitted
caption said "Per-Epoch Validation Accuracy" and was wrong.

Sizing: authored at 3.22 in = 0.92 x the 3.5 in IEEE column, matching the
existing \\includegraphics[width=0.92\\columnwidth]. bbox_inches='tight' trims
to the artists, so the final width is not guaranteed to equal figsize -- an
outside legend can push it wider, and LaTeX then scales the PDF down, shrinking
every glyph. save() therefore reads the trimmed MediaBox back out of the
written PDF and reports the true on-page size of the smallest text, warning
below 8 pt.

pdf.fonttype = 42 embeds TrueType subsets, required for the IEEE font check.
=============================================================================
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman No9 L",
                   "Liberation Serif", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

COL_W = 0.92 * 3.5
ASSETS = "assets"
MIN_PT = 8.0
ARCH = ["BiLSTM+LSTM", "BiLSTM+BiLSTM", "BiGRU+GRU", "BiGRU+BiGRU"]
os.makedirs(ASSETS, exist_ok=True)


def save(fig, stem):
    pdf = os.path.join(ASSETS, f"{stem}.pdf")
    png = os.path.join(ASSETS, f"_preview_{stem}.png")
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    with open(pdf, "rb") as fh:
        box = re.search(rb"/MediaBox\s*\[([^\]]*)\]", fh.read()).group(1).split()
    x0, y0, x1, y1 = (float(v) for v in box)
    w_in, h_in = (x1 - x0) / 72.0, (y1 - y0) / 72.0
    scale = COL_W / w_in
    on_page = MIN_PT * scale
    print(f"  {pdf}  ({os.path.getsize(pdf)/1024:.0f} KB)")
    print(f"    {w_in:.2f} x {h_in:.2f} in -> scale {scale:.3f}")
    flag = "OK" if on_page >= MIN_PT else "*** BELOW 8 pt ***"
    print(f"    smallest text renders at {on_page:.2f} pt  {flag}")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 -- per-epoch validation loss, 5 seeds, mean +/- 1 sd
# ════════════════════════════════════════════════════════════════════════════
def figure_1():
    src = os.path.join("results", "convergence_curves_5seed.csv")
    if not os.path.exists(src):
        print("Fig. 1  SKIPPED -- convergence_curves_5seed.csv not present yet")
        return
    d = pd.read_csv(src)
    order = (d[d.epoch == 1].groupby("af").val_acc_e1.mean().sort_values(ascending=False).index
             if "val_acc_e1" in d.columns
             else d[d.epoch == 1].groupby("af").val_acc.mean().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]

    # only epochs every seed reached, so the mean is over a constant n
    n_seeds = d.seed.nunique()
    keep = d.groupby(["af", "epoch"]).seed.nunique()
    for i, af in enumerate(order):
        g = d[d.af == af]
        eps = sorted(e for e in g.epoch.unique() if keep.loc[(af, e)] == n_seeds)
        m = np.array([g[g.epoch == e].val_loss.mean() for e in eps])
        s = np.array([g[g.epoch == e].val_loss.std(ddof=1) for e in eps])
        ax.plot(eps, m, color=cmap(i % 10), marker=markers[i % 10],
                markersize=2.6, markeredgewidth=0, label=af, zorder=3)
        ax.fill_between(eps, m - s, m + s, color=cmap(i % 10), alpha=0.10,
                        linewidth=0, zorder=1)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_xticks(sorted(e for e in d.epoch.unique()
                         if all(keep.get((a, e), 0) == n_seeds for a in order)))
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              frameon=False, handlelength=0.9, handletextpad=0.35,
              columnspacing=0.5, borderpad=0.2, labelspacing=0.3)

    print(f"Fig. 1  (validation LOSS; mean of {n_seeds} seeds, +/-1 sd bands)")
    save(fig, "convergence_curves")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 -- distribution of the 400 replicated cell means, by architecture
# ════════════════════════════════════════════════════════════════════════════
def figure_2():
    means, runs = {}, {}
    for a in ARCH:
        means[a] = pd.read_csv(f"{a}_5_seed_variance_results/mean_100_cells.csv")
        runs[a] = pd.read_csv(f"{a}_5_seed_variance_results/all_runs.csv")

    fig, ax = plt.subplots(figsize=(COL_W, 2.10))
    rng = np.random.default_rng(0)
    C_LSTM, C_GRU = "#9B59B6", "#2ECC71"

    for i, a in enumerate(ARCH):
        v = means[a].acc_mean.to_numpy()
        col = C_GRU if "GRU" in a and "LSTM" not in a else C_LSTM
        x = i + rng.uniform(-0.16, 0.16, len(v))
        ax.scatter(x, v, s=5, color=col, alpha=0.45, linewidths=0, zorder=2)
        r = runs[a].accuracy.to_numpy()
        mu = r.mean()
        ci = 1.96 * r.std(ddof=1) / np.sqrt(len(r))
        ax.errorbar(i, mu, yerr=ci, fmt="_", color="black", markersize=13,
                    markeredgewidth=1.4, elinewidth=1.4, capsize=3, zorder=4)

    ax.set_xticks(range(len(ARCH)))
    ax.set_xticklabels(["BiLSTM\n+LSTM", "BiLSTM\n+BiLSTM", "BiGRU\n+GRU",
                        "BiGRU\n+BiGRU"])
    ax.set_ylabel("Test accuracy (\\%)" if plt.rcParams.get("text.usetex")
                  else "Test accuracy (%)")
    ax.set_ylim(85.8, 88.1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=C_LSTM, markersize=3.5,
               label="LSTM cell"),
        Line2D([], [], marker="o", ls="", color=C_GRU, markersize=3.5,
               label="GRU cell"),
        Line2D([], [], marker="_", ls="", color="black", markersize=9,
               markeredgewidth=1.4, label="mean, 95\\% CI"
               if plt.rcParams.get("text.usetex") else "mean, 95% CI")],
        loc="lower right", frameon=False, handletextpad=0.4,
        borderpad=0.2, labelspacing=0.25)

    lo = min(means[a].acc_mean.min() for a in ARCH)
    print(f"Fig. 2  (400 cell means by architecture; y truncated at 85.8, "
          f"lowest cell {lo:.2f} clipped -- state in caption)")
    save(fig, "architecture_spread")


if __name__ == "__main__":
    figure_1()
    figure_2()
    print("\nInspect assets/_preview_*.png before compiling.")
