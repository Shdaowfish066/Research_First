"""
=============================================================================
FIGURES FOR THE INDEPENDENT-SEED CONVERGENCE RUN (seeds 10/20/30/40/50)
=============================================================================
  assets/convergence_curves_seeds10_50.pdf
        Direct companion to the published Fig. 2 -- per-epoch validation
        LOSS, mean over the five NEW seeds with +/-1 sd bands, drawn at the
        same size and with the same rcParams so the two can be set side by
        side or swapped in.

  assets/convergence_seed_comparison.pdf
        Why the two differ. Left: epoch-1 accuracy, published seeds vs new
        seeds, one line per activation (+/-1 sd). Right: the per-run epoch-1
        accuracies of the four "smooth" functions across all ten seeds,
        showing that GELU is the only one of them that lands runs on the
        ~50% plateau.

Style is imported from make_figures.py rather than restated, so rcParams,
COL_W and save() (which reads the trimmed MediaBox back out and warns if
text would render below 8 pt) are identical to the camera-ready figures.

COLOR CHOICE. make_figures.figure_1 assigns colors by epoch-1 rank, so a
function's color depends on the data being plotted. That is fine for one
standalone figure but useless for a comparison: the ranking is exactly what
changed. Here each activation is instead pinned to a fixed color/marker,
keyed by the PUBLISHED ranking -- so every function keeps the color it has
in the published Fig. 2, and a reader comparing the two sees like for like.

Data: results/convergence_curves_seeds10_50.csv  (new)
      results/convergence_curves_5seed.csv       (published)
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from make_figures import COL_W, save  # rcParams applied on import

NEW = "results/convergence_curves_seeds10_50.csv"
OLD = "results/convergence_curves_5seed.csv"

# published epoch-1 ranking -> fixes each function's color and marker
PUB_ORDER = ["Hardswish", "GELU", "Mish", "SiLU", "PReLU",
             "Tanh", "LeakyReLU", "ELU", "ReLU", "SELU"]
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]
CMAP = plt.get_cmap("tab10")
COLOR = {af: CMAP(i % 10) for i, af in enumerate(PUB_ORDER)}
MARKER = {af: MARKERS[i % 10] for i, af in enumerate(PUB_ORDER)}

SMOOTH_4 = ["Mish", "GELU", "SiLU", "Hardswish"]
RELU_3 = ["ReLU", "LeakyReLU", "SELU"]
PLATEAU = 55.0


def epoch1(d):
    e1 = d[d.epoch == 1]
    return (e1.groupby("af").agg(acc=("val_acc", "mean"), sd=("val_acc", "std"))
            .sort_values("acc", ascending=False))


def curves(d):
    """mean and sd of val_loss per epoch, only epochs every seed reached."""
    n = d.seed.nunique()
    reached = d.groupby(["af", "epoch"]).seed.nunique()
    out = {}
    for af in d.af.unique():
        g = d[d.af == af]
        eps = [e for e in sorted(g.epoch.unique()) if reached.loc[(af, e)] == n]
        m = np.array([g[g.epoch == e].val_loss.mean() for e in eps])
        s = np.array([g[g.epoch == e].val_loss.std(ddof=1) for e in eps])
        out[af] = (eps, m, s)
    return out


# ════════════════════════════════════════════════════════════════════════════
# FIG A -- validation loss curves, new seeds (companion to published Fig. 2)
# ════════════════════════════════════════════════════════════════════════════
def fig_curves():
    d = pd.read_csv(NEW)
    order = epoch1(d).index                       # legend follows NEW ranking
    cv = curves(d)

    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    for af in order:
        eps, m, s = cv[af]
        ax.plot(eps, m, color=COLOR[af], marker=MARKER[af], markersize=2.6,
                markeredgewidth=0, label=af, zorder=3)
        ax.fill_between(eps, m - s, m + s, color=COLOR[af], alpha=0.10,
                        linewidth=0, zorder=1)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_xticks(sorted({e for af in order for e in cv[af][0]}))
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              frameon=False, handlelength=0.9, handletextpad=0.35,
              columnspacing=0.5, borderpad=0.2, labelspacing=0.3)

    print(f"Fig. A  (validation LOSS; mean of {d.seed.nunique()} seeds "
          f"10/20/30/40/50, +/-1 sd bands)")
    save(fig, "convergence_curves_seeds10_50")


# ════════════════════════════════════════════════════════════════════════════
# FIG B -- what moved between the two seed sets
# ════════════════════════════════════════════════════════════════════════════
def fig_comparison():
    new, old = pd.read_csv(NEW), pd.read_csv(OLD)
    gn, go = epoch1(new), epoch1(old)

    # shared y so the two panels are read on one scale; headroom above the
    # highest run (81.1) leaves the right panel's legend clear of the points
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(COL_W, 2.35), sharey=True,
                                   gridspec_kw=dict(width_ratios=[1.25, 1],
                                                    wspace=0.12))
    axL.set_ylim(47, 90)

    # ---- left: paired epoch-1 accuracy, published -> new ------------------
    for af in PUB_ORDER:
        y0, y1 = go.acc[af], gn.acc[af]
        s0, s1 = go.sd[af], gn.sd[af]
        big = af == "GELU"
        axL.plot([0, 1], [y0, y1], color=COLOR[af], lw=1.8 if big else 0.9,
                 marker=MARKER[af], markersize=3.2, markeredgewidth=0,
                 alpha=1.0 if big else 0.75, zorder=4 if big else 3)
        for x, y, s in ((0, y0, s0), (1, y1, s1)):
            axL.plot([x, x], [y - s, y + s], color=COLOR[af], lw=0.7,
                     alpha=0.35, solid_capstyle="butt", zorder=2)

    axL.annotate("GELU", xy=(1, gn.acc["GELU"]), xytext=(1.06, gn.acc["GELU"]),
                 color=COLOR["GELU"], fontsize=7, va="center", ha="left")
    axL.axhspan(47, PLATEAU, color="0.85", zorder=0)
    axL.set_xlim(-0.30, 1.42)
    axL.set_xticks([0, 1])
    axL.set_xticklabels(["seeds\n42,0,1,2,3", "seeds\n10..50"])
    axL.set_ylabel("Epoch-1 val.\\ accuracy (%)"
                   if plt.rcParams.get("text.usetex")
                   else "Epoch-1 val. accuracy (%)")
    axL.grid(True, axis="y", alpha=0.3)
    axL.set_axisbelow(True)
    for side in ("top", "right"):
        axL.spines[side].set_visible(False)

    # ---- right: per-run epoch-1 accuracy of the four smooth functions -----
    e1n = new[new.epoch == 1]
    e1o = old[old.epoch == 1]
    rng = np.random.default_rng(0)
    for i, af in enumerate(SMOOTH_4):
        for d_, mfc in ((e1o, "none"), (e1n, None)):
            v = d_[d_.af == af].val_acc.to_numpy()
            x = i + rng.uniform(-0.17, 0.17, len(v))
            axR.scatter(x, v, s=9, facecolors=COLOR[af] if mfc is None else "none",
                        edgecolors=COLOR[af], linewidths=0.7, alpha=0.85, zorder=3)
        pooled = np.r_[e1o[e1o.af == af].val_acc, e1n[e1n.af == af].val_acc]
        axR.plot([i - 0.30, i + 0.30], [pooled.mean()] * 2, color="black",
                 lw=1.2, zorder=4)

    axR.axhspan(47, PLATEAU, color="0.85", zorder=0)
    axR.annotate("$\\sim$50\\% plateau" if plt.rcParams.get("text.usetex")
                 else "~50% plateau", xy=(3.45, PLATEAU + 0.8), fontsize=6,
                 color="0.35", ha="right", va="bottom", zorder=5)
    axR.set_xticks(range(len(SMOOTH_4)))
    axR.set_xticklabels(SMOOTH_4, rotation=30, ha="right")
    axR.set_xlim(-0.55, 3.55)
    axR.grid(True, axis="y", alpha=0.3)
    axR.set_axisbelow(True)
    for side in ("top", "right"):
        axR.spines[side].set_visible(False)

    axR.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor="0.35", markersize=3.2, label="published seeds"),
        Line2D([], [], marker="o", ls="", color="0.35", markersize=3.2,
               label="new seeds"),
        Line2D([], [], marker="_", ls="", color="black", markersize=8,
               markeredgewidth=1.2, label="10-seed mean")],
        loc="upper left", frameon=False, fontsize=6, handletextpad=0.4,
        borderpad=0.15, labelspacing=0.2)

    stuck = int(e1n[e1n.af == "GELU"].val_acc.lt(PLATEAU).sum())
    print(f"Fig. B  (left: epoch-1 accuracy shift; right: per-run spread of "
          f"the smooth four -- GELU puts {stuck}/5 new seeds on the plateau)")
    save(fig, "convergence_seed_comparison")


if __name__ == "__main__":
    fig_curves()
    fig_comparison()
    print("\nInspect assets/_preview_*.png.")
