"""
=============================================================================
ANALYSIS OF THE INDEPENDENT-SEED CONVERGENCE RUN (seeds 10/20/30/40/50)
=============================================================================
Aggregates results/convergence_curves_seeds10_50.csv exactly the way
make_paper_tables.py builds Table IX and make_figures.py builds Fig. 2:

  Table IX : groupby af -> mean val_loss(E1), mean val_acc(E1),
             std val_acc(E1) (ddof=1), sorted by accuracy descending.
  Fig. 2   : per-epoch mean +/- 1 sd of val_loss, restricted to epochs
             that every seed reached, so n is constant along the curve.

It then cross-checks the new seed set against the published one
(42/0/1/2/3). Because the two sets are disjoint, the Spearman correlation
between their epoch-1 rankings is the independent check the paper text
says the published means cannot provide.

Outputs: results/convergence_seeds10_50_epoch1.csv
         results/convergence_seeds10_50_by_epoch.csv
         results/convergence_seeds10_50_comparison.csv
         tables/tab9_convergence_seeds10_50.tex
=============================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind, mannwhitneyu

NEW = "results/convergence_curves_seeds10_50.csv"
OLD = "results/convergence_curves_5seed.csv"
SMOOTH_4 = ["Mish", "GELU", "SiLU", "Hardswish"]
RELU_3 = ["ReLU", "LeakyReLU", "SELU"]


def epoch1_table(d):
    e1 = d[d.epoch == 1]
    return (e1.groupby("af").agg(loss=("val_loss", "mean"),
                                 acc=("val_acc", "mean"),
                                 sd=("val_acc", "std"))
            .sort_values("acc", ascending=False))


def by_epoch(d):
    """mean +/- sd of val_loss per epoch, only epochs every seed reached."""
    n = d.seed.nunique()
    reached = d.groupby(["af", "epoch"]).seed.nunique()
    rows = []
    for af in sorted(d.af.unique()):
        g = d[d.af == af]
        for e in sorted(g.epoch.unique()):
            if reached.loc[(af, e)] != n:
                continue
            v = g[g.epoch == e].val_loss
            rows.append(dict(af=af, epoch=e, n_seeds=len(v),
                             mean_val_loss=round(v.mean(), 4),
                             sd_val_loss=round(v.std(ddof=1), 4),
                             mean_val_acc=round(g[g.epoch == e].val_acc.mean(), 2)))
    return pd.DataFrame(rows)


def fam(g):
    return g.loc[SMOOTH_4, "acc"].mean(), g.loc[RELU_3, "acc"].mean()


def latex(g, path):
    sm, rl = fam(g)
    L = [r"\begin{table}[htbp]",
         r"  \caption{Epoch-1 Convergence on BiGRU+GRU, Independent Seeds (10, 20, 30, 40, 50)}",
         r"  \label{tab:convergence_ind}",
         r"  \centering\footnotesize",
         r"  \renewcommand{\arraystretch}{1.05}",
         r"  \resizebox{\columnwidth}{!}{%",
         r"  \begin{tabular}{lccc}",
         r"    \toprule",
         r"    Activation & Val.\ loss (E1) & Val.\ acc.\ (E1)\% & $\sigma$ \\",
         r"    \midrule"]
    for f, r_ in g.iterrows():
        bold = r"\textbf{%s}" if f == g.index[0] else "%s"
        L.append(f"    {f} & {r_.loss:.3f} & {bold % f'{r_.acc:.2f}'} & {r_.sd:.2f} \\\\")
    L += [r"    \bottomrule",
          f"    \\multicolumn{{4}}{{l}}{{\\small Smooth family (Mish, GELU, SiLU, Hardswish) "
          f"averages {sm:.2f}\\%;}} \\\\",
          f"    \\multicolumn{{4}}{{l}}{{\\small the ReLU family (ReLU, LeakyReLU, SELU) "
          f"averages {rl:.2f}\\%.}} \\\\",
          r"  \end{tabular}%", r"  }", r"\end{table}"]
    os.makedirs("tables", exist_ok=True)
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


if __name__ == "__main__":
    new = pd.read_csv(NEW)
    gn = epoch1_table(new)
    en = by_epoch(new)
    gn.round(4).to_csv("results/convergence_seeds10_50_epoch1.csv")
    en.to_csv("results/convergence_seeds10_50_by_epoch.csv", index=False)
    latex(gn, "tables/tab9_convergence_seeds10_50.tex")

    sm, rl = fam(gn)
    print("=" * 74)
    print("EPOCH-1, SEEDS 10/20/30/40/50   (n = 5 per activation)")
    print("=" * 74)
    print(f"{'Activation':<11}{'loss E1':>9}{'acc E1 %':>10}{'sd':>8}   rank")
    for i, (f, r_) in enumerate(gn.iterrows(), 1):
        print(f"{f:<11}{r_.loss:>9.3f}{r_.acc:>10.2f}{r_.sd:>8.2f}{i:>7}")
    print("-" * 74)
    print(f"smooth family (Mish/GELU/SiLU/Hardswish) : {sm:.2f}%")
    print(f"ReLU family   (ReLU/LeakyReLU/SELU)      : {rl:.2f}%")
    print(f"gap                                      : {sm - rl:.2f} points")
    print(f"median sd across the ten functions       : {gn.sd.median():.2f} points")
    print(f"gap / median sd                          : {(sm - rl) / gn.sd.median():.2f}x")

    # per-seed family gap, so the gap itself carries an error bar
    e1 = new[new.epoch == 1]
    per_seed = []
    for s in sorted(e1.seed.unique()):
        q = e1[e1.seed == s].set_index("af").val_acc
        per_seed.append(q[SMOOTH_4].mean() - q[RELU_3].mean())
    per_seed = np.array(per_seed)
    print(f"per-seed gap                             : "
          f"{np.array2string(per_seed, precision=2)}  "
          f"mean {per_seed.mean():.2f} +/- {per_seed.std(ddof=1):.2f}")

    a = e1[e1.af.isin(SMOOTH_4)].val_acc.values
    b = e1[e1.af.isin(RELU_3)].val_acc.values
    t, pt = ttest_ind(a, b, equal_var=False)
    u, pu = mannwhitneyu(a, b, alternative="two-sided")
    print(f"smooth vs ReLU runs (n={len(a)} vs {len(b)})           : "
          f"Welch t={t:.2f}, p={pt:.2g};  Mann-Whitney p={pu:.2g}")

    print()
    print("=" * 74)
    print("CROSS-CHECK AGAINST THE PUBLISHED SEED SET (42/0/1/2/3)")
    print("=" * 74)
    old = pd.read_csv(OLD)
    go = epoch1_table(old)
    both = pd.DataFrame({"acc_new": gn.acc, "sd_new": gn.sd,
                         "acc_old": go.acc, "sd_old": go.sd})
    both["rank_new"] = both.acc_new.rank(ascending=False).astype(int)
    both["rank_old"] = both.acc_old.rank(ascending=False).astype(int)
    both["d_acc"] = both.acc_new - both.acc_old
    both = both.sort_values("acc_new", ascending=False)
    both.round(4).to_csv("results/convergence_seeds10_50_comparison.csv")
    print(f"{'Activation':<11}{'new %':>8}{'old %':>8}{'diff':>8}"
          f"{'r_new':>7}{'r_old':>7}")
    for f, r_ in both.iterrows():
        print(f"{f:<11}{r_.acc_new:>8.2f}{r_.acc_old:>8.2f}{r_.d_acc:>+8.2f}"
              f"{r_.rank_new:>7}{r_.rank_old:>7}")
    rho, p = spearmanr(both.acc_new, both.acc_old)
    print("-" * 74)
    print(f"Spearman rho (new 5 seeds vs published 5 seeds): {rho:.3f}  p={p:.4g}")

    # vs the single seed the original single-seed table was built on
    s42 = old[(old.epoch == 1) & (old.seed == 42)].set_index("af").val_acc
    r2, p2 = spearmanr(both.acc_new, s42.reindex(both.index))
    print(f"Spearman rho (new 5 seeds vs single seed 42)   : {r2:.3f}  p={p2:.4g}")

    smo, rlo = fam(go)
    print(f"family gap  published {smo - rlo:.2f} pts   independent {sm - rl:.2f} pts")

    print()
    print("=" * 74)
    print("MEAN VALIDATION LOSS BY EPOCH  (Fig. 2 data, seeds 10-50)")
    print("=" * 74)
    piv = en.pivot(index="af", columns="epoch", values="mean_val_loss")
    piv = piv.reindex(gn.index)
    print(piv.to_string(float_format=lambda x: f"{x:.4f}"))
    e_all = sorted(piv.columns[piv.notna().all()])
    print(f"\nepochs reached by all five seeds for every function: {e_all}")
    for e in e_all:
        col = piv[e].dropna()
        print(f"  epoch {e}: range {col.min():.3f}-{col.max():.3f}  "
              f"spread {col.max() - col.min():.3f}")
    print("\nwrote results/convergence_seeds10_50_{epoch1,by_epoch,comparison}.csv")
    print("      tables/tab9_convergence_seeds10_50.tex")
