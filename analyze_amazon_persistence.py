"""
=============================================================================
DOES THE IMDb TOP-10 RANKING PERSIST ON AMAZON POLARITY?
=============================================================================
Reads the 200-run cross-dataset sweep and answers three questions, in
increasing order of how much they actually matter:

  1. What are the mean and SD of test accuracy across the five seeds, per
     pair and per architecture? (the numbers that were asked for)

  2. Does the IMDb ordering of the ten pairs survive on Amazon? Spearman
     and Kendall on the ten, overall and within each architecture.

  3. Is the ordering separable at all on either corpus? A ranking can only
     "fail to persist" if it was real to begin with. We therefore compare
     the spread of the ten pair means against the seed noise underneath
     them, and run a one-way ANOVA and Kruskal-Wallis across the ten to
     test whether the pair identity explains any variance.

Inputs : amazon_polarity_results/top10_all_runs.csv   (this study)
         results/master_2000_runs.csv                 (IMDb, 2,000 runs)
Outputs: results/amazon_persistence_by_pair.csv
         results/amazon_persistence_by_pair_arch.csv
         results/amazon_persistence_comparison.csv
         tables/tab_amazon_persistence.tex
=============================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, f_oneway, kruskal

AMZ = "amazon_polarity_results/top10_all_runs.csv"
IMD = "results/master_2000_runs.csv"
ARCHS = ["BiLSTM+LSTM", "BiGRU+GRU", "BiLSTM+BiLSTM", "BiGRU+BiGRU"]


def load():
    a = pd.read_csv(AMZ)
    if "cell" not in a.columns:
        a["cell"] = a.act1 + "+" + a.act2
    i = pd.read_csv(IMD)
    pairs = sorted(a.cell.unique())
    i10 = i[i.cell.isin(pairs)].copy()
    return a, i, i10


def per_pair(d):
    g = (d.groupby("cell").accuracy
         .agg(mean="mean", sd="std", n="count")
         .sort_values("mean", ascending=False))
    g["se"] = g["sd"] / np.sqrt(g["n"])
    return g


def per_pair_arch(d):
    """mean +/- SD across the five seeds, the unit that was asked for."""
    g = (d.groupby(["experiment", "cell"]).accuracy
         .agg(mean="mean", sd="std", n="count").reset_index())
    return g


def sep_report(d, label):
    """Is the ten-pair ordering separable above seed noise?"""
    g = per_pair(d)
    span = g["mean"].max() - g["mean"].min()
    groups = [x.accuracy.values for _, x in d.groupby("cell")]
    F, pF = f_oneway(*groups)
    H, pH = kruskal(*groups)
    print(f"  {label}")
    print(f"    span of the ten pair means : {span:.3f} pts")
    print(f"    median SE of a pair mean   : {g.se.median():.3f} pts")
    print(f"    span / median SE           : {span / g.se.median():.2f}x")
    print(f"    within-pair seed SD (med)  : {g.sd.median():.3f} pts")
    print(f"    one-way ANOVA across 10    : F={F:.2f}, p={pF:.3g}")
    print(f"    Kruskal-Wallis across 10   : H={H:.2f}, p={pH:.3g}")
    return dict(label=label, span=span, med_se=g.se.median(),
                ratio=span / g.se.median(), anova_F=F, anova_p=pF,
                kruskal_H=H, kruskal_p=pH)


def latex(cmp_df, path):
    L = [r"\begin{table}[htbp]",
         r"  \caption{Cross-Dataset Persistence of the IMDb Top-10 Activation "
         r"Pairs. Mean test accuracy $\pm$ SD over five seeds, pooled across "
         r"the four architectures ($n=20$).}",
         r"  \label{tab:amazon_persistence}",
         r"  \centering\footnotesize",
         r"  \renewcommand{\arraystretch}{1.05}",
         r"  \resizebox{\columnwidth}{!}{%",
         r"  \begin{tabular}{lccccc}",
         r"    \toprule",
         r"    Pair & \multicolumn{2}{c}{IMDb} & \multicolumn{2}{c}{Amazon} "
         r"& $\Delta$rank \\",
         r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"    & Acc.\ \% & rank & Acc.\ \% & rank & \\",
         r"    \midrule"]
    for c, r_ in cmp_df.iterrows():
        L.append(f"    {c.replace('+', '+')} & {r_.imdb_mean:.2f} & "
                 f"{int(r_.imdb_rank)} & {r_.amz_mean:.2f} & "
                 f"{int(r_.amz_rank)} & {int(r_.d_rank):+d} \\\\")
    L += [r"    \bottomrule", r"  \end{tabular}%", r"  }", r"\end{table}"]
    os.makedirs("tables", exist_ok=True)
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


if __name__ == "__main__":
    amz, imdb_all, imdb = load()
    W = 78

    print("=" * W)
    print("1.  MEAN TEST ACCURACY +/- SD OVER FIVE SEEDS  (Amazon Polarity)")
    print("=" * W)
    pa = per_pair_arch(amz)
    piv_m = pa.pivot(index="cell", columns="experiment", values="mean")
    piv_s = pa.pivot(index="cell", columns="experiment", values="sd")
    gA = per_pair(amz)
    order = gA.index
    cols = [c for c in ARCHS if c in piv_m.columns]
    hdr = "".join(f"{c:>18}" for c in cols)
    print(f"{'pair':<20}{hdr}{'pooled':>16}")
    for c in order:
        row = "".join(f"{piv_m.loc[c, a]:>11.2f}±{piv_s.loc[c, a]:<6.2f}"
                      for a in cols)
        print(f"{c:<20}{row}{gA.loc[c,'mean']:>9.2f}±{gA.loc[c,'sd']:<6.2f}")
    print(f"\n  runs: {len(amz)}   pairs: {amz.cell.nunique()}   "
          f"archs: {amz.experiment.nunique()}   seeds: {sorted(amz.seed.unique())}")

    print()
    print("=" * W)
    print("2.  DOES THE IMDb ORDERING PERSIST?")
    print("=" * W)
    gI = per_pair(imdb)
    cmp_df = pd.DataFrame({
        "imdb_mean": gI["mean"], "imdb_sd": gI["sd"],
        "amz_mean": gA["mean"], "amz_sd": gA["sd"]})
    cmp_df["imdb_rank"] = cmp_df.imdb_mean.rank(ascending=False)
    cmp_df["amz_rank"] = cmp_df.amz_mean.rank(ascending=False)
    cmp_df["d_rank"] = cmp_df.imdb_rank - cmp_df.amz_rank
    cmp_df["d_acc"] = cmp_df.amz_mean - cmp_df.imdb_mean
    cmp_df = cmp_df.sort_values("imdb_rank")

    print(f"{'pair':<20}{'IMDb %':>9}{'rank':>6}{'Amazon %':>11}{'rank':>6}"
          f"{'d rank':>8}{'d acc':>8}")
    for c, r_ in cmp_df.iterrows():
        print(f"{c:<20}{r_.imdb_mean:>9.2f}{int(r_.imdb_rank):>6}"
              f"{r_.amz_mean:>11.2f}{int(r_.amz_rank):>6}"
              f"{int(r_.d_rank):>+8d}{r_.d_acc:>+8.2f}")

    rho, prho = spearmanr(cmp_df.imdb_mean, cmp_df.amz_mean)
    tau, ptau = kendalltau(cmp_df.imdb_mean, cmp_df.amz_mean)
    print("-" * W)
    print(f"  Spearman rho = {rho:+.3f}  p = {prho:.4g}")
    print(f"  Kendall  tau = {tau:+.3f}  p = {ptau:.4g}")
    print(f"  mean |rank shift| = {cmp_df.d_rank.abs().mean():.1f} places "
          f"(max {int(cmp_df.d_rank.abs().max())})")
    print(f"  IMDb best pair ({cmp_df.index[0]}) lands at rank "
          f"{int(cmp_df.amz_rank.iloc[0])} on Amazon")
    print(f"  accuracy level: IMDb {cmp_df.imdb_mean.mean():.2f}%  ->  "
          f"Amazon {cmp_df.amz_mean.mean():.2f}%  "
          f"({cmp_df.d_acc.mean():+.2f} pts)")

    print("\n  per-architecture Spearman (IMDb vs Amazon, same 10 pairs):")
    for a in ARCHS:
        ia = imdb[imdb.experiment == a].groupby("cell").accuracy.mean()
        aa = amz[amz.experiment == a].groupby("cell").accuracy.mean()
        common = ia.index.intersection(aa.index)
        r_, p_ = spearmanr(ia[common], aa[common])
        print(f"    {a:<16} rho = {r_:+.3f}   p = {p_:.3g}")

    print()
    print("=" * W)
    print("3.  WAS THE ORDERING SEPARABLE IN THE FIRST PLACE?")
    print("=" * W)
    rows = [sep_report(imdb, "IMDb   (top-10 subset, n=20 per pair)"),
            sep_report(amz, "Amazon (top-10 subset, n=20 per pair)")]

    gAll = per_pair(imdb_all)
    print(f"\n  for scale, all 100 IMDb pairs: span "
          f"{gAll['mean'].max()-gAll['mean'].min():.3f} pts, "
          f"median SE {gAll.se.median():.3f} pts "
          f"({(gAll['mean'].max()-gAll['mean'].min())/gAll.se.median():.1f}x)")

    os.makedirs("results", exist_ok=True)
    pa.round(4).to_csv("results/amazon_persistence_by_pair_arch.csv", index=False)
    gA.round(4).to_csv("results/amazon_persistence_by_pair.csv")
    cmp_df.round(4).to_csv("results/amazon_persistence_comparison.csv")
    pd.DataFrame(rows).round(4).to_csv(
        "results/amazon_persistence_separability.csv", index=False)
    latex(cmp_df, "tables/tab_amazon_persistence.tex")
    print("\nwrote results/amazon_persistence_{by_pair,by_pair_arch,"
          "comparison,separability}.csv")
    print("      tables/tab_amazon_persistence.tex")
