"""
=============================================================================
CAMERA-READY TABLE GENERATOR  (iCONEECT 2026)
=============================================================================
Emits every numeric table in the paper as a LaTeX fragment under tables/,
computed directly from the 2000-run dataset. Nothing in the paper is typed by
hand, so every printed value traces to a CSV row.

Inputs
  <ARCH>_5_seed_variance_results/all_runs.csv        (500 rows each)
  <ARCH>_5_seed_variance_results/mean_100_cells.csv  (100 rows each)
  results/convergence_seeds.csv                      (50 rows, epoch-1 study)
  results/<original single-seed sweep CSVs>          (for the reproducibility
                                                      comparison only)

Outputs (tables/)
  tab4_architecture.tex     cross-experiment accuracy, 5-seed
  tab5_variance.tex         variance decomposition            [NEW, headline]
  tab6_top10.tex            global top-10 with std and 95% CI [reviewer ask]
  tab7_marginal.tex         marginal means per activation
  tab8_reproducibility.tex  single-seed vs 5-seed agreement   [NEW]
  tab9_convergence.tex      epoch-1 convergence, 5-seed
  numbers.txt               every scalar quoted in the prose
=============================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

ARCH = ["BiLSTM+LSTM", "BiGRU+GRU", "BiLSTM+BiLSTM", "BiGRU+BiGRU"]
OLD = {"BiLSTM+LSTM": "results/result_activation_sweep/result_activation_sweep.csv",
       "BiGRU+GRU": "results/GRU_Results/result_gru_activation_sweep.csv",
       "BiLSTM+BiLSTM": "results/BiLSTM_Results/result_fullbilstm_sweep.csv",
       "BiGRU+BiGRU": "results/BiGRU_Results/result_fullbigru_sweep.csv"}
ACTS = ["ReLU", "ELU", "LeakyReLU", "Tanh", "GELU", "SiLU", "Mish", "SELU",
        "PReLU", "Hardswish"]
OUT = "tables"
os.makedirs(OUT, exist_ok=True)

R = {a: pd.read_csv(f"{a}_5_seed_variance_results/all_runs.csv") for a in ARCH}
M = {a: pd.read_csv(f"{a}_5_seed_variance_results/mean_100_cells.csv") for a in ARCH}
O = {a: pd.read_csv(p).set_index(["Act1", "Act2"]) for a, p in OLD.items()}
ALL = pd.concat([R[a] for a in ARCH], ignore_index=True)
CELLS = pd.concat([M[a].assign(arch=a) for a in ARCH], ignore_index=True)
CELLS = CELLS.sort_values("acc_mean", ascending=False).reset_index(drop=True)

N = {}          # every scalar the prose quotes


def w(name, lines):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  {OUT}/{name}")


def ci95(v):
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))


# ─────────────────────────────────────────────────────────────────────────────
# TABLE IV -- cross-experiment accuracy
# ─────────────────────────────────────────────────────────────────────────────
L = [r"\begin{table}[htbp]",
     r"  \caption{Cross-Experiment Accuracy over 2{,}000 Runs (500 per Experiment)}",
     r"  \label{tab:sweep_summary}",
     r"  \centering\footnotesize",
     r"  \setlength{\tabcolsep}{3pt}",
     r"  \renewcommand{\arraystretch}{1.1}",
     r"  \resizebox{\columnwidth}{!}{%",
     r"  \begin{tabular}{lccccc}",
     r"    \toprule",
     r"    Experiment & Min\% & Max\% & Mean\% & 95\% CI & $\sigma_{\text{cell}}$ \\",
     r"    \midrule"]
for a in ARCH:
    v, m = R[a].accuracy, M[a]
    L.append(f"    {a} & {m.acc_mean.min():.2f} & {m.acc_mean.max():.2f} & "
             f"\\textbf{{{v.mean():.2f}}} & $\\pm${ci95(v):.2f} & {m.acc_std.median():.2f} \\\\")
    N[f"arch_mean_{a}"] = round(v.mean(), 2)
    N[f"arch_ci_{a}"] = round(ci95(v), 2)
    N[f"arch_std_{a}"] = round(m.acc_std.median(), 2)
L += [r"    \midrule",
      r"    \multicolumn{3}{l}{\textbf{Benchmark}~\cite{ahmed2025}} & \textbf{88.40} & --- & --- \\",
      r"    \bottomrule",
      r"    \multicolumn{6}{l}{\small Min/Max are over the 100 cell means; "
      r"$\sigma_{\text{cell}}$ is the median} \\",
      r"    \multicolumn{6}{l}{\small per-cell standard deviation across the five seeds.} \\",
      r"  \end{tabular}%", r"  }", r"\end{table}"]
w("tab4_architecture.tex", L)

# ─────────────────────────────────────────────────────────────────────────────
# TABLE V -- variance decomposition   [the headline result]
# ─────────────────────────────────────────────────────────────────────────────
d = ALL.copy()
d["family"] = np.where(d.experiment.str.contains("GRU"), "GRU", "LSTM")
d["bidir2"] = d.experiment.isin(["BiLSTM+BiLSTM", "BiGRU+BiGRU"])
gm = d.accuracy.mean()
tss = ((d.accuracy - gm) ** 2).sum()
def ss(cols):
    return (d.groupby(cols).accuracy.transform("mean") - gm).pow(2).sum()
rows = [("Architecture (4 levels)", ss(["experiment"])),
        ("\\quad cell type (GRU vs.\\ LSTM)", ss(["family"])),
        ("\\quad layer-2 bidirectionality", ss(["bidir2"])),
        ("Act$_1$ (10 levels)", ss(["act1"])),
        ("Act$_2$ (10 levels)", ss(["act2"])),
        ("Act$_1\\times$Act$_2$ pair (100 levels)", ss(["act1", "act2"]))]
full = ss(["experiment", "act1", "act2"])
L = [r"\begin{table}[htbp]",
     r"  \caption{Variance Decomposition of Test Accuracy (2{,}000 Runs)}",
     r"  \label{tab:variance}",
     r"  \centering\footnotesize",
     r"  \setlength{\tabcolsep}{4pt}",
     r"  \renewcommand{\arraystretch}{1.08}",
     r"  \begin{tabular}{lr}",
     r"    \toprule",
     r"    Source of variation & \% of total variance \\",
     r"    \midrule"]
for lbl, v in rows:
    L.append(f"    {lbl} & {100*v/tss:.2f} \\\\")
    N["var_" + lbl.split("(")[0].replace("\\quad ", "").strip()] = round(100 * v / tss, 2)
L += [r"    \midrule",
      f"    \\textbf{{Seed (residual)}} & \\textbf{{{100*(tss-full)/tss:.2f}}} \\\\",
      r"    \bottomrule",
      r"  \end{tabular}",
      r"\end{table}"]
N["var_seed"] = round(100 * (tss - full) / tss, 2)
N["var_arch"] = round(100 * ss(["experiment"]) / tss, 2)
N["var_celltype"] = round(100 * ss(["family"]) / tss, 2)
N["var_bidir"] = round(100 * ss(["bidir2"]) / tss, 2)
N["var_pair"] = round(100 * ss(["act1", "act2"]) / tss, 2)
w("tab5_variance.tex", L)

# ─────────────────────────────────────────────────────────────────────────────
# TABLE VI -- global top 10 with std and CI   [Reviewer 3, point 2]
# ─────────────────────────────────────────────────────────────────────────────
t = CELLS.head(10)
L = [r"\begin{table}[htbp]",
     r"  \caption{Ten Highest-Accuracy Configurations of 400, with Seed Variability}",
     r"  \label{tab:top_configs}",
     r"  \centering\footnotesize",
     r"  \renewcommand{\arraystretch}{1.1}",
     r"  \resizebox{\columnwidth}{!}{%",
     r"  \begin{tabular}{llccc}",
     r"    \toprule",
     r"    Act$_1$ + Act$_2$ & Experiment & Acc.\% & 95\% CI & F1\% \\",
     r"     &  & mean $\pm$ $\sigma$ & $\pm$ & mean \\",
     r"    \midrule"]
for x in t.itertuples():
    L.append(f"    {x.act1} + {x.act2} & {x.arch} & "
             f"{x.acc_mean:.2f} $\\pm$ {x.acc_std:.2f} & {x.acc_ci95:.2f} & {x.f1_mean:.2f} \\\\")
L += [r"    \bottomrule",
      r"    \multicolumn{5}{l}{\small All ten are GRU-based. The ten means span "
      f"{t.acc_mean.max()-t.acc_mean.min():.2f} points, less than}} \\\\".replace("}}", "}"),
      r"    \multicolumn{5}{l}{\small the widest half-interval ($\pm$"
      f"{t.acc_ci95.max():.2f}); every pair of intervals overlaps.}} \\\\".replace("}}", "}"),
      r"  \end{tabular}%", r"  }", r"\end{table}"]
w("tab6_top10.tex", L)
N["top10_span"] = round(t.acc_mean.max() - t.acc_mean.min(), 2)
N["top10_widest_ci"] = round(t.acc_ci95.max(), 2)
N["top10_best"] = f"{t.iloc[0].act1} + {t.iloc[0].act2}"
N["top10_best_mean"] = round(t.iloc[0].acc_mean, 2)
N["top10_best_std"] = round(t.iloc[0].acc_std, 2)
N["top10_best_arch"] = t.iloc[0].arch
lo = (t.acc_mean - t.acc_ci95).max(); hi = (t.acc_mean + t.acc_ci95).min()
N["top10_common_overlap"] = bool(lo <= hi)
N["top50_gru"] = int(CELLS.head(50).arch.str.contains("GRU").sum())

# ─────────────────────────────────────────────────────────────────────────────
# TABLE VII -- marginal means
# ─────────────────────────────────────────────────────────────────────────────
L = [r"\begin{table}[htbp]",
     r"  \caption{Marginal Mean Accuracy (\%) per Activation Function, 5-Seed}",
     r"  \label{tab:marginal}",
     r"  \centering\scriptsize",
     r"  \setlength{\tabcolsep}{2pt}",
     r"  \renewcommand{\arraystretch}{1.08}",
     r"  \resizebox{\columnwidth}{!}{%",
     r"  \begin{tabular}{l cc c cc c cc c cc}",
     r"    \toprule",
     r"    & \multicolumn{2}{c}{BiLSTM+LSTM} &",
     r"    & \multicolumn{2}{c}{BiGRU+GRU} &",
     r"    & \multicolumn{2}{c}{BiLSTM+BiLSTM} &",
     r"    & \multicolumn{2}{c}{BiGRU+BiGRU} \\",
     r"    \cmidrule{2-3} \cmidrule{5-6} \cmidrule{8-9} \cmidrule{11-12}",
     r"    Function & Act$_1$ & Act$_2$ && Act$_1$ & Act$_2$ && Act$_1$ & Act$_2$ "
     r"&& Act$_1$ & Act$_2$ \\",
     r"    \midrule"]
mg = {a: (R[a].groupby("act1").accuracy.mean(), R[a].groupby("act2").accuracy.mean())
      for a in ARCH}
for f in ACTS:
    cells = []
    for a in ARCH:
        cells += [f"{mg[a][0][f]:.2f}", f"{mg[a][1][f]:.2f}"]
    L.append(f"    {f} & {cells[0]} & {cells[1]} && {cells[2]} & {cells[3]} && "
             f"{cells[4]} & {cells[5]} && {cells[6]} & {cells[7]} \\\\")
sp = max(mg[a][0].max() - mg[a][0].min() for a in ARCH)
L += [r"    \bottomrule",
      r"    \multicolumn{12}{l}{\small Each entry averages 50 runs (10 partners "
      r"$\times$ 5 seeds). The widest} \\",
      f"    \\multicolumn{{12}}{{l}}{{\\small spread within any column is {sp:.2f} points.}} \\\\",
      r"  \end{tabular}%", r"  }", r"\end{table}"]
w("tab7_marginal.tex", L)
N["marginal_widest_spread"] = round(sp, 2)

# ─────────────────────────────────────────────────────────────────────────────
# TABLE VIII -- reproducibility of the single-seed ranking   [NEW]
# ─────────────────────────────────────────────────────────────────────────────
L = [r"\begin{table}[htbp]",
     r"  \caption{Agreement Between the Single-Seed Ranking and the 5-Seed Means}",
     r"  \label{tab:reproducibility}",
     r"  \centering\footnotesize",
     r"  \setlength{\tabcolsep}{3pt}",
     r"  \renewcommand{\arraystretch}{1.1}",
     r"  \resizebox{\columnwidth}{!}{%",
     r"  \begin{tabular}{lcccc}",
     r"    \toprule",
     r"    Experiment & $\rho$ & Mean rank $\Delta$ & Top-10 overlap & "
     r"Mean $|\Delta$acc$|$ \\",
     r"    \midrule"]
for a in ARCH:
    m = M[a].copy()
    m["single"] = [float(O[a].loc[(x.act1, x.act2), "Accuracy"]) for x in m.itertuples()]
    ro = m.single.rank(ascending=False, method="min")
    rn = m.acc_mean.rank(ascending=False, method="min")
    pt = set(O[a].Accuracy.nlargest(10).index)
    nt = {(x.act1, x.act2) for x in m.nlargest(10, "acc_mean").itertuples()}
    rho = stats.spearmanr(m.single, m.acc_mean)[0]
    L.append(f"    {a} & ${rho:+.3f}$ & {(ro-rn).abs().mean():.1f} & "
             f"{len(pt&nt)}/10 & {(m.acc_mean-m.single).abs().mean():.2f} \\\\")
    N[f"rho_{a}"] = round(rho, 3)
    N[f"rankchg_{a}"] = round((ro - rn).abs().mean(), 1)
    N[f"overlap_{a}"] = len(pt & nt)
L += [r"    \bottomrule",
      r"    \multicolumn{5}{l}{\small $\rho$ is the Spearman correlation between the "
      r"100 single-seed cell} \\",
      r"    \multicolumn{5}{l}{\small accuracies and the corresponding 5-seed means. "
      r"$\rho\!\approx\!0$ in all four.} \\",
      r"  \end{tabular}%", r"  }", r"\end{table}"]
w("tab8_reproducibility.tex", L)

# ─────────────────────────────────────────────────────────────────────────────
# TABLE IX -- convergence
# ─────────────────────────────────────────────────────────────────────────────
c = pd.read_csv("results/convergence_seeds.csv")
g = c.groupby("af").agg(loss=("val_loss_e1", "mean"), acc=("val_acc_e1", "mean"),
                        sd=("val_acc_e1", "std")).sort_values("acc", ascending=False)
L = [r"\begin{table}[htbp]",
     r"  \caption{Epoch-1 Convergence on BiGRU+GRU, Five Seeds}",
     r"  \label{tab:convergence}",
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
sm = g.loc[["Mish", "GELU", "SiLU", "Hardswish"], "acc"].mean()
rl = g.loc[["ReLU", "LeakyReLU", "SELU"], "acc"].mean()
L += [r"    \bottomrule",
      f"    \\multicolumn{{4}}{{l}}{{\\small Smooth family (Mish, GELU, SiLU, Hardswish) "
      f"averages {sm:.2f}\\%;}} \\\\",
      f"    \\multicolumn{{4}}{{l}}{{\\small the ReLU family (ReLU, LeakyReLU, SELU) "
      f"averages {rl:.2f}\\%.}} \\\\",
      r"  \end{tabular}%", r"  }", r"\end{table}"]
w("tab9_convergence.tex", L)
N["conv_smooth"] = round(sm, 2)
N["conv_relu"] = round(rl, 2)
N["conv_gap"] = round(sm - rl, 2)
N["conv_sd"] = round(g.sd.median(), 2)
N["conv_ratio"] = round((sm - rl) / g.sd.median(), 1)
N["conv_rho"] = round(stats.spearmanr(
    g.acc, pd.read_csv("results/convergence/convergence_per_af.csv")
    .set_index("AF").loc[g.index, "val_acc_e1"])[0], 3)

# ─────────────────────────────────────────────────────────────────────────────
# PROSE NUMBERS
# ─────────────────────────────────────────────────────────────────────────────
G = ALL[ALL.experiment.str.contains("GRU")].accuracy
Lf = ALL[~ALL.experiment.str.contains("GRU")].accuracy
t_, p_ = stats.ttest_ind(G, Lf, equal_var=False)
sp_ = np.sqrt(((len(G)-1)*G.var(ddof=1) + (len(Lf)-1)*Lf.var(ddof=1)) / (len(G)+len(Lf)-2))
N.update(dict(
    n_runs=len(ALL), n_cells=len(CELLS),
    gru_mean=round(G.mean(), 2), gru_ci=round(ci95(G), 2),
    lstm_mean=round(Lf.mean(), 2), lstm_ci=round(ci95(Lf), 2),
    gru_gap=round(G.mean()-Lf.mean(), 2), gru_t=round(t_, 1), gru_p=f"{p_:.0e}",
    gru_d=round((G.mean()-Lf.mean())/sp_, 2),
    std_gru=round(pd.concat([M["BiGRU+GRU"], M["BiGRU+BiGRU"]]).acc_std.median(), 2),
    std_lstm=round(pd.concat([M["BiLSTM+LSTM"], M["BiLSTM+BiLSTM"]]).acc_std.median(), 2),
    std_p=f"{stats.ttest_ind(pd.concat([M['BiLSTM+LSTM'],M['BiLSTM+BiLSTM']]).acc_std, pd.concat([M['BiGRU+GRU'],M['BiGRU+BiGRU']]).acc_std, equal_var=False)[1]:.0e}",
    cells_over_benchmark=int((CELLS.acc_mean >= 88.40).sum()),
    runs_over_benchmark=int((ALL.accuracy >= 88.40).sum()),
    max_run=round(ALL.accuracy.max(), 2),
    best_ci_upper=round(CELLS.iloc[0].acc_mean + CELLS.iloc[0].acc_ci95, 2),
    failures=int((ALL.accuracy < 60).sum()),
    unstable_cells=int((CELLS.acc_std > 1.0).sum()),
    unstable_all_lstm=bool((CELLS[CELLS.acc_std > 1.0].arch.str.contains("LSTM")).all()),
    median_cell_std=round(CELLS.acc_std.median(), 2),
    seeds_for_arch=int(np.ceil(2*CELLS.acc_std.median()**2*(1.96+0.84)**2/0.79**2)),
    seeds_for_top10=int(np.ceil(2*CELLS.acc_std.median()**2*(1.96+0.84)**2/max(N["top10_span"],0.01)**2)),
))
for pair, lbl in [(("BiLSTM+LSTM", "BiLSTM+BiLSTM"), "lstm"), (("BiGRU+GRU", "BiGRU+BiGRU"), "gru")]:
    a, b = pair
    t2, p2 = stats.ttest_ind(R[b].accuracy, R[a].accuracy, equal_var=False)
    N[f"bidir_{lbl}_delta"] = round(R[b].accuracy.mean() - R[a].accuracy.mean(), 2)
    N[f"bidir_{lbl}_p"] = round(p2, 3)
rhos = []
mar = {a: R[a].groupby("act1").accuracy.mean() for a in ARCH}
for i in range(4):
    for j in range(i+1, 4):
        rhos.append(stats.spearmanr([mar[ARCH[i]][f] for f in ACTS],
                                    [mar[ARCH[j]][f] for f in ACTS]))
N["cross_rho_min"] = round(min(r[0] for r in rhos), 3)
N["cross_rho_max"] = round(max(r[0] for r in rhos), 3)
N["cross_rho_minp"] = round(min(r[1] for r in rhos), 3)

with open(os.path.join(OUT, "numbers.txt"), "w", encoding="utf-8") as fh:
    for k, v in N.items():
        fh.write(f"{k:34s} {v}\n")
print(f"  {OUT}/numbers.txt  ({len(N)} values)")
print("\nKey values:")
for k in ["n_runs", "var_seed", "var_arch", "var_pair", "gru_gap", "gru_p", "gru_d",
          "top10_span", "top10_widest_ci", "top50_gru", "runs_over_benchmark",
          "max_run", "conv_gap", "conv_ratio", "seeds_for_top10"]:
    print(f"  {k:24s} {N[k]}")
