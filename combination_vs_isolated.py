"""
=============================================================================
COMBINATION (heterogeneous) vs ISOLATED (homogeneous) ACTIVATION ANALYSIS
=============================================================================
Question (mentor / supervisor):
   For each recurrent architecture (GRU, BiGRU, BiLSTM) does sentiment
   accuracy improve more when the two dense-head activations are MIXED
   (a "combination", Act1 != Act2) or when they are the SAME activation
   used in isolation (Act1 == Act2)?  And which architecture wins overall?

Inputs (already produced by the activation sweeps; full 10x10 grids):
   results/GRU_Results/result_gru_activation_sweep.csv
   results/BiGRU_Results/result_fullbigru_sweep.csv
   results/BiLSTM_Results/result_fullbilstm_sweep.csv

Outputs (results/comparison/):
   combination_vs_isolated_summary.csv   per-architecture stats + winner
   per_activation_isolated_vs_partner.csv
   01_combo_vs_isolated_bars.png         mean acc, isolated vs combination
   02_distribution_box.png               full distribution (box + points)
   03_best_pairs.png                     best isolated vs best combination
   04_per_activation_isolated_vs_partner.png
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")

OUT = os.path.join("results", "comparison")
os.makedirs(OUT, exist_ok=True)

# colours reused from the existing sweep figures
C_ISO   = "#3498DB"   # isolated  (blue)
C_COMBO = "#E74C3C"   # combination (red)
ARCH_C  = {"GRU": "#2ECC71", "BiGRU": "#E67E22", "BiLSTM": "#9B59B6"}

SOURCES = {
    "GRU"   : "results/GRU_Results/result_gru_activation_sweep.csv",
    "BiGRU" : "results/BiGRU_Results/result_fullbigru_sweep.csv",
    "BiLSTM": "results/BiLSTM_Results/result_fullbilstm_sweep.csv",
}

# ── load + split each grid into isolated (diagonal) / combination (off-diag) ──
frames = {}
for arch, path in SOURCES.items():
    d = pd.read_csv(path)
    d["Arch"]  = arch
    d["Group"] = np.where(d.Act1 == d.Act2, "Isolated", "Combination")
    frames[arch] = d
all_df = pd.concat(frames.values(), ignore_index=True)

# ── per-architecture summary ────────────────────────────────────────────────
rows = []
for arch, d in frames.items():
    iso   = d[d.Group == "Isolated"]["Accuracy"]
    combo = d[d.Group == "Combination"]["Accuracy"]
    t, p  = stats.ttest_ind(combo, iso, equal_var=False)

    best_iso   = d[d.Group == "Isolated"].sort_values("Accuracy").iloc[-1]
    best_combo = d[d.Group == "Combination"].sort_values("Accuracy").iloc[-1]
    best_overall = d.sort_values("Accuracy").iloc[-1]

    winner = "Combination" if combo.mean() > iso.mean() else "Isolated"
    rows.append({
        "Architecture"      : arch,
        "iso_mean"          : round(iso.mean(), 3),
        "iso_std"           : round(iso.std(), 3),
        "iso_max"           : round(iso.max(), 2),
        "combo_mean"        : round(combo.mean(), 3),
        "combo_std"         : round(combo.std(), 3),
        "combo_max"         : round(combo.max(), 2),
        "combo_minus_iso"   : round(combo.mean() - iso.mean(), 3),
        "winner_on_mean"    : winner,
        "welch_t"           : round(t, 3),
        "p_value"           : round(p, 4),
        "significant_0.05"  : "yes" if p < 0.05 else "no",
        "best_isolated"     : f"{best_iso.Act1} ({best_iso.Accuracy:.2f}%)",
        "best_combination"  : f"{best_combo.Act1}+{best_combo.Act2} ({best_combo.Accuracy:.2f}%)",
        "best_overall"      : f"{best_overall.Act1}+{best_overall.Act2} ({best_overall.Accuracy:.2f}%)",
    })
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT, "combination_vs_isolated_summary.csv"), index=False)

# ── per-activation: isolated accuracy vs mean accuracy as a partner ─────────
acts = sorted(all_df.Act1.unique())
pa_rows = []
for arch, d in frames.items():
    for a in acts:
        iso_acc = d[(d.Act1 == a) & (d.Act2 == a)]["Accuracy"].values
        iso_acc = iso_acc[0] if len(iso_acc) else np.nan
        # mean accuracy of every heterogeneous pair that INCLUDES this AF
        partner = d[((d.Act1 == a) | (d.Act2 == a)) & (d.Act1 != d.Act2)]["Accuracy"].mean()
        pa_rows.append({"Architecture": arch, "Activation": a,
                        "isolated_acc": round(iso_acc, 2),
                        "mean_as_partner": round(partner, 2),
                        "combo_gain": round(partner - iso_acc, 2)})
per_act = pd.DataFrame(pa_rows)
per_act.to_csv(os.path.join(OUT, "per_activation_isolated_vs_partner.csv"), index=False)

# ============================================================================
# FIGURE 1 — mean accuracy: isolated vs combination (grouped bars + errorbars)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
archs = list(SOURCES.keys())
x = np.arange(len(archs)); w = 0.36
iso_m   = [summary.set_index("Architecture").loc[a, "iso_mean"]   for a in archs]
iso_s   = [summary.set_index("Architecture").loc[a, "iso_std"]    for a in archs]
combo_m = [summary.set_index("Architecture").loc[a, "combo_mean"] for a in archs]
combo_s = [summary.set_index("Architecture").loc[a, "combo_std"]  for a in archs]

b1 = ax.bar(x - w/2, iso_m,   w, yerr=iso_s,   capsize=5, color=C_ISO,
            alpha=0.9, edgecolor="white", label="Isolated (same AF, Act1=Act2)")
b2 = ax.bar(x + w/2, combo_m, w, yerr=combo_s, capsize=5, color=C_COMBO,
            alpha=0.9, edgecolor="white", label="Combination (mixed AF, Act1≠Act2)")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(archs, fontsize=12)
ax.set_ylabel("Mean test accuracy (%)", fontsize=12)
ax.set_ylim(min(iso_m + combo_m) - 0.6, max(iso_m + combo_m) + 0.6)
ax.set_title("Isolated vs Combination activations — mean accuracy per architecture\n"
             "(error bars = std across pairs)", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_combo_vs_isolated_bars.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# FIGURE 2 — full distribution (box + jittered points)
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 6))
sns.boxplot(data=all_df, x="Arch", y="Accuracy", hue="Group",
            order=archs, hue_order=["Isolated", "Combination"],
            palette={"Isolated": C_ISO, "Combination": C_COMBO},
            width=0.6, fliersize=0, ax=ax)
sns.stripplot(data=all_df, x="Arch", y="Accuracy", hue="Group",
              order=archs, hue_order=["Isolated", "Combination"],
              palette={"Isolated": "#1B4F72", "Combination": "#7B241C"},
              dodge=True, alpha=0.5, size=3, ax=ax, legend=False)
ax.set_xlabel(""); ax.set_ylabel("Test accuracy (%)", fontsize=12)
ax.set_title("Accuracy distribution: isolated vs combination activations", fontsize=13)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title="", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_distribution_box.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# FIGURE 3 — best isolated vs best combination vs best overall per architecture
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
si = summary.set_index("Architecture")
best_iso_v   = [si.loc[a, "iso_max"]   for a in archs]
best_combo_v = [si.loc[a, "combo_max"] for a in archs]
b1 = ax.bar(x - w/2, best_iso_v,   w, color=C_ISO,   alpha=0.9, edgecolor="white",
            label="Best isolated AF")
b2 = ax.bar(x + w/2, best_combo_v, w, color=C_COMBO, alpha=0.9, edgecolor="white",
            label="Best combination pair")
for a, xi in zip(archs, x):
    ax.text(xi - w/2, best_iso_v[archs.index(a)] + 0.02,
            si.loc[a, "best_isolated"].split(" ")[0], ha="center", va="bottom", fontsize=8)
    ax.text(xi + w/2, best_combo_v[archs.index(a)] + 0.02,
            si.loc[a, "best_combination"].split(" ")[0], ha="center", va="bottom", fontsize=8)
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.12,
                f"{bar.get_height():.2f}", ha="center", va="top", fontsize=9, color="white")
ax.set_xticks(x); ax.set_xticklabels(archs, fontsize=12)
ax.set_ylabel("Best test accuracy (%)", fontsize=12)
ax.set_ylim(min(best_iso_v + best_combo_v) - 0.6, max(best_iso_v + best_combo_v) + 0.5)
ax.set_title("Peak accuracy: best isolated AF vs best mixed combination", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_best_pairs.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# FIGURE 4 — per-activation isolated vs mean-as-partner (one panel / arch)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax, arch in zip(axes, archs):
    d = per_act[per_act.Architecture == arch].sort_values("isolated_acc")
    xa = np.arange(len(d))
    ax.bar(xa - 0.2, d.isolated_acc,    0.4, color=C_ISO,   alpha=0.9,
           edgecolor="white", label="Isolated")
    ax.bar(xa + 0.2, d.mean_as_partner, 0.4, color=C_COMBO, alpha=0.9,
           edgecolor="white", label="Mean as partner")
    ax.set_xticks(xa); ax.set_xticklabels(d.Activation, rotation=45, ha="right", fontsize=9)
    ax.set_title(arch, fontsize=12, color=ARCH_C[arch])
    ax.set_ylim(85, 89)
    if ax is axes[0]:
        ax.set_ylabel("Test accuracy (%)", fontsize=12); ax.legend(fontsize=9)
fig.suptitle("Per-activation: accuracy used in isolation vs averaged over all mixed partners",
             fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_per_activation_isolated_vs_partner.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── console report ──────────────────────────────────────────────────────────
pd.set_option("display.width", 160, "display.max_columns", 20)
print("\n" + "=" * 78)
print("COMBINATION vs ISOLATED — per-architecture summary")
print("=" * 78)
print(summary[["Architecture", "iso_mean", "combo_mean", "combo_minus_iso",
               "winner_on_mean", "p_value", "significant_0.05",
               "iso_max", "combo_max"]].to_string(index=False))
print("\nBest configurations:")
for _, r in summary.iterrows():
    print(f"  {r.Architecture:<7}  isolated -> {r.best_isolated:<22} "
          f"combination -> {r.best_combination}")

overall = all_df.sort_values("Accuracy").iloc[-1]
print(f"\nOverall best across all architectures: "
      f"{overall.Arch}  {overall.Act1}+{overall.Act2}  {overall.Accuracy:.2f}%")
print("\nSaved figures + CSVs to:", OUT)
